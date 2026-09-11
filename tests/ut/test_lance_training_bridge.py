import json
import os
from pathlib import Path

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance.training_bridge import (
    UPSTREAM_TRAINING_FILES,
    sha256_file,
    validate_forwarded_training_arguments,
    validate_training_manifest,
    validate_upstream_training_source,
)
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.training_contract import STAGES, training_manifest
from mindspeed_mm.models.omni.lance.upstream_training import (
    compile_strict_training_entrypoint,
    validate_strict_training_entrypoint,
)


def _source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "Lance"
    for relative in UPSTREAM_TRAINING_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test fixture\n", encoding="utf-8")
    return source


def _prepared_manifest(tmp_path: Path, source: Path, init_mode: str = "qwen2_5_vl"):
    dataset = source / "config" / "train_local" / "pt.yaml"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("D1: {}\n", encoding="utf-8")
    init_path = tmp_path / "initialization"
    init_path.mkdir()
    manifest = training_manifest("pt", init_mode, world_size=8)
    manifest["model"] = LanceNativeConfig.for_variant("video").to_dict()
    manifest["initialization"]["path"] = str(init_path) if init_mode != "random" else None
    manifest["dataset_manifests"] = [
        {
            "path": str(dataset),
            "bytes": dataset.stat().st_size,
            "sha256": sha256_file(dataset),
        }
    ]
    manifest["status"] = "ready-for-runtime-preflight"
    path = tmp_path / "training.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, dataset, init_path


def _paper_arguments(dataset: Path, init_path: Path):
    stage = STAGES["pt"]
    return [
        "--llm_path", str(init_path),
        "--vit_path", str(init_path),
        "--init_from_vlm_checkpoint", "true",
        "--load_from_lance_checkpoint", "false",
        "--copy_init_moe", "true",
        "--layer_module", "Qwen2MoTDecoderLayer",
        "--vit_type", "qwen2_5_vl",
        "--vae_model_type", "wan",
        "--max_num_frames", "121",
        "--max_latent_size", "64",
        "--latent_patch_size", "1", "1", "1",
        "--visual_gen", "true",
        "--visual_und", "true",
        "--freeze_vit", "true",
        "--freeze_vae", "true",
        "--freeze_llm", "false",
        "--freeze_llm_embed_tokens", "false",
        "--freeze_vit_connector", "false",
        "--freeze_und_params", "false",
        "--freeze_und", "false",
        "--use_ema", "true",
        "--use_flex", "true",
        "--cpu_offload", "false",
        "--sharding_strategy", "HYBRID_SHARD",
        "--backward_prefetch", "BACKWARD_PRE",
        "--dataset_config_file", str(dataset),
        "--total_steps", str(stage.steps),
        "--warmup_steps", str(stage.warmup_steps),
        "--lr", str(stage.learning_rate),
        "--lr_scheduler", stage.scheduler,
        "--expected_num_tokens", str(stage.expected_tokens_per_rank),
        "--max_num_tokens", str(stage.max_tokens_per_rank),
        "--max_num_tokens_per_sample", str(stage.max_context),
        "--timestep_shift", str(stage.timestep_shift),
        "--ce_weight", str(stage.ce_weight),
        "--mse_weight", str(stage.mse_weight),
        "--text_cond_dropout_prob", str(stage.text_dropout),
        "--vae_cond_dropout_prob", str(stage.multimodal_dropout),
        "--vit_cond_dropout_prob", str(stage.multimodal_dropout),
        "--global_seed", "2025",
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--eps", "1e-15",
        "--max_grad_norm", "1.0",
        "--ema", "0.9999",
        "--min_lr", "1e-7",
        "--num_replicate", "1",
        "--num_shard", "8",
    ]


def test_training_manifest_revalidates_dataset_digest(tmp_path):
    source = _source_tree(tmp_path)
    manifest, dataset, _ = _prepared_manifest(tmp_path, source)
    assert validate_training_manifest(manifest, source)["status"] == "valid"

    dataset.write_text("D1: changed\n", encoding="utf-8")
    result = validate_training_manifest(manifest, source)
    assert result["status"] == "invalid"
    assert "changed after preparation" in " ".join(result["issues"])


def test_forwarded_paper_arguments_match_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    source = _source_tree(tmp_path)
    manifest, dataset, init_path = _prepared_manifest(tmp_path, source)
    manifest_result = validate_training_manifest(manifest, source)
    result = validate_forwarded_training_arguments(
        _paper_arguments(dataset, init_path), manifest_result, source
    )
    assert result["status"] == "valid"
    assert not result["warnings"]


def test_forwarded_arguments_require_model_geometry(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    source = _source_tree(tmp_path)
    manifest, dataset, init_path = _prepared_manifest(tmp_path, source)
    manifest_result = validate_training_manifest(manifest, source)
    arguments = _paper_arguments(dataset, init_path)
    patch_index = arguments.index("--latent_patch_size")
    del arguments[patch_index:patch_index + 4]

    result = validate_forwarded_training_arguments(arguments, manifest_result, source)

    assert result["status"] == "invalid"
    assert "critical upstream argument is missing: --latent_patch_size" in result["issues"]


def test_smoke_test_allows_only_coherent_reductions(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    source = _source_tree(tmp_path)
    manifest, dataset, init_path = _prepared_manifest(tmp_path, source)
    manifest_result = validate_training_manifest(manifest, source)
    arguments = _paper_arguments(dataset, init_path)
    overrides = {
        "--total_steps": "20",
        "--warmup_steps": "2",
        "--expected_num_tokens": "1024",
        "--max_num_tokens": "1280",
        "--max_num_tokens_per_sample": "768",
    }
    for name, value in overrides.items():
        arguments[arguments.index(name) + 1] = value

    strict = validate_forwarded_training_arguments(arguments, manifest_result, source)
    assert strict["status"] == "invalid"
    smoke = validate_forwarded_training_arguments(
        arguments, manifest_result, source, smoke_test=True
    )
    assert smoke["status"] == "valid"
    assert len(smoke["warnings"]) == len(overrides)

    arguments[arguments.index("--warmup_steps") + 1] = "20"
    incoherent = validate_forwarded_training_arguments(
        arguments, manifest_result, source, smoke_test=True
    )
    assert incoherent["status"] == "invalid"
    assert "smaller than --total_steps" in " ".join(incoherent["issues"])


def test_understanding_only_smoke_requires_visual_generation_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    source = _source_tree(tmp_path)
    manifest, dataset, init_path = _prepared_manifest(tmp_path, source)
    i2t_dataset = dataset.with_name("i2t_local.yaml")
    dataset.rename(i2t_dataset)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dataset_manifests"][0] = {
        "path": str(i2t_dataset),
        "bytes": i2t_dataset.stat().st_size,
        "sha256": sha256_file(i2t_dataset),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest_result = validate_training_manifest(manifest, source)

    arguments = _paper_arguments(i2t_dataset, init_path)
    strict_result = validate_forwarded_training_arguments(
        arguments,
        manifest_result,
        source,
        smoke_test=True,
    )
    assert strict_result["status"] == "invalid"
    assert "--visual_gen=True differs from bridge contract False" in strict_result["issues"]

    arguments[arguments.index("--visual_gen") + 1] = "false"
    smoke_result = validate_forwarded_training_arguments(
        arguments,
        manifest_result,
        source,
        smoke_test=True,
    )
    assert smoke_result["status"] == "valid"
    assert smoke_result["smoke_profile"] == "understanding-only"
    assert "understanding-only smoke test overrides --visual_gen" in " ".join(
        smoke_result["warnings"]
    )


def test_upstream_bridge_rejects_random_initialization(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    source = _source_tree(tmp_path)
    manifest, dataset, init_path = _prepared_manifest(tmp_path, source, "random")
    manifest_result = validate_training_manifest(manifest, source)
    result = validate_forwarded_training_arguments(
        _paper_arguments(dataset, init_path), manifest_result, source
    )
    assert result["status"] == "invalid"
    assert "no strict-random initialization path" in " ".join(result["issues"])


def test_upstream_training_source_requires_clean_complete_checkout(tmp_path, monkeypatch):
    source = _source_tree(tmp_path)

    def fake_git_value(_source, *arguments):
        return "a" * 40 if arguments == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(
        "mindspeed_mm.models.omni.lance.training_bridge._git_value",
        fake_git_value,
    )
    assert validate_upstream_training_source(source)["status"] == "valid"

    (source / UPSTREAM_TRAINING_FILES[-1]).unlink()
    result = validate_upstream_training_source(source)
    assert result["status"] == "invalid"
    assert result["missing"] == [UPSTREAM_TRAINING_FILES[-1]]


def test_strict_entrypoint_re_raises_the_released_step_exception(tmp_path):
    entrypoint = tmp_path / "unified_train.py"
    entrypoint.write_text(
        """
def main():
    for curr_step in range(1):
        try:
            raise ValueError("bad batch")
        except Exception:
            print(f"[TRAINING EXCEPTION] Step {curr_step}")
            try:
                dist.barrier()
            except:
                pass
            continue

if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )

    validation = validate_strict_training_entrypoint(entrypoint)
    assert validation["status"] == "valid"
    assert validation["upstream_file_modified"] is False
    code, _ = compile_strict_training_entrypoint(entrypoint)
    try:
        exec(code, {"__name__": "__main__", "__file__": str(entrypoint)})
    except ValueError as exc:
        assert str(exc) == "bad batch"
    else:
        raise AssertionError("the upstream step exception must be re-raised")


def test_strict_entrypoint_rejects_an_unknown_handler_shape(tmp_path):
    entrypoint = tmp_path / "unified_train.py"
    entrypoint.write_text(
        "try:\n    pass\nexcept Exception:\n    print('[TRAINING EXCEPTION]')\n",
        encoding="utf-8",
    )
    result = validate_strict_training_entrypoint(entrypoint)
    assert result["status"] == "invalid"
    assert "changed shape" in " ".join(result["issues"])
