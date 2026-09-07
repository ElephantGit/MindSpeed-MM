"""Paper-aligned Lance benchmark definitions.

The values here are a machine-readable contract.  They deliberately follow the
released benchmark shell scripts when those scripts and prose documentation
differ (notably VBench uses 30 steps and timestep shift 3.0).
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mindspeed_mm.models.omni.lance.checkpoint import (
    LanceCheckpointError,
    audit_checkpoint_metadata,
    read_safetensors_header,
)
from mindspeed_mm.models.omni.lance.native_config import LanceConfigError, LanceNativeConfig


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    task: str
    entrypoint: Optional[str]
    dataset_path: Optional[str]
    dataset_format: Optional[str]
    dataset_sha256: Optional[str]
    expected_samples: int
    samples_per_prompt: int
    num_inference_steps: Optional[int]
    timestep_shift: Optional[float]
    seed: int
    cfg_text_scale: Optional[float]
    cfg_interval: Optional[Tuple[float, float]]
    resolution: Optional[str]
    height: Optional[int]
    width: Optional[int]
    num_frames: Optional[int]
    fps: Optional[int]
    metric: str
    paper_score: float
    parity_tolerance: float
    scorer: str
    paper_comparability: str
    artifacts_per_prompt: Optional[int] = None
    extra_outputs: int = 0

    @property
    def expected_outputs(self) -> int:
        multiplier = self.artifacts_per_prompt or self.samples_per_prompt
        return self.expected_samples * multiplier + self.extra_outputs

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["expected_outputs"] = self.expected_outputs
        return result


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "geneval": BenchmarkSpec(
        name="geneval",
        task="t2i",
        entrypoint="benchmarks/image_gen/GenEVAL/sample_GenEVAL.py",
        dataset_path="benchmarks/image_gen/GenEVAL/GenEVAL.jsonl",
        dataset_format="jsonl",
        dataset_sha256="2b455ba7255c5289da8586f3547acc1da6590a022ea817122c2898718dfa703f",
        expected_samples=553,
        samples_per_prompt=4,
        num_inference_steps=50,
        timestep_shift=3.5,
        seed=42,
        cfg_text_scale=4.0,
        cfg_interval=(0.4, 1.0),
        resolution="image_768res",
        height=768,
        width=768,
        num_frames=1,
        fps=None,
        metric="overall",
        paper_score=0.90,
        parity_tolerance=0.02,
        scorer="official GenEval detector/evaluator",
        paper_comparability="Original GenEval, four 768x768 samples per prompt.",
    ),
    "dpgbench": BenchmarkSpec(
        name="dpgbench",
        task="t2i",
        entrypoint="benchmarks/image_gen/DPG/sample_DPG.py",
        dataset_path="benchmarks/image_gen/DPG/DPG.jsonl",
        dataset_format="jsonl",
        dataset_sha256="da4fb02b04b51ee42053a570bfc621def56be977e358c638b018c531dad2eee3",
        expected_samples=1065,
        samples_per_prompt=4,
        num_inference_steps=50,
        timestep_shift=3.5,
        seed=42,
        cfg_text_scale=4.0,
        cfg_interval=(0.4, 1.0),
        resolution="image_768res",
        height=768,
        width=768,
        num_frames=1,
        fps=None,
        metric="overall",
        paper_score=84.67,
        parity_tolerance=1.0,
        scorer="official DPG-Bench mPLUG evaluator",
        paper_comparability="Original DPG-Bench, four 768x768 samples per prompt.",
        artifacts_per_prompt=1,
    ),
    "gedit": BenchmarkSpec(
        name="gedit",
        task="image_edit",
        entrypoint="benchmarks/image_gen/GEdit/sample_GEdit.py",
        dataset_path="benchmarks/image_gen/GEdit/GEdit_en.json",
        dataset_format="json",
        dataset_sha256="5b33a3198106095374e7ab4e0c1c63e8107f4dc7cded4370c3d24c6de5c40e2e",
        expected_samples=606,
        samples_per_prompt=1,
        num_inference_steps=50,
        timestep_shift=3.5,
        seed=42,
        cfg_text_scale=4.0,
        cfg_interval=(0.4, 1.0),
        resolution="image_768res",
        height=None,
        width=None,
        num_frames=1,
        fps=None,
        metric="avg_g_o",
        paper_score=7.30,
        parity_tolerance=0.20,
        scorer="GEdit-Bench GPT-4.1 G_O judge",
        paper_comparability="Qwen offline judging is Q_O and must not be reported as paper G_O.",
    ),
    "vbench": BenchmarkSpec(
        name="vbench",
        task="t2v",
        entrypoint="benchmarks/video_gen/Vbench/sample_vbench.py",
        dataset_path="benchmarks/video_gen/Vbench/Vbench_recaption.jsonl",
        dataset_format="jsonl",
        dataset_sha256="a40ce0fc8a765d925e2dfe5a6205d19e8b1b158ca1a968918a4ec2f882450747",
        expected_samples=946,
        samples_per_prompt=5,
        num_inference_steps=30,
        timestep_shift=3.0,
        seed=42,
        cfg_text_scale=4.0,
        cfg_interval=(0.4, 1.0),
        resolution="video_480p",
        height=480,
        width=848,
        num_frames=50,
        fps=12,
        metric="total_score",
        paper_score=85.11,
        parity_tolerance=1.0,
        scorer="official VBench (MindSpeed-MM NPU patches supported)",
        paper_comparability=(
            "Uses the released frozen recaption file; five videos per regular prompt and "
            "25 for each of the 75 temporal-flickering prompts."
        ),
        extra_outputs=75 * (25 - 5),
    ),
    "mvbench": BenchmarkSpec(
        name="mvbench",
        task="x2t_video",
        entrypoint="inference_lance.py",
        dataset_path=None,
        dataset_format="json",
        dataset_sha256=None,
        expected_samples=3800,
        samples_per_prompt=1,
        num_inference_steps=None,
        timestep_shift=None,
        seed=42,
        cfg_text_scale=None,
        cfg_interval=None,
        resolution=None,
        height=None,
        width=None,
        num_frames=None,
        fps=None,
        metric="accuracy",
        paper_score=62.0,
        parity_tolerance=1.0,
        scorer="Lance paper MVBench 19-task multiple-choice evaluator",
        paper_comparability=(
            "Lance Table 8 reports 19 tasks (Fine-grained Pose is omitted), macro-averaged; "
            "the official complete 20-task result must be reported separately."
        ),
    ),
}

VBENCH_TEMPORAL_PROMPTS_SHA256 = "bedc9e5a6fbfd0a9ce78baa9a377f16e44ae36b0e161c147cbc0eab033d0ea05"
PINNED_LANCE_REVISION = "4baeee086648996f6ab12e673cbe461b0b149997"


def get_benchmark(name: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[name.lower()]
    except KeyError as exc:
        raise ValueError("Unknown Lance benchmark: {}".format(name)) from exc


def _count_records(path: Path, dataset_format: str) -> int:
    if dataset_format == "jsonl":
        with path.open("r", encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    if dataset_format == "json":
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, (list, dict)):
            raise ValueError("JSON benchmark data must be a list or object")
        return len(value)
    raise ValueError("Unsupported dataset format: {}".format(dataset_format))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(
    spec: BenchmarkSpec,
    lance_source_root: Union[str, Path],
    dataset_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    if (spec.dataset_path is None and dataset_path is None) or spec.dataset_format is None:
        return {
            "benchmark": spec.name,
            "status": "external-adapter-required",
            "issues": ["No released Lance sampling adapter is available for this benchmark."],
        }
    path = Path(dataset_path) if dataset_path else Path(lance_source_root) / spec.dataset_path
    path = path.expanduser().resolve()
    issues: List[str] = []
    count = None
    digest = None
    auxiliary = []
    if not path.is_file():
        issues.append("dataset does not exist: {}".format(path))
    else:
        try:
            count = _count_records(path, spec.dataset_format)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append("dataset cannot be read: {}".format(exc))
        if count is not None and count != spec.expected_samples:
            issues.append("expected {} samples, found {}".format(spec.expected_samples, count))
        digest = sha256_file(path)
        if dataset_path is None and spec.dataset_sha256 and digest != spec.dataset_sha256:
            issues.append("released dataset SHA-256 does not match the pinned protocol")
    if spec.name == "vbench":
        temporal_path = (
            Path(lance_source_root)
            / "benchmarks/video_gen/Vbench/temporal_flickering_prompts.json"
        ).expanduser().resolve()
        temporal_record = {"path": str(temporal_path), "samples": None, "sha256": None}
        if not temporal_path.is_file():
            issues.append("VBench temporal-flickering prompt file is missing")
        else:
            try:
                with temporal_path.open("r", encoding="utf-8") as stream:
                    temporal_prompts = json.load(stream)
                temporal_record["samples"] = len(temporal_prompts) if isinstance(temporal_prompts, list) else None
                temporal_record["sha256"] = sha256_file(temporal_path)
                if temporal_record["samples"] != 75:
                    issues.append("VBench requires exactly 75 temporal-flickering prompts")
                if temporal_record["sha256"] != VBENCH_TEMPORAL_PROMPTS_SHA256:
                    issues.append("VBench temporal-flickering prompt SHA-256 does not match")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues.append("VBench temporal-flickering prompt file cannot be read: {}".format(exc))
        auxiliary.append(temporal_record)
    return {
        "benchmark": spec.name,
        "status": "valid" if not issues else "invalid",
        "path": str(path),
        "samples": count,
        "expected_samples": spec.expected_samples,
        "expected_outputs": spec.expected_outputs,
        "sha256": digest,
        "auxiliary": auxiliary,
        "issues": issues,
    }


def validate_model_path(
    model_path: Union[str, Path],
    fingerprint: bool = False,
    variant: Optional[str] = None,
) -> Dict[str, object]:
    path = Path(model_path).expanduser().resolve()
    required = ["llm_config.json", "vocab.json", "merges.txt"]
    missing = [name for name in required if not (path / name).is_file()]
    checkpoint_files = [path / "model.safetensors", path / "ema.safetensors"]
    if not any(item.is_file() for item in checkpoint_files):
        missing.append("model.safetensors|ema.safetensors")
    checkpoint = next((item for item in checkpoint_files if item.is_file()), None)
    checkpoint_contract = None
    if checkpoint and variant:
        try:
            config_path = path / "llm_config.json"
            config = LanceNativeConfig.from_llm_config(config_path, variant=variant)
            checkpoint_contract = audit_checkpoint_metadata(
                read_safetensors_header(checkpoint),
                config,
            )
            if not checkpoint_contract["valid"]:
                missing.append("checkpoint-contract:{}".format(variant))
        except (LanceCheckpointError, LanceConfigError) as exc:
            checkpoint_contract = {
                "status": "invalid",
                "valid": False,
                "variant": variant,
                "error": str(exc),
            }
            missing.append("checkpoint-contract:{}".format(variant))
    return {
        "path": str(path),
        "status": "valid" if not missing else "invalid",
        "missing": missing,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_size": checkpoint.stat().st_size if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint and fingerprint else None,
        "checkpoint_contract": checkpoint_contract,
    }


def _resolve_configured_path(source_root: Path, value: str, base_dir: str) -> Path:
    rendered = value.replace("${base_dir}", base_dir)
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (source_root / path).resolve()


def validate_runtime_assets(
    lance_source_root: Union[str, Path],
    require_vae: bool = True,
) -> Dict[str, object]:
    """Validate ViT/VAE files resolved by Lance's path_default.yaml."""
    source = Path(lance_source_root).expanduser().resolve()
    config_path = source / "config/path_default.yaml"
    issues = []
    assets: Dict[str, object] = {}
    if not config_path.is_file():
        return {"status": "invalid", "config": str(config_path), "assets": assets,
                "issues": ["path_default.yaml is missing"]}
    with config_path.open("r", encoding="utf-8") as stream:
        config_text = stream.read()
    try:
        import yaml

        config = yaml.safe_load(config_text)
    except ImportError:
        # path_default.yaml only uses nested string mappings.  Keeping this tiny
        # fallback lets preflight run on login nodes before the training env is
        # activated; model execution still requires the full dependencies.
        config = {}
        section = None
        for raw_line in config_text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip() or ":" not in line:
                continue
            indent = len(line) - len(line.lstrip())
            key, value = line.strip().split(":", 1)
            value = value.strip().strip('"').strip("'")
            if indent == 0:
                section = key
                config[key] = value if value else {}
            elif section and isinstance(config.get(section), dict):
                config[section][key] = value
    base_dir = str(config.get("base_dir", "downloads"))
    values = {"vit": config.get("vit", {}).get("qwen2_5_vl")}
    if require_vae:
        values["vae"] = config.get("vae", {}).get("wan")
    for name, value in values.items():
        if not value:
            issues.append("{} path is not configured".format(name))
            continue
        path = _resolve_configured_path(source, str(value), base_dir)
        required = [path / "config.json", path / "vit.safetensors"] if name == "vit" else [path]
        missing = [str(item) for item in required if not item.is_file()]
        assets[name] = {"path": str(path), "missing": missing}
        if missing:
            issues.extend("missing {} asset: {}".format(name, item) for item in missing)
    return {
        "status": "valid" if not issues else "invalid",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "assets": assets,
        "issues": issues,
    }


def lance_source_revision(lance_source_root: Union[str, Path]) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(Path(lance_source_root).resolve()), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_lance_source(lance_source_root: Union[str, Path]) -> Dict[str, object]:
    source = Path(lance_source_root).expanduser().resolve()
    revision = lance_source_revision(source)
    issues = []
    if revision is None:
        issues.append("Lance source is not a readable Git checkout")
    elif revision != PINNED_LANCE_REVISION:
        issues.append(
            "Lance source revision {} does not match pinned {}".format(
                revision, PINNED_LANCE_REVISION
            )
        )
    dirty_result = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = bool(dirty_result.stdout.strip()) if dirty_result.returncode == 0 else None
    if dirty:
        issues.append("Lance source checkout has uncommitted changes")
    return {
        "path": str(source),
        "status": "valid" if not issues else "invalid",
        "revision": revision,
        "pinned_revision": PINNED_LANCE_REVISION,
        "dirty": dirty,
        "issues": issues,
    }


def build_sample_arguments(
    spec: BenchmarkSpec,
    model_path: Union[str, Path],
    dataset_path: Union[str, Path],
    output_path: Union[str, Path],
    world_size: int = 8,
) -> List[str]:
    if world_size < 1:
        raise ValueError("world_size must be positive")

    if spec.name == "mvbench":
        return [
            "--model_path", str(model_path),
            "--val_dataset_config_file", str(dataset_path),
            "--vit_type", "qwen_2_5_vl_original",
            "--llm_qk_norm", "true",
            "--llm_qk_norm_und", "true",
            "--llm_qk_norm_gen", "true",
            "--tie_word_embeddings", "false",
            "--copy_init_moe", "true",
            "--max_num_frames", "121",
            "--max_latent_size", "64",
            "--latent_patch_size", "1", "1", "1",
            "--visual_und", "true",
            "--visual_gen", "false",
            "--apply_qwen_2_5_vl_pos_emb", "true",
            "--apply_chat_template", "false",
            "--validation_data_seed", str(spec.seed),
            "--validation_max_samples", str(spec.expected_samples),
            "--task", spec.task,
            "--save_path_gen", str(output_path),
            "--resolution", "video_480p",
            "--text_template", "true",
            "--use_KVcache", "true",
        ]

    arguments = [
        "--model_path", str(model_path),
        "--val_dataset_config_file", str(dataset_path),
        "--vit_type", "qwen_2_5_vl_original",
        "--llm_qk_norm", "true",
        "--llm_qk_norm_und", "true",
        "--llm_qk_norm_gen", "true",
        "--tie_word_embeddings", "false",
        "--validation_num_timesteps", str(spec.num_inference_steps),
        "--validation_timestep_shift", str(spec.timestep_shift),
        "--copy_init_moe", "true",
        "--max_num_frames", "121" if spec.task == "t2v" else "1",
        "--max_latent_size", "64",
        "--latent_patch_size", "1", "1", "1",
        "--visual_und", "true",
        "--visual_gen", "true",
        "--vae_model_type", "wan",
        "--apply_qwen_2_5_vl_pos_emb", "true",
        "--apply_chat_template", "false",
        "--cfg_type", "0",
        "--task", spec.task,
        "--save_path_gen", str(output_path),
        "--resolution", str(spec.resolution),
        "--text_template", "true",
        "--sample_num_per_prompt", str(spec.samples_per_prompt),
        "--cfg_text_scale", str(spec.cfg_text_scale),
        "--cfg_interval", str(spec.cfg_interval[0]), str(spec.cfg_interval[1]),
        "--use_KVcache", "true",
    ]
    seed_flag = "--evaluation_seed" if spec.name == "vbench" else "--validation_data_seed"
    arguments.extend([seed_flag, str(spec.seed)])
    if spec.height is not None:
        arguments.extend(["--video_height", str(spec.height)])
    if spec.width is not None:
        arguments.extend(["--video_width", str(spec.width)])
    if spec.num_frames is not None:
        arguments.extend(["--num_frames", str(spec.num_frames)])
    if spec.fps is not None:
        arguments.extend(["--validation_video_saving_fps", str(spec.fps), "--validation_log_type", "direct"])
    if spec.name in {"dpgbench", "gedit", "vbench"}:
        arguments.extend(["--use_flex", "true", "--num_replicate", str(world_size), "--num_shard", "1"])
    if spec.name == "gedit":
        arguments.extend(["--validation_max_samples", "100000"])
    return arguments


def protocol_manifest(names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    selected = list(names) if names is not None else list(BENCHMARKS)
    return {
        "schema_version": 1,
        "lance_source_revision": PINNED_LANCE_REVISION,
        "benchmarks": {name: get_benchmark(name).to_dict() for name in selected},
        "notes": [
            "Paper scores refer to the released final checkpoint/post-training recipe, not PT-only checkpoints.",
            "VBench parameters follow sample_vbench.sh (30 steps, shift 3.0), not the conflicting README prose.",
            "Lance Table 8 contains 19 MVBench tasks and omits Fine-grained Pose; its 62.0 is not the official 20-task score.",
            "A score is comparable only when the stated scorer and prompt protocol are preserved.",
        ],
    }


def audit_outputs(spec: BenchmarkSpec, output_path: Union[str, Path]) -> Dict[str, object]:
    """Check that a sampling run produced the scorer-facing artifact count."""
    root = Path(output_path).expanduser().resolve()
    if spec.name == "geneval":
        pattern = "*/samples/*.png"
        expected = spec.expected_outputs
    elif spec.name == "dpgbench":
        # Each PNG is the official 2x2 grid containing four generations.
        pattern = "*.png"
        expected = spec.expected_outputs
    elif spec.name == "gedit":
        pattern = "fullset/*/*/*.webp"
        expected = spec.expected_outputs
    elif spec.name == "vbench":
        pattern = "*.mp4"
        expected = spec.expected_outputs
    elif spec.name == "mvbench":
        result_path = root if root.is_file() else root / "result.json"
        issues = []
        count = None
        if not result_path.is_file():
            issues.append("result.json does not exist: {}".format(result_path))
        else:
            try:
                with result_path.open("r", encoding="utf-8") as stream:
                    payload = json.load(stream)
                count = len(payload) if isinstance(payload, list) else None
                if count is None:
                    issues.append("MVBench result.json must contain a list")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues.append("MVBench result.json cannot be read: {}".format(exc))
        if count is not None and count != spec.expected_outputs:
            issues.append("expected {} results, found {}".format(spec.expected_outputs, count))
        return {
            "benchmark": spec.name,
            "status": "valid" if not issues else "invalid",
            "path": str(result_path),
            "artifacts": count,
            "expected_artifacts": spec.expected_outputs,
            "issues": issues,
        }
    else:
        raise ValueError("Unsupported benchmark: {}".format(spec.name))

    artifacts = sorted(root.glob(pattern)) if root.is_dir() else []
    issues = []
    if not root.is_dir():
        issues.append("output directory does not exist: {}".format(root))
    if len(artifacts) != expected:
        issues.append("expected {} artifacts matching {}, found {}".format(expected, pattern, len(artifacts)))
    return {
        "benchmark": spec.name,
        "status": "valid" if not issues else "invalid",
        "path": str(root),
        "pattern": pattern,
        "artifacts": len(artifacts),
        "expected_artifacts": expected,
        "issues": issues,
    }
