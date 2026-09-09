import importlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance.ascend_runtime import (
    _install_flash_attn_shim,
    _patch_device_mesh,
    _patch_transformers_flash_attn_probe,
    cumulative_lengths,
)
from mindspeed_mm.models.omni.lance.runner import LanceSourceError, resolve_entrypoint
from mindspeed_mm.tasks.evaluation.lance.protocol import (
    BENCHMARKS,
    PINNED_LANCE_REVISION,
    audit_outputs,
    build_sample_arguments,
    validate_model_path,
    validate_dataset,
)
from mindspeed_mm.tasks.evaluation.lance.report import build_alignment_report
from mindspeed_mm.tasks.evaluation.lance.scores import PINNED_SCORER_REVISIONS


class FakeCumulativeLengths:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def to(self, device):
        assert device == "cpu"
        return self

    def tolist(self):
        return self.values


def test_paper_protocol_is_pinned_to_released_sampler_values():
    assert BENCHMARKS["geneval"].paper_score == 0.90
    assert BENCHMARKS["dpgbench"].paper_score == 84.67
    assert BENCHMARKS["gedit"].paper_score == 7.30
    assert BENCHMARKS["vbench"].paper_score == 85.11
    assert BENCHMARKS["mvbench"].paper_score == 62.0
    assert BENCHMARKS["mvbench"].expected_samples == 3800
    assert BENCHMARKS["vbench"].num_inference_steps == 30
    assert BENCHMARKS["vbench"].timestep_shift == 3.0
    assert BENCHMARKS["vbench"].expected_outputs == 946 * 5 + 75 * 20
    assert BENCHMARKS["dpgbench"].expected_outputs == 1065
    assert len(PINNED_LANCE_REVISION) == 40


def test_sample_arguments_preserve_geneval_contract():
    args = build_sample_arguments(
        BENCHMARKS["geneval"],
        model_path="/checkpoint",
        dataset_path="/dataset.jsonl",
        output_path="/output",
    )
    joined = " ".join(args)
    assert "--validation_num_timesteps 50" in joined
    assert "--validation_timestep_shift 3.5" in joined
    assert "--sample_num_per_prompt 4" in joined
    assert "--video_height 768 --video_width 768" in joined
    assert "--use_KVcache true" in joined


def test_dataset_validation_rejects_wrong_cardinality(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"index": 0}\n', encoding="utf-8")
    result = validate_dataset(BENCHMARKS["geneval"], tmp_path, dataset)
    assert result["status"] == "invalid"
    assert "expected 553 samples, found 1" in result["issues"]


def test_alignment_report_requires_metric_provenance():
    payload = {
        "scores": {
            "geneval": 0.90,
            "dpgbench": 84.67,
            "gedit": 7.30,
            "vbench": 85.11,
            "mvbench": 62.0,
        },
        "provenance": {
            "geneval": {"scorer": "official-geneval"},
            "dpgbench": {"scorer": "official-mplug"},
            "gedit": {"judge": "Qwen2.5-VL-72B"},
            "vbench": {"scorer": "official-vbench", "released_recaption": False},
            "mvbench": {
                "tasks": 19,
                "samples": 3800,
                "task_set": "paper-lance-19",
                "aggregation": "macro-task-accuracy",
            },
        },
    }
    report = build_alignment_report(payload)
    assert report["complete"] is True
    assert report["paper_comparable"] is False
    assert report["all_within_tolerance"] is False
    assert report["results"]["gedit"]["comparability_issue"]


def test_alignment_report_accepts_exact_protocol():
    common = {
        "run_manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "results_sha256": "d" * 64,
        "lance_revision": PINNED_LANCE_REVISION,
    }
    payload = {
        "scores": {
            "geneval": 0.90,
            "dpgbench": 84.67,
            "gedit": 7.30,
            "vbench": 85.11,
            "mvbench": 62.0,
        },
        "provenance": {
            "geneval": dict(common, scorer="official-geneval",
                            scorer_revision=PINNED_SCORER_REVISIONS["geneval"],
                            images=2212, prompts=553, tasks=6,
                            aggregation="macro-task-accuracy"),
            "dpgbench": dict(common, scorer="official-mplug",
                              scorer_revision=PINNED_SCORER_REVISIONS["dpgbench"],
                              grids=1065, images_per_prompt=4),
            "gedit": dict(common, judge="gpt-4.1",
                          scorer_revision=PINNED_SCORER_REVISIONS["gedit"],
                          samples=606, groups=11, language="en",
                          aggregation="macro-group-mean-geometric-score"),
            "vbench": dict(common, scorer="official-vbench", scorer_version="0.1.2",
                           released_recaption=True, prompts=946, videos=6230, dimensions=16),
            "mvbench": {
                **common,
                "tasks": 19,
                "samples": 3800,
                "task_set": "paper-lance-19",
                "aggregation": "macro-task-accuracy",
                "answer_extraction": "official-leading-option",
                "metadata_sha256": "e" * 64,
            },
        },
    }
    report = build_alignment_report(payload)
    assert report["all_within_tolerance"] is True


def test_cumulative_lengths_contract():
    assert cumulative_lengths(FakeCumulativeLengths([0, 4, 9])) == (4, 9)


def test_npu_flash_attention_shim_uses_tnd_and_causal_mode():
    calls = []

    class FakeTensor:
        ndim = 3
        shape = (9, 16, 128)
        device = "npu:0"

        def new_ones(self, shape, dtype):
            assert shape == (2048, 2048)
            assert dtype == "bool"
            return self

        def triu(self, diagonal):
            assert diagonal == 1
            return "causal-mask"

    class FakeTorchNpu:
        @staticmethod
        def npu_fusion_attention(q, k, v, **kwargs):
            calls.append(kwargs)
            return ("output",)

    class FakeTorch:
        bool = "bool"

        @staticmethod
        def stack(values, dim=-1):
            raise AssertionError("rotary path is not used in this test")

    old_modules = {name: sys.modules.get(name) for name in (
        "flash_attn", "flash_attn.layers", "flash_attn.layers.rotary"
    )}
    try:
        _install_flash_attn_shim(FakeTorch, FakeTorchNpu)
        output = sys.modules["flash_attn"].flash_attn_varlen_func(
            FakeTensor(), FakeTensor(), FakeTensor(),
            FakeCumulativeLengths([0, 4, 9]),
            FakeCumulativeLengths([0, 6, 12]),
            5, 7, causal=True,
        )
        try:
            sys.modules["flash_attn"].flash_attn_varlen_func(
                FakeTensor(), FakeTensor(), FakeTensor(),
                FakeCumulativeLengths([0, 4, 9]),
                FakeCumulativeLengths([0, 6, 12]),
                5, 7, dropout_p=0.25,
            )
        except ValueError as exc:
            assert "inference requires" in str(exc)
        else:
            raise AssertionError("inference attention dropout must be rejected")

        _install_flash_attn_shim(FakeTorch, FakeTorchNpu, allow_dropout=True)
        training_output = sys.modules["flash_attn"].flash_attn_varlen_func(
            FakeTensor(), FakeTensor(), FakeTensor(),
            FakeCumulativeLengths([0, 4, 9]),
            FakeCumulativeLengths([0, 6, 12]),
            5, 7, dropout_p=0.25,
        )
    finally:
        for name, value in old_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    assert output == "output"
    assert training_output == "output"
    assert calls[0]["input_layout"] == "TND"
    assert calls[0]["actual_seq_qlen"] == (4, 9)
    assert calls[0]["actual_seq_kvlen"] == (6, 12)
    assert calls[0]["sparse_mode"] == 3
    assert calls[0]["atten_mask"] == "causal-mask"
    assert calls[1]["keep_prob"] == 0.75


def test_transformers_flash_attention_probe_accepts_process_local_shim(monkeypatch):
    class FakeTransformersUtils:
        @staticmethod
        def is_flash_attn_2_available():
            return False

    original_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "transformers.utils":
            return FakeTransformersUtils
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    _patch_transformers_flash_attn_probe()

    assert FakeTransformersUtils.is_flash_attn_2_available() is True
    assert FakeTransformersUtils.is_flash_attn_2_available._lance_npu_compatible is True


def test_training_device_mesh_maps_cuda_to_npu(monkeypatch):
    calls = []

    class FakeDeviceMeshModule:
        @staticmethod
        def init_device_mesh(device_type, *args, **kwargs):
            calls.append((device_type, args, kwargs))
            return "mesh"

    original_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "torch.distributed.device_mesh":
            return FakeDeviceMeshModule
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    _patch_device_mesh()
    assert FakeDeviceMeshModule.init_device_mesh(
        "cuda", mesh_shape=(1, 8), mesh_dim_names=("replicate", "shard")
    ) == "mesh"
    assert calls == [
        ("npu", (), {"mesh_shape": (1, 8), "mesh_dim_names": ("replicate", "shard")})
    ]


def test_entrypoint_cannot_escape_source_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    try:
        resolve_entrypoint(source, "../outside.py")
    except LanceSourceError:
        pass
    else:
        raise AssertionError("path traversal must be rejected")


def test_output_audit_understands_dpg_grid_contract(tmp_path):
    for index in range(BENCHMARKS["dpgbench"].expected_samples):
        (tmp_path / (str(index) + ".png")).touch()
    result = audit_outputs(BENCHMARKS["dpgbench"], tmp_path)
    assert result["status"] == "valid"
    assert result["artifacts"] == 1065


def test_model_validation_accepts_either_released_checkpoint_name(tmp_path):
    (tmp_path / "llm_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "merges.txt").write_text("", encoding="utf-8")
    (tmp_path / "ema.safetensors").write_bytes(b"checkpoint")
    result = validate_model_path(tmp_path, fingerprint=True)
    assert result["status"] == "valid"
    assert result["checkpoint"].endswith("ema.safetensors")
    assert len(result["checkpoint_sha256"]) == 64


def test_model_validation_rejects_truncated_checkpoint_when_variant_is_known(tmp_path):
    (tmp_path / "llm_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "merges.txt").write_text("", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"header-only")
    result = validate_model_path(tmp_path, variant="image")
    assert result["status"] == "invalid"
    assert "checkpoint-contract:image" in result["missing"]
    assert result["checkpoint_contract"]["valid"] is False
