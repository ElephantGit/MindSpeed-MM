import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/check_lance_native_artifacts.py"
TRACE_SCRIPT = REPO_ROOT / "scripts/compare_lance_native_traces.py"


def _artifacts(tmp_path, batches=1):
    config = {
        "text_config": {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 11008,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "rope_scaling": {"mrope_section": [16, 24, 24]},
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    load = tmp_path / "load"
    (load / "release").mkdir(parents=True)
    (load / "release/.metadata").touch()
    (load / "latest_checkpointed_iteration.txt").write_text("release", encoding="utf-8")
    native_config = {
        "vocab_size": 151936,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "latent_channels": 48,
        "latent_patch_size": [1, 1, 1],
        "max_latent_size": 64,
        "max_num_frames": 121,
    }
    (load / "lance_native_initialization.json").write_text(
        json.dumps({
            "initialization": {
                "config": native_config,
                "effective_vocab_size": 151665,
            }
        }),
        encoding="utf-8",
    )
    data = tmp_path / "data"
    data.mkdir()
    for index in range(batches):
        (data / "batch-{:08d}.pt".format(index)).touch()
    (data / "manifest.json").write_text(
        json.dumps({
            "status": "completed", "variant": "video",
            "batch_count": batches, "total_tokens": 10,
            "skipped_sample_count": 0,
            "config": native_config,
            "effective_vocab_size": 151665,
        }),
        encoding="utf-8",
    )
    return config_path, load, data


def test_native_preflight_accepts_complete_release(tmp_path):
    config, load, data = _artifacts(tmp_path, batches=2)
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--load", str(load), "--data", str(data),
            "--llm-config", str(config), "--variant", "video",
            "--effective-vocab-size", "151665", "--world-size", "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["status"] == "valid"


def test_native_preflight_rejects_too_few_batches(tmp_path):
    config, load, data = _artifacts(tmp_path, batches=1)
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--load", str(load), "--data", str(data),
            "--llm-config", str(config), "--variant", "video",
            "--effective-vocab-size", "151665", "--world-size", "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "at least one each" in result.stderr


def test_native_preflight_uses_explicit_latent_geometry(tmp_path):
    config, load, data = _artifacts(tmp_path, batches=1)
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--load", str(load), "--data", str(data),
            "--llm-config", str(config), "--variant", "video",
            "--latent-patch-size", "1", "2", "2",
            "--max-latent-size", "64", "--max-num-frames", "121",
            "--effective-vocab-size", "151665", "--world-size", "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "configurations differ" in result.stderr


def _write_trace(base, *, loss_delta=0.0):
    path = base.with_name(base.stem + ".rank00000" + base.suffix)
    records = [
        {
            "rank": 0,
            "iteration": iteration,
            "consumed_train_samples": iteration,
            "batch_paths": ["batch-{}.pt".format(iteration)],
            "loss": 1.0 / iteration + loss_delta,
            "grad_norm": 0.5,
            "learning_rate": 1e-4,
            "metrics": {
                "ce": 1.0, "mse": 2.0,
                "ce_tokens": 3.0, "mse_tokens": 4.0,
            },
            "parameter_checksum": {
                "sum": 10.0, "square_sum": 20.0,
                "global_numel": 30, "max_abs": 0.5,
            },
        }
        for iteration in (1, 2)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_native_trace_comparator_checks_resume_continuity(tmp_path):
    continuous = tmp_path / "continuous.jsonl"
    resumed = tmp_path / "resumed.jsonl"
    _write_trace(continuous)
    _write_trace(resumed)
    valid = subprocess.run(
        [
            sys.executable, str(TRACE_SCRIPT),
            "--continuous", str(continuous), "--resumed", str(resumed),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(valid.stdout)["status"] == "valid"

    _write_trace(tmp_path / "different.jsonl", loss_delta=0.1)
    invalid = subprocess.run(
        [
            sys.executable, str(TRACE_SCRIPT),
            "--continuous", str(continuous),
            "--resumed", str(tmp_path / "different.jsonl"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 1
