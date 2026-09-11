import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "pretrain_lance_pt.sh"


def _argument_value(arguments, name):
    return arguments[arguments.index(name) + 1]


def _run_smoke(
    tmp_path,
    *,
    config_name="i2i_local.yaml",
    config_path=None,
    dataset_relative_paths=("image2image/local_256.parquet",),
    **overrides,
):
    lance_root = tmp_path / "Lance"
    lance_root.mkdir()
    if config_path is None:
        config = lance_root / "config" / "train_local" / config_name
        config.parent.mkdir(parents=True)
        config.write_text("D1: {}\n", encoding="utf-8")
    else:
        config = Path(config_path)

    dataset_root = tmp_path / "Lance_example_dataset"
    for relative_path in dataset_relative_paths:
        parquet = dataset_root / relative_path
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.touch()

    model_root = tmp_path / "models"
    qwen_path = model_root / "Qwen2.5-VL-3B-Instruct"
    vit_path = model_root / "Qwen2.5-VL-ViT"
    qwen_path.mkdir(parents=True)
    vit_path.mkdir(parents=True)
    (vit_path / "config.json").write_text("{}\n", encoding="utf-8")
    (vit_path / "vit.safetensors").touch()
    vae_path = model_root / "Wan2.2_VAE.pth"
    vae_path.touch()

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    torchrun = fake_bin / "torchrun"
    torchrun.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    torchrun.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
            "SMOKE_TEST": "1",
            "PREFLIGHT_ONLY": "1",
            "LANCE_SOURCE_ROOT": str(lance_root),
            "MODEL_ROOT": str(model_root),
            "QWEN_PATH": str(qwen_path),
            "VIT_PATH": str(vit_path),
            "WAN_VAE_PATH": str(vae_path),
            "DATASET_ROOT": str(dataset_root),
            "DATASET_CONFIG_FILE": str(config),
            "TRAINING_MANIFEST": str(manifest),
        }
    )
    environment.update(overrides)

    return subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_i2i_smoke_uses_a_budget_that_accepts_768px_samples(tmp_path):
    result = _run_smoke(tmp_path)
    arguments = result.stdout.splitlines()

    assert _argument_value(arguments, "--expected_num_tokens") == "4096"
    assert _argument_value(arguments, "--max_num_tokens") == "10240"
    assert _argument_value(arguments, "--max_num_tokens_per_sample") == "10240"
    assert "Token budget: expected=4096, max=10240, per-sample=10240" in result.stdout


def test_i2i_smoke_budget_can_be_overridden(tmp_path):
    result = _run_smoke(
        tmp_path,
        SMOKE_EXPECTED_NUM_TOKENS="6144",
        SMOKE_MAX_NUM_TOKENS="10240",
        SMOKE_MAX_NUM_TOKENS_PER_SAMPLE="9216",
    )
    arguments = result.stdout.splitlines()

    assert _argument_value(arguments, "--expected_num_tokens") == "6144"
    assert _argument_value(arguments, "--max_num_tokens") == "10240"
    assert _argument_value(arguments, "--max_num_tokens_per_sample") == "9216"


def test_v2t_smoke_accepts_two_second_clips_and_disables_generation(tmp_path):
    result = _run_smoke(
        tmp_path,
        config_name="v2t_local.yaml",
        dataset_relative_paths=("video2text/local_256.parquet",),
    )
    arguments = result.stdout.splitlines()

    assert _argument_value(arguments, "--max_num_tokens") == "10240"
    assert _argument_value(arguments, "--max_num_tokens_per_sample") == "10240"
    assert _argument_value(arguments, "--visual_gen") == "false"


def test_pt_smoke_uses_only_pt_modalities_and_paper_context_limits(tmp_path):
    config = REPO_ROOT / "examples" / "lance" / "config" / "train_local" / "pt_smoke.yaml"
    result = _run_smoke(
        tmp_path,
        config_path=config,
        dataset_relative_paths=(
            "text2image/local_256.parquet",
            "text2video/local_128.parquet",
            "image2text/local_256.parquet",
            "video2text/local_256.parquet",
        ),
    )
    arguments = result.stdout.splitlines()

    assert _argument_value(arguments, "--expected_num_tokens") == "4096"
    assert _argument_value(arguments, "--max_num_tokens") == "50000"
    assert _argument_value(arguments, "--max_num_tokens_per_sample") == "40000"
    assert _argument_value(arguments, "--visual_gen") == "true"
    assert _argument_value(arguments, "--visual_und") == "true"
    assert _argument_value(arguments, "--require_und_gen") == "true"
    config_text = config.read_text(encoding="utf-8")
    assert "datasets/image2image" not in config_text
    assert "datasets/video2video" not in config_text
