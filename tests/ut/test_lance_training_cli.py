import json
import os

os.environ.setdefault("NON_MEGATRON", "true")

import prepare_lance_training


def test_training_cli_binds_dataset_manifest_hash(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"name":"smoke"}\n', encoding="utf-8")
    output = tmp_path / "launch.json"
    result = prepare_lance_training.main(
        [
            "--stage",
            "pt",
            "--init-mode",
            "random",
            "--variant",
            "video",
            "--world-size",
            "8",
            "--dataset-manifest",
            str(dataset),
            "--output",
            str(output),
        ]
    )
    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ready-for-runtime-preflight"
    assert payload["initialization"]["label"] == "strict-random-initialization"
    assert payload["dataset_manifests"][0]["sha256"]
    assert payload["model"]["latent_position_count"] == 126976

