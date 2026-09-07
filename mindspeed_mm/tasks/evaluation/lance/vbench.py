"""Standalone NPU VBench scoring without loading a generation pipeline."""

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Union


PINNED_VBENCH_VERSION = "0.1.2"
VBENCH_EXPECTED_VIDEOS = 6230
VBENCH_EXPECTED_PROMPTS = 946


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_vbench(
    videos_path: Union[str, Path],
    full_info_path: Union[str, Path],
    output_dir: Union[str, Path],
    dimensions: Optional[Sequence[str]] = None,
    load_checkpoints_locally: bool = False,
    result_name: str = "lance_vbench",
) -> Optional[Dict[str, object]]:
    """Run official VBench with MindSpeed-MM's Ascend operator patches.

    All ranks call this function.  Only rank zero returns the report dictionary.
    """
    videos_root = Path(videos_path).expanduser().resolve()
    video_count = len(list(videos_root.glob("*.mp4"))) if videos_root.is_dir() else 0
    if video_count != VBENCH_EXPECTED_VIDEOS:
        raise ValueError(
            "Expected {} VBench videos, found {}".format(VBENCH_EXPECTED_VIDEOS, video_count)
        )
    full_info = Path(full_info_path).expanduser().resolve()
    if not full_info.is_file():
        raise FileNotFoundError("VBench full-info JSON does not exist: {}".format(full_info))
    try:
        vbench_version = importlib_metadata.version("vbench")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError("Pinned vbench=={} is not installed".format(PINNED_VBENCH_VERSION)) from exc
    if vbench_version != PINNED_VBENCH_VERSION:
        raise RuntimeError(
            "VBench scorer version {} does not match pinned {}".format(
                vbench_version, PINNED_VBENCH_VERSION
            )
        )

    from mindspeed_mm.models.omni.lance.ascend_runtime import enable_lance_ascend_runtime

    enable_lance_ascend_runtime()
    import torch
    import torch.distributed as dist
    from vbench import VBench

    from mindspeed_mm.tasks.evaluation.gen_impl.vbench_utils.compute_score import compute_score
    from mindspeed_mm.tasks.evaluation.gen_impl.vbench_utils.vbench_t2v_patch import patch_t2v

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group("hccl")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_t2v()
    evaluator = VBench(
        torch.device("npu", local_rank),
        str(full_info),
        str(output_dir),
    )
    dimension_list = list(dimensions) if dimensions else evaluator.build_full_dimension_list()
    evaluator.evaluate(
        videos_path=str(videos_root),
        name=result_name,
        prompt_list=[],
        dimension_list=dimension_list,
        local=load_checkpoints_locally,
        read_frame=False,
        mode="vbench_standard",
    )
    if dist.is_initialized():
        dist.barrier()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return None

    raw_result_path = output_dir / (result_name + "_eval_results.json")
    with raw_result_path.open("r", encoding="utf-8") as stream:
        raw_result = json.load(stream)
    raw_dimension_scores = {}
    for key, value in raw_result.items():
        score = value[0] if isinstance(value, list) else value
        raw_dimension_scores[key] = float(score)
    if set(raw_dimension_scores) != set(evaluator.build_full_dimension_list()):
        raise ValueError("VBench total score requires every official dimension")
    dimension_scores = {key.replace("_", " "): value for key, value in raw_dimension_scores.items()}
    aggregate = compute_score(dimension_scores)
    return {
        "scores": {"vbench": aggregate["total_score"] * 100.0},
        "provenance": {
            "vbench": {
                "scorer": "official-vbench",
                "scorer_version": vbench_version,
                "released_recaption": True,
                "prompts": VBENCH_EXPECTED_PROMPTS,
                "videos": video_count,
                "dimensions": len(dimension_scores),
                "full_info_sha256": _sha256(full_info),
                "results_sha256": _sha256(raw_result_path),
            }
        },
        "details": {
            "quality_score": aggregate["quality_score"] * 100.0,
            "semantic_score": aggregate["semantic_score"] * 100.0,
            "dimension_scores": dimension_scores,
            "raw_result": str(raw_result_path),
        },
    }
