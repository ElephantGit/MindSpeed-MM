"""MVBench 20-task adapter for Lance video understanding."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union


SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. Based on "
    "your observations, select the best option that accurately addresses the question."
)


@dataclass(frozen=True)
class MVBenchTask:
    name: str
    abbreviation: str
    annotation: str
    media_prefix: str
    media_type: str
    bounded: bool


MVBENCH_TASKS: Tuple[MVBenchTask, ...] = (
    MVBenchTask("Action Sequence", "AS", "action_sequence.json", "star/Charades_v1_480", "video", True),
    MVBenchTask("Action Prediction", "AP", "action_prediction.json", "star/Charades_v1_480", "video", True),
    MVBenchTask("Action Antonym", "AA", "action_antonym.json", "ssv2_video", "video", False),
    MVBenchTask("Fine-grained Action", "FA", "fine_grained_action.json", "Moments_in_Time_Raw/videos", "video", False),
    MVBenchTask("Unexpected Action", "UA", "unexpected_action.json", "FunQA_test/test", "video", False),
    MVBenchTask("Object Existence", "OE", "object_existence.json", "clevrer/video_validation", "video", False),
    MVBenchTask("Object Interaction", "OI", "object_interaction.json", "star/Charades_v1_480", "video", True),
    MVBenchTask("Object Shuffle", "OS", "object_shuffle.json", "perception/videos", "video", False),
    MVBenchTask("Moving Direction", "MD", "moving_direction.json", "clevrer/video_validation", "video", False),
    MVBenchTask("Action Localization", "AL", "action_localization.json", "sta/sta_video", "video", True),
    MVBenchTask("Scene Transition", "ST", "scene_transition.json", "scene_qa/video", "video", False),
    MVBenchTask("Action Count", "AC", "action_count.json", "perception/videos", "video", False),
    MVBenchTask("Moving Count", "MC", "moving_count.json", "clevrer/video_validation", "video", False),
    MVBenchTask("Moving Attribute", "MA", "moving_attribute.json", "clevrer/video_validation", "video", False),
    MVBenchTask("State Change", "SC", "state_change.json", "perception/videos", "video", False),
    MVBenchTask("Fine-grained Pose", "FP", "fine_grained_pose.json", "nturgbd", "video", False),
    MVBenchTask("Character Order", "CO", "character_order.json", "perception/videos", "video", False),
    MVBenchTask("Egocentric Navigation", "EN", "egocentric_navigation.json", "vlnqa", "video", False),
    MVBenchTask("Episodic Reasoning", "ER", "episodic_reasoning.json", "tvqa/frames_fps3_hq", "frame", True),
    MVBenchTask("Counterfactual Inference", "CI", "counterfactual_inference.json", "clevrer/video_validation", "video", False),
)

# Lance Table 8 contains 19 task columns.  Fine-grained Pose (FP), which is part
# of the official 20-task MVBench release, is not reported.  The 19 published
# per-task scores average to 62.0, so reproducing the paper means evaluating
# this exact subset rather than silently using the 20-task leaderboard suite.
MVBENCH_PAPER_TASKS: Tuple[MVBenchTask, ...] = tuple(
    task for task in MVBENCH_TASKS if task.name != "Fine-grained Pose"
)
MVBENCH_TASK_SETS = {
    "paper": MVBENCH_PAPER_TASKS,
    "official": MVBENCH_TASKS,
}
MVBENCH_EXPECTED_SAMPLES = {"paper": 3800, "official": 4000}
MVBENCH_PAPER_SCORES = {
    "Action Sequence": 73.9,
    "Action Prediction": 76.5,
    "Action Antonym": 71.5,
    "Fine-grained Action": 49.0,
    "Unexpected Action": 63.5,
    "Object Existence": 96.0,
    "Object Interaction": 72.5,
    "Object Shuffle": 33.0,
    "Moving Direction": 63.5,
    "Action Localization": 33.0,
    "Scene Transition": 86.0,
    "Action Count": 41.0,
    "Moving Count": 82.0,
    "Moving Attribute": 97.5,
    "State Change": 43.0,
    "Character Order": 47.5,
    "Egocentric Navigation": 31.5,
    "Episodic Reasoning": 40.0,
    "Counterfactual Inference": 77.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_and_answer(sample: Mapping[str, object]) -> Tuple[str, str, str]:
    question = str(sample["question"])
    candidates = sample["candidates"]
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("MVBench candidates must be a sequence")
    answer = str(sample["answer"])
    try:
        answer_index = list(candidates).index(answer)
    except ValueError as exc:
        raise ValueError("MVBench answer is not present in candidates") from exc
    if answer_index >= 26:
        raise ValueError("MVBench supports at most 26 answer choices")
    letter = chr(ord("A") + answer_index)
    lines = ["Question: {}".format(question), "Options:"]
    lines.extend("({}) {}".format(chr(ord("A") + index), candidate) for index, candidate in enumerate(candidates))
    lines.append("Only give the best option.")
    return "\n".join(lines), "({}) {}".format(letter, answer), letter


def _run_ffmpeg(command: List[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ffmpeg failed with exit code {}".format(exc.returncode)) from exc


def _materialize_media(
    source: Path,
    target: Path,
    media_type: str,
    bound: Optional[Tuple[float, float]],
    ffmpeg: str,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return target
    if media_type == "frame":
        command = [
            ffmpeg, "-loglevel", "error", "-y", "-framerate", "3", "-start_number", "1",
            "-i", str(source / "%05d.jpg"),
        ]
    else:
        command = [ffmpeg, "-loglevel", "error", "-y", "-i", str(source)]
    if bound is not None:
        start, end = bound
        if end <= start:
            raise ValueError("MVBench media bound must have end > start")
        command.extend(["-ss", str(start), "-t", str(end - start)])
    command.extend(["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)])
    _run_ffmpeg(command)
    return target


def prepare_mvbench(
    annotation_root: Union[str, Path],
    media_root: Union[str, Path],
    output_dir: Union[str, Path],
    expected_samples: Optional[int] = None,
    materialize: bool = True,
    task_set: str = "paper",
) -> Dict[str, object]:
    """Convert official MVBench data and crop bounded samples for Lance."""
    if task_set not in MVBENCH_TASK_SETS:
        raise ValueError("Unknown MVBench task set: {}".format(task_set))
    tasks = MVBENCH_TASK_SETS[task_set]
    if expected_samples is None:
        expected_samples = MVBENCH_EXPECTED_SAMPLES[task_set]
    annotation_root = Path(annotation_root).expanduser().resolve()
    media_root = Path(media_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") if materialize else None
    if materialize and not ffmpeg:
        raise RuntimeError("ffmpeg is required to materialize bounded/frame MVBench samples")

    lance_data: Dict[str, object] = {}
    metadata: List[Dict[str, object]] = []
    missing: List[str] = []
    global_index = 0
    for task in tasks:
        annotation_path = annotation_root / task.annotation
        if not annotation_path.is_file():
            missing.append(str(annotation_path))
            continue
        with annotation_path.open("r", encoding="utf-8") as stream:
            samples = json.load(stream)
        if not isinstance(samples, list):
            raise ValueError("{} must contain a JSON list".format(annotation_path))
        for task_index, sample in enumerate(samples):
            item_id = "{:08d}".format(global_index)
            source = media_root / task.media_prefix / str(sample["video"])
            if not source.exists():
                missing.append(str(source))
                global_index += 1
                continue
            bound = None
            if task.bounded:
                bound = (float(sample["start"]), float(sample["end"]))
            requires_materialization = task.media_type == "frame" or bound is not None
            if requires_materialization:
                if not materialize:
                    raise ValueError("bounded/frame MVBench samples require --materialize")
                media_path = _materialize_media(
                    source,
                    output_dir / "media" / (item_id + ".mp4"),
                    task.media_type,
                    bound,
                    str(ffmpeg),
                )
            else:
                media_path = source
            question, answer, answer_letter = _question_and_answer(sample)
            lance_data[item_id] = {
                "interleave_array": [str(media_path), [SYSTEM_PROMPT, question, answer]],
                "element_dtype_array": ["video", "text"],
                "istarget_in_interleave": [0, 1],
                "additional_information": {
                    "mvbench_task": task.name,
                    "mvbench_abbreviation": task.abbreviation,
                },
            }
            metadata.append(
                {
                    "id": item_id,
                    "task": task.name,
                    "abbreviation": task.abbreviation,
                    "task_index": task_index,
                    "answer": answer,
                    "answer_letter": answer_letter,
                    "media": str(media_path),
                    "question": question,
                }
            )
            global_index += 1
    if missing:
        preview = missing[:10]
        raise FileNotFoundError("Missing MVBench files (first {}): {}".format(len(preview), preview))
    if len(lance_data) != expected_samples:
        raise ValueError("Expected {} MVBench samples, found {}".format(expected_samples, len(lance_data)))

    dataset_path = output_dir / "mvbench_lance.json"
    metadata_path = output_dir / "mvbench_metadata.json"
    dataset_path.write_text(json.dumps(lance_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "valid",
        "samples": len(lance_data),
        "tasks": len(tasks),
        "task_set": "paper-lance-19" if task_set == "paper" else "official-mvbench-20",
        "dataset_path": str(dataset_path),
        "metadata_path": str(metadata_path),
    }


_LEADING_OPTION = re.compile(
    r"^\s*(?:"
    r"best\s+option\s*[:：]?\s*[\(\[]?\s*([A-Za-z])|"
    r"[\(\[]\s*([A-Za-z])\s*[\)\]]|"
    r"([A-Za-z])\s*[\)\].:：]|"
    r"([A-Za-z])(?:\s|$)"
    r")",
    flags=re.IGNORECASE,
)


def extract_option(text: object) -> Optional[str]:
    match = _LEADING_OPTION.search(str(text).replace("<|im_end|>", "").strip())
    if not match:
        return None
    return next(group for group in match.groups() if group is not None).upper()


def score_mvbench(
    metadata: Union[str, Path, Sequence[Mapping[str, object]]],
    results: Union[str, Path, Sequence[Mapping[str, object]]],
    task_set: str = "paper",
    strict: bool = True,
) -> Dict[str, object]:
    if task_set not in MVBENCH_TASK_SETS:
        raise ValueError("Unknown MVBench task set: {}".format(task_set))
    metadata_path = Path(metadata).expanduser().resolve() if isinstance(metadata, (str, Path)) else None
    results_path = Path(results).expanduser().resolve() if isinstance(results, (str, Path)) else None
    if metadata_path is not None:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    if results_path is not None:
        with results_path.open("r", encoding="utf-8") as stream:
            results = json.load(stream)
    if len(metadata) != len(results):
        raise ValueError("MVBench metadata/results length mismatch: {} != {}".format(len(metadata), len(results)))

    per_task: Dict[str, List[int]] = {}
    records = []
    correct = 0
    for expected, actual in zip(metadata, results):
        if expected.get("question") and actual.get("question") != expected["question"]:
            raise ValueError("MVBench metadata/results order mismatch at id {}".format(expected["id"]))
        task = str(expected["task"])
        prediction = extract_option(actual.get("answer", ""))
        target = str(expected["answer_letter"])
        is_correct = prediction == target
        correct += int(is_correct)
        counts = per_task.setdefault(task, [0, 0])
        counts[0] += int(is_correct)
        counts[1] += 1
        records.append(
            {
                "id": expected["id"],
                "task": task,
                "prediction": prediction,
                "target": target,
                "correct": is_correct,
            }
        )
    total = len(records)
    task_scores = {
        task: {"correct": value[0], "total": value[1], "accuracy": 100.0 * value[0] / value[1]}
        for task, value in per_task.items()
    }
    actual_tasks = set(per_task)
    expected_tasks = {task.name for task in MVBENCH_TASK_SETS[task_set]}
    if strict and actual_tasks != expected_tasks:
        missing = sorted(expected_tasks - actual_tasks)
        unexpected = sorted(actual_tasks - expected_tasks)
        raise ValueError("MVBench task-set mismatch: missing={}, unexpected={}".format(missing, unexpected))
    expected_samples = MVBENCH_EXPECTED_SAMPLES[task_set]
    if strict and total != expected_samples:
        raise ValueError("Expected {} MVBench results, found {}".format(expected_samples, total))
    # Lance Table 8 reports the arithmetic mean of its published task columns.
    # Keep the micro average as a diagnostic, but use macro for paper parity.
    macro_accuracy = (
        sum(value["accuracy"] for value in task_scores.values()) / len(task_scores)
        if task_scores else 0.0
    )
    micro_accuracy = 100.0 * correct / total if total else 0.0
    provenance = {
        "tasks": len(actual_tasks),
        "samples": total,
        "task_set": "paper-lance-19" if task_set == "paper" else "official-mvbench-20",
        "aggregation": "macro-task-accuracy",
        "answer_extraction": "official-leading-option",
    }
    if metadata_path is not None:
        provenance["metadata_sha256"] = _sha256(metadata_path)
    if results_path is not None:
        provenance["results_sha256"] = _sha256(results_path)
    paper_alignment = None
    if task_set == "paper":
        paper_alignment = {
            task: {
                "score": task_scores[task]["accuracy"],
                "paper_score": target,
                "delta": task_scores[task]["accuracy"] - target,
            }
            for task, target in MVBENCH_PAPER_SCORES.items()
            if task in task_scores
        }
    return {
        "scores": {"mvbench": macro_accuracy},
        "provenance": {"mvbench": provenance},
        "details": {
            "correct": correct,
            "total": total,
            "micro_accuracy": micro_accuracy,
            "per_task": task_scores,
            "paper_alignment": paper_alignment,
            "records": records,
        },
    }
