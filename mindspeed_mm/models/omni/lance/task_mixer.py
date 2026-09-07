"""Deterministic, resumable task mixing for Lance pretraining."""

from functools import reduce
import math
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .training_contract import PAPER_TASK_MIX


class LanceTaskMixerError(ValueError):
    pass


class LanceTaskMixer:
    """Shuffle exact-ratio cycles with a rank-local reproducible RNG."""

    def __init__(
        self,
        *,
        global_seed: int,
        world_size: int,
        global_rank: int,
        weights: Optional[Mapping[str, int]] = None,
    ) -> None:
        if world_size <= 0 or not 0 <= global_rank < world_size:
            raise LanceTaskMixerError("global_rank must be in [0, world_size)")
        selected = dict(PAPER_TASK_MIX if weights is None else weights)
        if not selected or any(not name or not isinstance(value, int) or value <= 0 for name, value in selected.items()):
            raise LanceTaskMixerError("task weights must be positive integers")
        divisor = reduce(math.gcd, selected.values())
        self.weights = selected
        self.cycle_counts = {name: value // divisor for name, value in selected.items()}
        self.global_seed = int(global_seed)
        self.world_size = world_size
        self.global_rank = global_rank
        self.rank_seed = self.global_seed * self.world_size + self.global_rank
        self._rng = random.Random(self.rank_seed)
        self._cycle: List[str] = []
        self._cursor = 0
        self._completed_cycles = 0

    @property
    def cycle_length(self) -> int:
        return sum(self.cycle_counts.values())

    def _refill(self) -> None:
        self._cycle = [
            name
            for name, count in self.cycle_counts.items()
            for _ in range(count)
        ]
        self._rng.shuffle(self._cycle)
        self._cursor = 0

    def next_task(self) -> str:
        if not self._cycle or self._cursor == len(self._cycle):
            if self._cycle:
                self._completed_cycles += 1
            self._refill()
        task = self._cycle[self._cursor]
        self._cursor += 1
        return task

    def take(self, count: int) -> Tuple[str, ...]:
        if count < 0:
            raise LanceTaskMixerError("take count must be non-negative")
        return tuple(self.next_task() for _ in range(count))

    def state_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "weights": dict(self.weights),
            "cycle_counts": dict(self.cycle_counts),
            "global_seed": self.global_seed,
            "world_size": self.world_size,
            "global_rank": self.global_rank,
            "rank_seed": self.rank_seed,
            "cycle": list(self._cycle),
            "cursor": self._cursor,
            "completed_cycles": self._completed_cycles,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "weights": self.weights,
            "cycle_counts": self.cycle_counts,
            "global_seed": self.global_seed,
            "world_size": self.world_size,
            "global_rank": self.global_rank,
            "rank_seed": self.rank_seed,
        }
        mismatches = [name for name, value in expected.items() if state.get(name) != value]
        if mismatches:
            raise LanceTaskMixerError(
                "task mixer state is incompatible: {}".format(", ".join(mismatches))
            )
        cycle = list(state.get("cycle", []))
        cursor = state.get("cursor")
        completed = state.get("completed_cycles")
        expected_cycle = sorted(
            name for name, count in self.cycle_counts.items() for _ in range(count)
        )
        if cycle and sorted(cycle) != expected_cycle:
            raise LanceTaskMixerError("task mixer state contains an invalid cycle")
        if not isinstance(cursor, int) or not 0 <= cursor <= len(cycle):
            raise LanceTaskMixerError("task mixer state contains an invalid cursor")
        if not isinstance(completed, int) or completed < 0:
            raise LanceTaskMixerError("task mixer state contains an invalid cycle count")
        try:
            self._rng.setstate(_tupleize(state["rng_state"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LanceTaskMixerError("task mixer state contains invalid RNG state") from exc
        self._cycle = cycle
        self._cursor = cursor
        self._completed_cycles = completed


def _tupleize(value):
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_tupleize(item) for item in value)
    return value


def group_by_token_budget(
    samples: Sequence[object],
    *,
    expected_tokens: int,
    max_tokens: int,
    max_sample_tokens: Optional[int] = None,
) -> Tuple[Tuple[object, ...], ...]:
    """Group an ordered sample stream without crossing a hard rank budget."""

    if expected_tokens <= 0 or max_tokens <= 0 or expected_tokens > max_tokens:
        raise LanceTaskMixerError("token budgets must satisfy 0 < expected <= max")
    sample_limit = max_tokens if max_sample_tokens is None else max_sample_tokens
    if sample_limit <= 0 or sample_limit > max_tokens:
        raise LanceTaskMixerError("max_sample_tokens must be in (0, max_tokens]")
    groups: List[Tuple[object, ...]] = []
    current: List[object] = []
    current_tokens = 0
    for sample in samples:
        length = getattr(sample, "length", None)
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise LanceTaskMixerError("each sample must expose a positive integer length")
        if length > sample_limit:
            raise LanceTaskMixerError(
                "sample length {} exceeds max_sample_tokens {}".format(length, sample_limit)
            )
        if current and current_tokens + length > max_tokens:
            groups.append(tuple(current))
            current = []
            current_tokens = 0
        current.append(sample)
        current_tokens += length
        if current_tokens >= expected_tokens:
            groups.append(tuple(current))
            current = []
            current_tokens = 0
    if current:
        groups.append(tuple(current))
    return tuple(groups)
