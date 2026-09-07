"""Native Lance packed-sequence contracts and dependency-free golden helpers.

Production kernels consume block schedules from this module.  The dense mask is
deliberately limited to short sequences and exists only as an unambiguous oracle
for CPU unit tests; it must never be used for the 40K/70K training path.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


class LanceSequenceError(ValueError):
    """Raised when a packed Lance sequence is structurally invalid."""


ATTENTION_MODES = ("causal", "full", "noise", "full_noise", "full_noise_target")
EXPERTS = ("understanding", "generation")


def normalize_attention_mode(mode: str) -> str:
    if mode in ("full_noise", "full_noise_target"):
        return "full"
    if mode not in ("causal", "full", "noise"):
        raise LanceSequenceError("unsupported attention mode: {}".format(mode))
    return mode


@dataclass(frozen=True)
class LanceSegment:
    length: int
    attention_mode: str
    modality: str
    expert: str

    def __post_init__(self) -> None:
        if not isinstance(self.length, int) or isinstance(self.length, bool) or self.length <= 0:
            raise LanceSequenceError("segment length must be a positive integer")
        if self.attention_mode not in ATTENTION_MODES:
            raise LanceSequenceError("unsupported attention mode: {}".format(self.attention_mode))
        if self.expert not in EXPERTS:
            raise LanceSequenceError("unsupported expert: {}".format(self.expert))
        if not self.modality:
            raise LanceSequenceError("segment modality must not be empty")

    @property
    def normalized_attention_mode(self) -> str:
        return normalize_attention_mode(self.attention_mode)


@dataclass(frozen=True)
class LanceDocument:
    sample_id: str
    segments: Tuple[LanceSegment, ...]

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise LanceSequenceError("sample_id must not be empty")
        if not self.segments:
            raise LanceSequenceError("a document must contain at least one segment")

    @property
    def length(self) -> int:
        return sum(segment.length for segment in self.segments)


@dataclass(frozen=True)
class AttentionBlock:
    document_index: int
    query_start: int
    query_end: int
    key_start: int
    key_end: int
    causal: bool

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LancePackedSequence:
    documents: Tuple[LanceDocument, ...]

    def __post_init__(self) -> None:
        if not self.documents:
            raise LanceSequenceError("a packed sequence must contain at least one document")
        sample_ids = [document.sample_id for document in self.documents]
        if len(sample_ids) != len(set(sample_ids)):
            raise LanceSequenceError("sample_id values must be unique inside a packed sequence")

    @property
    def length(self) -> int:
        return sum(document.length for document in self.documents)

    @property
    def document_lengths(self) -> Tuple[int, ...]:
        return tuple(document.length for document in self.documents)

    @property
    def split_lengths(self) -> Tuple[int, ...]:
        return tuple(segment.length for document in self.documents for segment in document.segments)

    @property
    def attention_modes(self) -> Tuple[str, ...]:
        return tuple(segment.attention_mode for document in self.documents for segment in document.segments)

    def token_expert_indexes(self) -> Dict[str, Tuple[int, ...]]:
        indexes: Dict[str, List[int]] = {"understanding": [], "generation": []}
        offset = 0
        for document in self.documents:
            for segment in document.segments:
                indexes[segment.expert].extend(range(offset, offset + segment.length))
                offset += segment.length
        result = {name: tuple(values) for name, values in indexes.items()}
        combined = sorted(result["understanding"] + result["generation"])
        if combined != list(range(self.length)):
            raise LanceSequenceError("expert routing must cover every packed token exactly once")
        return result

    def block_schedule(self) -> Tuple[AttentionBlock, ...]:
        """Compile the upstream sparse-mask semantics into rectangular blocks.

        Every query segment can read all earlier non-noise segments.  It can read
        itself causally for text or bidirectionally for full/noise visual spans.
        A noise segment is never exposed as KV to another segment, including a
        later segment in the same sample.
        """

        blocks: List[AttentionBlock] = []
        document_offset = 0
        for document_index, document in enumerate(self.documents):
            segment_ranges: List[Tuple[int, int, LanceSegment]] = []
            cursor = document_offset
            for segment in document.segments:
                segment_ranges.append((cursor, cursor + segment.length, segment))
                cursor += segment.length
            for query_index, (query_start, query_end, query_segment) in enumerate(segment_ranges):
                for key_start, key_end, key_segment in segment_ranges[:query_index]:
                    if key_segment.normalized_attention_mode != "noise":
                        blocks.append(
                            AttentionBlock(
                                document_index,
                                query_start,
                                query_end,
                                key_start,
                                key_end,
                                causal=False,
                            )
                        )
                blocks.append(
                    AttentionBlock(
                        document_index,
                        query_start,
                        query_end,
                        query_start,
                        query_end,
                        causal=query_segment.normalized_attention_mode == "causal",
                    )
                )
            document_offset = cursor
        return tuple(blocks)

    def dense_attention_mask(self, max_tokens: int = 4096) -> Tuple[Tuple[bool, ...], ...]:
        """Materialize the exact mask for tests and small-model numerical checks."""

        if self.length > max_tokens:
            raise LanceSequenceError(
                "dense attention oracle is limited to {} tokens, got {}".format(max_tokens, self.length)
            )
        rows = [[False] * self.length for _ in range(self.length)]
        for block in self.block_schedule():
            for query in range(block.query_start, block.query_end):
                for key in range(block.key_start, block.key_end):
                    if not block.causal or query - block.query_start >= key - block.key_start:
                        rows[query][key] = True
        return tuple(tuple(row) for row in rows)


@dataclass(frozen=True)
class LanceLossSelection:
    ce_indexes: Tuple[int, ...] = ()
    ce_labels: Tuple[int, ...] = ()
    ce_weights: Tuple[float, ...] = ()
    mse_indexes: Tuple[int, ...] = ()

    def validate(self, sequence_length: int) -> None:
        if sequence_length <= 0:
            raise LanceSequenceError("sequence_length must be positive")
        if len(self.ce_indexes) != len(self.ce_labels) or len(self.ce_indexes) != len(self.ce_weights):
            raise LanceSequenceError("CE indexes, labels, and weights must have equal length")
        for name, indexes in (("CE", self.ce_indexes), ("MSE", self.mse_indexes)):
            if len(indexes) != len(set(indexes)):
                raise LanceSequenceError("{} loss indexes must be unique".format(name))
            if any(index < 0 or index >= sequence_length for index in indexes):
                raise LanceSequenceError("{} loss index is outside the packed sequence".format(name))
        if set(self.ce_indexes) & set(self.mse_indexes):
            raise LanceSequenceError("CE and MSE loss indexes must not overlap")
        if any(weight <= 0 for weight in self.ce_weights):
            raise LanceSequenceError("CE loss weights must be positive")


def flatten_latent_position_ids(t: int, h: int, w: int, max_latent_size: int = 64) -> Tuple[int, ...]:
    """Match Lance ``get_flattened_position_ids_extrapolate_video`` exactly."""

    if min(t, h, w, max_latent_size) <= 0:
        raise LanceSequenceError("latent grid dimensions must be positive")
    if h > max_latent_size or w > max_latent_size:
        raise LanceSequenceError("latent spatial grid exceeds max_latent_size")
    return tuple(
        frame * max_latent_size * max_latent_size + row * max_latent_size + column
        for frame in range(t)
        for row in range(h)
        for column in range(w)
    )


def shift_timestep(timestep: float, shift: float) -> float:
    """Apply the released Lance/Wan flow timestep shift."""

    if not 0.0 <= timestep <= 1.0:
        raise LanceSequenceError("timestep must be in [0, 1]")
    if shift <= 0.0:
        raise LanceSequenceError("timestep shift must be positive")
    return shift * timestep / (1.0 + (shift - 1.0) * timestep)


def flow_interpolate(clean: float, noise: float, timestep: float) -> float:
    """Return ``(1-t)*clean + t*noise`` as implemented by upstream Lance."""

    if not 0.0 <= timestep <= 1.0:
        raise LanceSequenceError("timestep must be in [0, 1]")
    return (1.0 - timestep) * clean + timestep * noise


def flow_velocity_target(clean: float, noise: float) -> float:
    """Return the upstream training target ``noise - clean``."""

    return noise - clean

