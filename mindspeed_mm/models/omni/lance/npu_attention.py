"""Ascend block-scheduled attention backend for native Lance.

The generalized Lance mask cannot be represented by one ordinary causal call:
visual spans are bidirectional and noisy target spans must never leak into later
KV.  This baseline decomposes a packed document by query segment.  Each kernel
sees all earlier clean KV plus the current segment; causal text uses right-down
causal alignment, while visual/noise segments use full attention.

No ``[sequence, sequence]`` mask is allocated, so the implementation is safe for
the 70K context target.  A later optimization may coalesce compatible segments
without changing this contract.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .sequence import AttentionBlock, LancePackedSequence


class LanceAscendAttentionError(RuntimeError):
    pass


@dataclass(frozen=True)
class UpstreamSegmentedAttentionMask:
    """Compact form of the sparse mask built by upstream Lance training.

    The released model normally turns the same metadata into a PyTorch
    FlexAttention ``BlockMask``. TorchInductor cannot compile that operation
    for NPU, so the process-local upstream bridge keeps the original document
    and split structure and evaluates it with fused TND attention instead.
    """

    document_lens: Tuple[int, ...]
    split_lens: Tuple[int, ...]
    attn_modes: Tuple[str, ...]

    @classmethod
    def from_upstream(
        cls,
        document_lens: Sequence[int],
        split_lens: Sequence[int],
        attn_modes: Sequence[str],
    ) -> "UpstreamSegmentedAttentionMask":
        normalized_modes = tuple(
            "full" if str(mode) in ("full_noise", "full_noise_target") else str(mode)
            for mode in attn_modes
        )
        mask = cls(
            tuple(int(value) for value in document_lens),
            tuple(int(value) for value in split_lens),
            normalized_modes,
        )
        mask.validate()
        return mask

    def validate(self) -> None:
        if not self.document_lens or any(value <= 0 for value in self.document_lens):
            raise LanceAscendAttentionError("document_lens must contain positive lengths")
        if not self.split_lens or any(value <= 0 for value in self.split_lens):
            raise LanceAscendAttentionError("split_lens must contain positive lengths")
        if len(self.split_lens) != len(self.attn_modes):
            raise LanceAscendAttentionError("split_lens and attn_modes must have equal lengths")
        unsupported = sorted(set(self.attn_modes) - {"causal", "full", "noise"})
        if unsupported:
            raise LanceAscendAttentionError(
                "unsupported upstream attention modes: {}".format(", ".join(unsupported))
            )
        if sum(self.document_lens) != sum(self.split_lens):
            raise LanceAscendAttentionError(
                "document_lens and split_lens must describe the same token count"
            )

    def to(self, *args: Any, **kwargs: Any) -> "UpstreamSegmentedAttentionMask":
        del args, kwargs
        return self


def run_upstream_segmented_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: UpstreamSegmentedAttentionMask,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
) -> torch.Tensor:
    """Evaluate upstream Lance's FlexAttention mask without materialising it.

    Within each document, every query split sees earlier non-noise splits and
    its own split. A causal split uses right-down causal alignment; full and
    noise splits are bidirectional. Noise keys are excluded from subsequent
    splits, matching ``data.data_utils.create_sparse_mask``.
    """

    mask.validate()
    token_count = query.shape[0]
    if key.shape[0] != token_count or value.shape[0] != token_count:
        raise LanceAscendAttentionError("upstream training attention requires equal Q/K/V lengths")
    if sum(mask.document_lens) != token_count:
        raise LanceAscendAttentionError(
            "segmented mask token count does not match the upstream attention tensors"
        )

    outputs = []
    token_offset = 0
    split_index = 0
    for document_len in mask.document_lens:
        document_end = token_offset + document_len
        prefix_keys = []
        prefix_values = []
        while token_offset < document_end:
            if split_index >= len(mask.split_lens):
                raise LanceAscendAttentionError("not enough splits for document_lens")
            split_len = mask.split_lens[split_index]
            split_end = token_offset + split_len
            if split_end > document_end:
                raise LanceAscendAttentionError("an attention split crosses a document boundary")

            query_part = query[token_offset:split_end]
            key_part = key[token_offset:split_end]
            value_part = value[token_offset:split_end]
            attended_keys = torch.cat([*prefix_keys, key_part], dim=0).contiguous()
            attended_values = torch.cat([*prefix_values, value_part], dim=0).contiguous()
            mode = mask.attn_modes[split_index]
            outputs.append(
                attention_fn(
                    query_part,
                    attended_keys,
                    attended_values,
                    mode == "causal",
                )
            )
            if mode != "noise":
                prefix_keys.append(key_part)
                prefix_values.append(value_part)
            token_offset = split_end
            split_index += 1

    if split_index != len(mask.split_lens):
        raise LanceAscendAttentionError("too many splits for document_lens")
    return torch.cat(outputs, dim=0)


def run_upstream_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: UpstreamSegmentedAttentionMask,
    attention_backend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
) -> torch.Tensor:
    """Match the BHLD FlexAttention call shape used by upstream Qwen2-NaViT."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise LanceAscendAttentionError("upstream FlexAttention bridge expects BHLD tensors")
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise LanceAscendAttentionError("upstream packed training requires batch size one")
    query_tnd = query[0].transpose(0, 1).contiguous()
    key_tnd = key[0].transpose(0, 1).contiguous()
    value_tnd = value[0].transpose(0, 1).contiguous()
    output = run_upstream_segmented_attention(
        query_tnd,
        key_tnd,
        value_tnd,
        mask,
        attention_backend,
    )
    return output.transpose(0, 1).unsqueeze(0)


class AscendVisionAttentionBackend:
    """Qwen2.5-VL window/frame attention using one TND varlen NPU call."""

    def __init__(self, torch_npu_module: Any = None) -> None:
        if torch_npu_module is None:
            try:
                import torch_npu as torch_npu_module
            except ImportError as exc:
                raise LanceAscendAttentionError("native Lance vision attention requires torch_npu") from exc
        self.torch_npu = torch_npu_module

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cumulative_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.shape != query.shape or value.shape != query.shape:
            raise LanceAscendAttentionError("Ascend vision attention requires equal-shape TND tensors")
        boundaries = cumulative_lengths.tolist()
        if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != query.shape[0]:
            raise LanceAscendAttentionError("vision cumulative lengths must cover every token")
        cumulative = tuple(int(value) for value in boundaries[1:])
        output = self.torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            head_num=query.shape[1],
            input_layout="TND",
            pse=None,
            padding_mask=None,
            atten_mask=None,
            scale=1.0 / math.sqrt(query.shape[-1]),
            keep_prob=1.0,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            actual_seq_qlen=cumulative,
            actual_seq_kvlen=cumulative,
            sparse_mode=0,
        )[0]
        if output.shape != query.shape:
            raise LanceAscendAttentionError(
                "NPU vision attention returned {}, expected {}".format(
                    tuple(output.shape), tuple(query.shape)
                )
            )
        return output


class AscendKVCacheAttentionBackend:
    """Single-sample q_len != kv_len TND attention for diffusion KV reuse."""

    def __init__(self, torch_npu_module: Any = None) -> None:
        if torch_npu_module is None:
            try:
                import torch_npu as torch_npu_module
            except ImportError as exc:
                raise LanceAscendAttentionError("native Lance KV-cache attention requires torch_npu") from exc
        self.torch_npu = torch_npu_module
        self._causal_masks: Dict[str, torch.Tensor] = {}

    def _causal_mask(self, query: torch.Tensor) -> torch.Tensor:
        key = str(query.device)
        mask = self._causal_masks.get(key)
        if mask is None:
            mask = query.new_ones((2048, 2048), dtype=torch.bool).triu(diagonal=1)
            self._causal_masks[key] = mask
        return mask

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise LanceAscendAttentionError("Ascend KV-cache attention requires TND tensors")
        if (
            key.shape != value.shape
            or key.shape[-1] != query.shape[-1]
            or key.shape[1] <= 0
            or query.shape[1] % key.shape[1]
        ):
            raise LanceAscendAttentionError("invalid KV-cache attention tensor shapes")
        if key.shape[0] < query.shape[0]:
            raise LanceAscendAttentionError("KV-cache key length must be >= query length")
        output = self.torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            head_num=query.shape[1],
            input_layout="TND",
            pse=None,
            padding_mask=None,
            atten_mask=self._causal_mask(query) if is_causal else None,
            scale=1.0 / math.sqrt(query.shape[-1]),
            keep_prob=1.0,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            actual_seq_qlen=(query.shape[0],),
            actual_seq_kvlen=(key.shape[0],),
            sparse_mode=3 if is_causal else 0,
        )[0]
        if output.shape != query.shape:
            raise LanceAscendAttentionError(
                "NPU KV-cache attention returned {}, expected {}".format(
                    tuple(output.shape), tuple(query.shape)
                )
            )
        return output


class AscendBlockAttentionBackend:
    """Callable attention backend matching ``modeling_lance.AttentionBackend``."""

    def __init__(
        self,
        packed_sequence: Optional[LancePackedSequence] = None,
        torch_npu_module: Any = None,
    ) -> None:
        self.packed_sequence = packed_sequence
        self._cached_sequence = None
        self._cached_groups = None
        if torch_npu_module is None:
            try:
                import torch_npu as torch_npu_module
            except ImportError as exc:
                raise LanceAscendAttentionError("native Lance attention requires torch_npu") from exc
        self.torch_npu = torch_npu_module
        self._causal_masks: Dict[str, torch.Tensor] = {}

    def _resolve_schedule(
        self,
        attention_metadata: Any,
    ) -> Tuple[LancePackedSequence, Tuple[Tuple[Tuple[int, int], Tuple[AttentionBlock, ...]], ...]]:
        if isinstance(attention_metadata, LancePackedSequence):
            packed_sequence = attention_metadata
        elif self.packed_sequence is not None:
            # Backward-compatible static mode ignores the dense-mask argument;
            # production callers should pass LancePackedSequence dynamically.
            packed_sequence = self.packed_sequence
        else:
            raise LanceAscendAttentionError(
                "dynamic native attention requires LancePackedSequence metadata"
            )
        if packed_sequence is not self._cached_sequence:
            self._cached_sequence = packed_sequence
            self._cached_groups = self._group_blocks(packed_sequence.block_schedule())
        return packed_sequence, self._cached_groups

    @staticmethod
    def _group_blocks(
        blocks: Tuple[AttentionBlock, ...],
    ) -> Tuple[Tuple[Tuple[int, int], Tuple[AttentionBlock, ...]], ...]:
        grouped: "OrderedDict[Tuple[int, int], List[AttentionBlock]]" = OrderedDict()
        for block in blocks:
            grouped.setdefault((block.query_start, block.query_end), []).append(block)
        return tuple((query_range, tuple(values)) for query_range, values in grouped.items())

    def _causal_mask(self, query: torch.Tensor) -> torch.Tensor:
        key = str(query.device)
        mask = self._causal_masks.get(key)
        if mask is None:
            # ACLNN sparse_mode=3 consumes this canonical compressed mask and
            # applies bottom-right alignment for q_len != kv_len.
            mask = query.new_ones((2048, 2048), dtype=torch.bool).triu(diagonal=1)
            self._causal_masks[key] = mask
        return mask

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise LanceAscendAttentionError("Ascend Lance attention requires TND tensors")
        packed_sequence, groups = self._resolve_schedule(attention_mask)
        if query.shape[0] != packed_sequence.length:
            raise LanceAscendAttentionError(
                "attention tokens do not match the compiled packed-sequence schedule"
            )
        output = torch.empty_like(query)
        scale = 1.0 / math.sqrt(query.shape[-1])
        covered = 0
        for (query_start, query_end), blocks in groups:
            query_slice = query[query_start:query_end]
            key_slice = torch.cat([key[block.key_start:block.key_end] for block in blocks], dim=0)
            value_slice = torch.cat([value[block.key_start:block.key_end] for block in blocks], dim=0)
            self_block = blocks[-1]
            if (self_block.key_start, self_block.key_end) != (query_start, query_end):
                raise LanceAscendAttentionError("compiled schedule must end with the query self-block")
            causal = self_block.causal
            result = self.torch_npu.npu_fusion_attention(
                query_slice,
                key_slice,
                value_slice,
                head_num=query.shape[1],
                input_layout="TND",
                pse=None,
                padding_mask=None,
                atten_mask=self._causal_mask(query_slice) if causal else None,
                scale=scale,
                keep_prob=1.0,
                pre_tockens=2147483647,
                next_tockens=2147483647,
                actual_seq_qlen=(query_end - query_start,),
                actual_seq_kvlen=(key_slice.shape[0],),
                sparse_mode=3 if causal else 0,
            )[0]
            if result.shape != query_slice.shape:
                raise LanceAscendAttentionError(
                    "npu_fusion_attention returned {}, expected {}".format(
                        tuple(result.shape), tuple(query_slice.shape)
                    )
                )
            output[query_start:query_end] = result
            covered += query_end - query_start
        if covered != query.shape[0]:
            raise LanceAscendAttentionError("compiled attention schedule did not cover every query token")
        return output
