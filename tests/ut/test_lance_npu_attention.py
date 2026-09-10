import os
import sys
import types

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.ascend_runtime import (
    patch_upstream_lance_training_attention,
)
from mindspeed_mm.models.omni.lance.modeling_lance import reference_sdpa
from mindspeed_mm.models.omni.lance.npu_attention import (
    AscendBlockAttentionBackend,
    AscendKVCacheAttentionBackend,
    AscendVisionAttentionBackend,
    LanceAscendAttentionError,
    UpstreamSegmentedAttentionMask,
    run_upstream_flex_attention,
)
from mindspeed_mm.models.omni.lance.sequence import (
    LanceDocument,
    LancePackedSequence,
    LanceSegment,
)


class FakeTorchNPU:
    calls = []

    @classmethod
    def npu_fusion_attention(cls, query, key, value, **kwargs):
        cls.calls.append(kwargs)
        q_length, kv_length = query.shape[0], key.shape[0]
        if kwargs["sparse_mode"] == 3:
            row = torch.arange(q_length).unsqueeze(1)
            column = torch.arange(kv_length).unsqueeze(0)
            mask = column <= row + (kv_length - q_length)
        else:
            mask = torch.ones(q_length, kv_length, dtype=torch.bool)
        return (reference_sdpa(query, key, value, mask),)


def _segment(length, mode, modality="text", expert="understanding"):
    return LanceSegment(length, mode, modality, expert)


def test_block_scheduled_npu_calls_equal_dense_attention_oracle():
    FakeTorchNPU.calls = []
    packed = LancePackedSequence(
        (
            LanceDocument(
                "a",
                (
                    _segment(2, "causal"),
                    _segment(2, "full", "vit"),
                    _segment(2, "noise", "vae", "generation"),
                    _segment(1, "causal"),
                ),
            ),
            LanceDocument(
                "b",
                (
                    _segment(1, "causal"),
                    _segment(2, "full_noise", "clean_vae", "generation"),
                ),
            ),
        )
    )
    torch.manual_seed(31)
    query = torch.randn(packed.length, 4, 8)
    key = torch.randn(packed.length, 2, 8)
    value = torch.randn(packed.length, 2, 8)
    dense_mask = torch.tensor(packed.dense_attention_mask(), dtype=torch.bool)
    expected = reference_sdpa(query, key, value, dense_mask)
    backend = AscendBlockAttentionBackend(packed, torch_npu_module=FakeTorchNPU)
    actual = backend(query, key, value, None)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    assert len(FakeTorchNPU.calls) == 6
    assert [call["sparse_mode"] for call in FakeTorchNPU.calls] == [3, 0, 0, 3, 3, 0]
    assert FakeTorchNPU.calls[3]["actual_seq_kvlen"] == (5,)
    assert all(call["input_layout"] == "TND" for call in FakeTorchNPU.calls)


def test_block_backend_rejects_schedule_length_mismatch():
    packed = LancePackedSequence((LanceDocument("a", (_segment(2, "causal"),)),))
    backend = AscendBlockAttentionBackend(packed, torch_npu_module=FakeTorchNPU)
    with pytest.raises(LanceAscendAttentionError, match="schedule"):
        backend(
            torch.randn(3, 4, 8),
            torch.randn(3, 2, 8),
            torch.randn(3, 2, 8),
            torch.ones(3, 3, dtype=torch.bool),
        )


def test_dynamic_backend_accepts_different_batch_schedules_without_model_rebuild():
    first = LancePackedSequence((LanceDocument("first", (_segment(2, "causal"),)),))
    second = LancePackedSequence((LanceDocument("second", (_segment(3, "full", "vit"),)),))
    backend = AscendBlockAttentionBackend(torch_npu_module=FakeTorchNPU)

    for packed in (first, second):
        query = torch.randn(packed.length, 4, 8)
        key = torch.randn(packed.length, 2, 8)
        value = torch.randn(packed.length, 2, 8)
        expected = reference_sdpa(
            query,
            key,
            value,
            torch.tensor(packed.dense_attention_mask(), dtype=torch.bool),
        )
        actual = backend(query, key, value, packed)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class FakeVisionTorchNPU:
    call = None

    @classmethod
    def npu_fusion_attention(cls, query, key, value, **kwargs):
        cls.call = kwargs
        cumulative = torch.tensor((0,) + kwargs["actual_seq_qlen"], dtype=torch.int32)
        from mindspeed_mm.models.omni.lance.modeling_lance import reference_vision_sdpa

        return (reference_vision_sdpa(query, key, value, cumulative),)


def test_vision_backend_uses_one_full_tnd_varlen_call():
    torch.manual_seed(37)
    query = torch.randn(5, 2, 4)
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    cumulative = torch.tensor([0, 2, 5], dtype=torch.int32)
    backend = AscendVisionAttentionBackend(torch_npu_module=FakeVisionTorchNPU)
    actual = backend(query, key, value, cumulative)
    from mindspeed_mm.models.omni.lance.modeling_lance import reference_vision_sdpa

    expected = reference_vision_sdpa(query, key, value, cumulative)
    torch.testing.assert_close(actual, expected)
    assert FakeVisionTorchNPU.call["input_layout"] == "TND"
    assert FakeVisionTorchNPU.call["actual_seq_qlen"] == (2, 5)
    assert FakeVisionTorchNPU.call["actual_seq_kvlen"] == (2, 5)
    assert FakeVisionTorchNPU.call["atten_mask"] is None
    assert FakeVisionTorchNPU.call["sparse_mode"] == 0


@pytest.mark.parametrize("is_causal", [False, True])
def test_kv_cache_backend_matches_non_equal_length_oracle(is_causal):
    FakeTorchNPU.calls = []
    torch.manual_seed(47)
    query = torch.randn(3, 4, 8)
    key = torch.randn(8, 2, 8)
    value = torch.randn(8, 2, 8)
    row = torch.arange(3).unsqueeze(1)
    column = torch.arange(8).unsqueeze(0)
    mask = column <= row + 5 if is_causal else torch.ones(3, 8, dtype=torch.bool)
    expected = reference_sdpa(query, key, value, mask)
    backend = AscendKVCacheAttentionBackend(torch_npu_module=FakeTorchNPU)
    actual = backend(query, key, value, is_causal)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    call = FakeTorchNPU.calls[-1]
    assert call["actual_seq_qlen"] == (3,)
    assert call["actual_seq_kvlen"] == (8,)
    assert call["sparse_mode"] == (3 if is_causal else 0)
    assert (call["atten_mask"] is not None) is is_causal


def test_upstream_segmented_flex_bridge_matches_sparse_mask_semantics():
    FakeTorchNPU.calls = []
    mask = UpstreamSegmentedAttentionMask.from_upstream(
        document_lens=(7, 3),
        split_lens=(2, 2, 2, 1, 1, 2),
        attn_modes=("causal", "full", "noise", "causal", "causal", "full_noise"),
    )
    packed = LancePackedSequence(
        (
            LanceDocument(
                "first",
                (
                    _segment(2, "causal"),
                    _segment(2, "full", "vit"),
                    _segment(2, "noise", "vae", "generation"),
                    _segment(1, "causal"),
                ),
            ),
            LanceDocument(
                "second",
                (
                    _segment(1, "causal"),
                    _segment(2, "full_noise", "clean_vae", "generation"),
                ),
            ),
        )
    )
    torch.manual_seed(53)
    query = torch.randn(1, 4, packed.length, 8)
    key = torch.randn(1, 2, packed.length, 8)
    value = torch.randn(1, 2, packed.length, 8)
    expected = reference_sdpa(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1),
        value[0].transpose(0, 1),
        torch.tensor(packed.dense_attention_mask(), dtype=torch.bool),
    ).transpose(0, 1).unsqueeze(0)
    backend = AscendKVCacheAttentionBackend(torch_npu_module=FakeTorchNPU)
    actual = run_upstream_flex_attention(query, key, value, mask, backend)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert [call["sparse_mode"] for call in FakeTorchNPU.calls] == [3, 0, 0, 3, 3, 0]


def test_training_runtime_intercepts_upstream_block_mask_without_editing_source(monkeypatch):
    lance_module = types.ModuleType("modeling.lance.lance")
    navit_module = types.ModuleType("modeling.lance.qwen2_navit")
    lance_module.create_sparse_mask = lambda *args, **kwargs: "original-sparse-mask"
    lance_module.create_block_mask = lambda *args, **kwargs: "original-block-mask"
    navit_module.flex_attention = lambda *args, **kwargs: "original-flex-attention"
    monkeypatch.setitem(sys.modules, "modeling.lance.lance", lance_module)
    monkeypatch.setitem(sys.modules, "modeling.lance.qwen2_navit", navit_module)

    result = patch_upstream_lance_training_attention(FakeTorchNPU)
    mask = lance_module.create_sparse_mask(
        (4,),
        (2, 2),
        ("causal", "full_noise"),
        types.SimpleNamespace(type="npu"),
    )

    assert result["status"] == "installed"
    assert isinstance(mask, UpstreamSegmentedAttentionMask)
    assert mask.attn_modes == ("causal", "full")
    assert lance_module.create_block_mask(mask) is mask
    assert lance_module.create_sparse_mask((1,), (1,), ("causal",), "cpu") == (
        "original-sparse-mask"
    )
    assert navit_module.flex_attention(None, None, None) == "original-flex-attention"
