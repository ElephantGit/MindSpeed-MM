import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance.sequence import (
    LanceDocument,
    LanceLossSelection,
    LancePackedSequence,
    LanceSegment,
    LanceSequenceError,
    flatten_latent_position_ids,
    flow_interpolate,
    flow_velocity_target,
    normalize_attention_mode,
    shift_timestep,
)


def _segment(length, mode, modality="text", expert="understanding"):
    return LanceSegment(length, mode, modality, expert)


def test_dense_oracle_matches_causal_full_noise_semantics():
    # text[0:2], clean visual[2:4], noisy target[4:6], trailing text[6]
    packed = LancePackedSequence(
        (
            LanceDocument(
                "sample-0",
                (
                    _segment(2, "causal"),
                    _segment(2, "full", "vit"),
                    _segment(2, "noise", "vae", "generation"),
                    _segment(1, "causal"),
                ),
            ),
        )
    )
    mask = packed.dense_attention_mask()
    assert mask[0] == (True, False, False, False, False, False, False)
    assert mask[1] == (True, True, False, False, False, False, False)
    assert mask[2] == (True, True, True, True, False, False, False)
    assert mask[3] == (True, True, True, True, False, False, False)
    assert mask[4] == (True, True, True, True, True, True, False)
    assert mask[5] == (True, True, True, True, True, True, False)
    # Later queries can see clean context but the noisy target never enters KV.
    assert mask[6] == (True, True, True, True, False, False, True)


def test_packed_documents_are_strictly_isolated():
    packed = LancePackedSequence(
        (
            LanceDocument("a", (_segment(2, "full", "vit"),)),
            LanceDocument("b", (_segment(2, "causal"),)),
        )
    )
    mask = packed.dense_attention_mask()
    assert not any(mask[0][2:])
    assert not any(mask[1][2:])
    assert not any(mask[2][:2])
    assert not any(mask[3][:2])
    assert mask[2][2] and not mask[2][3]


def test_full_noise_is_full_attention_but_routes_to_generation_expert():
    packed = LancePackedSequence(
        (
            LanceDocument(
                "edit",
                (
                    _segment(1, "causal"),
                    _segment(2, "full_noise", "clean_vae", "generation"),
                    _segment(2, "noise", "noisy_vae", "generation"),
                ),
            ),
        )
    )
    assert normalize_attention_mode("full_noise") == "full"
    indexes = packed.token_expert_indexes()
    assert indexes["understanding"] == (0,)
    assert indexes["generation"] == (1, 2, 3, 4)
    mask = packed.dense_attention_mask()
    assert mask[3] == (True, True, True, True, True)


def test_block_schedule_never_materializes_dense_mask_for_long_sequence():
    packed = LancePackedSequence(
        (LanceDocument("long", (_segment(70000, "causal"),)),)
    )
    blocks = packed.block_schedule()
    assert len(blocks) == 1
    assert blocks[0].query_end == 70000
    assert blocks[0].causal
    with pytest.raises(LanceSequenceError, match="limited"):
        packed.dense_attention_mask()


def test_loss_selection_enforces_index_contract():
    selection = LanceLossSelection((0, 1), (10, 11), (1.0, 0.5), (3, 4))
    selection.validate(5)
    with pytest.raises(LanceSequenceError, match="overlap"):
        LanceLossSelection((1,), (10,), (1.0,), (1,)).validate(2)
    with pytest.raises(LanceSequenceError, match="equal length"):
        LanceLossSelection((0,), (), (1.0,), ()).validate(1)


def test_latent_position_ids_match_upstream_extrapolation_order():
    assert flatten_latent_position_ids(2, 2, 3, max_latent_size=4) == (
        0,
        1,
        2,
        4,
        5,
        6,
        16,
        17,
        18,
        20,
        21,
        22,
    )
    with pytest.raises(LanceSequenceError, match="exceeds"):
        flatten_latent_position_ids(1, 65, 1)


def test_flow_matching_direction_and_shift_are_pinned_to_released_code():
    assert shift_timestep(0.0, 4.0) == 0.0
    assert shift_timestep(1.0, 4.0) == 1.0
    assert shift_timestep(0.5, 4.0) == pytest.approx(0.8)
    assert flow_interpolate(clean=2.0, noise=10.0, timestep=0.25) == 4.0
    assert flow_velocity_target(clean=2.0, noise=10.0) == 8.0


def test_invalid_segment_and_duplicate_sample_are_rejected():
    with pytest.raises(LanceSequenceError, match="positive"):
        _segment(0, "causal")
    with pytest.raises(LanceSequenceError, match="unsupported attention"):
        _segment(1, "dense")
    document = LanceDocument("same", (_segment(1, "causal"),))
    with pytest.raises(LanceSequenceError, match="unique"):
        LancePackedSequence((document, document))

