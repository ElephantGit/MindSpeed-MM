"""Ascend runtime and native-model contracts for Lance."""

from .checkpoint import (
    LanceCheckpointError,
    audit_checkpoint_metadata,
    build_checkpoint_conversion_plan,
    expected_state_shapes,
    read_safetensors_header,
)
from .dcp import LanceDCPError, convert_safetensors_to_dcp, verify_dcp_metadata
from .native_config import LanceConfigError, LanceNativeConfig
from .runner import LanceSourceError, resolve_lance_source, run_lance_entrypoint
from .sequence import (
    LanceDocument,
    LanceLossSelection,
    LancePackedSequence,
    LanceSegment,
    LanceSequenceError,
)
from .training_contract import PAPER_OPTIMIZER, LanceStage, LanceStageError, STAGES, training_manifest
from .training_bridge import (
    LanceTrainingBridgeError,
    UPSTREAM_TRAINING_ENTRYPOINT,
    build_training_bridge_preflight,
    validate_training_manifest,
    validate_upstream_training_source,
)
from .upstream_training import (
    LanceUpstreamTrainingError,
    compile_strict_training_entrypoint,
    validate_strict_training_entrypoint,
)
from .task_mixer import LanceTaskMixer, LanceTaskMixerError, group_by_token_budget


def get_native_model_class():
    """Import the torch-dependent native model only when the runtime is ready."""

    from .modeling_lance import LanceNativeModel

    return LanceNativeModel


def get_native_sampler():
    """Load the torch-dependent native sampler without eager torch imports."""

    from .sampling import sample_native_lance

    return sample_native_lance


def get_native_cached_sampler():
    """Load the KV-cached native sampler without eager torch imports."""

    from .sampling import sample_native_lance_cached

    return sample_native_lance_cached


def get_preencoded_collator():
    """Load the torch-dependent prepared-sample collator lazily."""

    from .data import pack_preencoded_samples

    return pack_preencoded_samples


def get_training_state_class():
    """Load the torch-dependent resumable training-state contract lazily."""

    from .training_state import LanceTrainingState

    return LanceTrainingState


def get_native_checkpoint_loader():
    """Load the strict streaming safetensors loader lazily."""

    from .initialization import load_native_lance_checkpoint

    return load_native_lance_checkpoint

__all__ = [
    "LanceCheckpointError",
    "LanceConfigError",
    "LanceDCPError",
    "LanceDocument",
    "LanceLossSelection",
    "LanceNativeConfig",
    "LancePackedSequence",
    "LanceSegment",
    "LanceSequenceError",
    "LanceSourceError",
    "LanceStage",
    "LanceStageError",
    "LanceTaskMixer",
    "LanceTaskMixerError",
    "LanceTrainingBridgeError",
    "LanceUpstreamTrainingError",
    "STAGES",
    "PAPER_OPTIMIZER",
    "UPSTREAM_TRAINING_ENTRYPOINT",
    "audit_checkpoint_metadata",
    "build_checkpoint_conversion_plan",
    "build_training_bridge_preflight",
    "convert_safetensors_to_dcp",
    "compile_strict_training_entrypoint",
    "expected_state_shapes",
    "read_safetensors_header",
    "get_native_model_class",
    "get_native_cached_sampler",
    "get_native_checkpoint_loader",
    "get_native_sampler",
    "get_preencoded_collator",
    "get_training_state_class",
    "group_by_token_budget",
    "resolve_lance_source",
    "run_lance_entrypoint",
    "training_manifest",
    "validate_training_manifest",
    "validate_strict_training_entrypoint",
    "validate_upstream_training_source",
    "verify_dcp_metadata",
]
