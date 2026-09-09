"""Compatibility layer used to execute Lance inference on Ascend.

The released Lance inference implementation imports ``flash_attn`` directly and
uses CUDA spellings throughout its launcher.  MindSpeed-MM already relies on
``torch_npu.contrib.transfer_to_npu`` for CUDA-compatible third-party code.  The
remaining gap is FlashAttention's variable-length API; this module supplies the
same narrow call contract using ``torch_npu.npu_fusion_attention`` in TND layout.

This compatibility layer is intentionally installed only by the dedicated Lance
launcher.  It must not be imported as a global side effect by training jobs.
"""

from dataclasses import asdict, dataclass
import importlib
from importlib.machinery import ModuleSpec
import math
import sys
import types
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class LanceRuntimeInfo:
    execution_mode: str
    device_type: str
    distributed_backend: str
    torch_version: str
    torch_npu_version: str
    flash_attention_backend: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def cumulative_lengths(cu_seqlens: Any) -> Tuple[int, ...]:
    """Convert FlashAttention cumulative lengths to the Ascend ACLNN contract."""
    values = cu_seqlens.detach().to(device="cpu").tolist()
    if not values or values[0] != 0:
        raise ValueError("cu_seqlens must begin with zero")
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError("cu_seqlens must be monotonically non-decreasing")
    return tuple(int(value) for value in values[1:])


def _install_flash_attn_shim(
    torch: Any,
    torch_npu: Any,
    *,
    allow_dropout: bool = False,
) -> None:
    """Expose the subset of flash_attn used by Lance through an NPU kernel."""
    causal_masks: Dict[str, Any] = {}

    def flash_attn_varlen_func(
        q: Any,
        k: Any,
        v: Any,
        cu_seqlens_q: Any,
        cu_seqlens_k: Any,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        **_: Any,
    ) -> Any:
        del max_seqlen_q, max_seqlen_k
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Lance NPU varlen attention expects TND tensors")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("attention dropout_p must be in [0, 1)")
        if dropout_p and not allow_dropout:
            raise ValueError("Lance inference requires attention dropout_p=0")

        scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
        atten_mask = None
        if causal:
            device_key = str(q.device)
            atten_mask = causal_masks.get(device_key)
            if atten_mask is None:
                # ACLNN compressed right-down causal mode uses the canonical
                # 2048x2048 upper-triangular mask for arbitrary actual lengths.
                atten_mask = q.new_ones((2048, 2048), dtype=torch.bool).triu(diagonal=1)
                causal_masks[device_key] = atten_mask
        kwargs = {
            "head_num": q.shape[1],
            "input_layout": "TND",
            "pse": None,
            "padding_mask": None,
            "atten_mask": atten_mask,
            "scale": scale,
            "keep_prob": 1.0 - dropout_p,
            "actual_seq_qlen": cumulative_lengths(cu_seqlens_q),
            "actual_seq_kvlen": cumulative_lengths(cu_seqlens_k),
            "sparse_mode": 3 if causal else 0,
        }
        # sparse_mode=3 is the NPU causal mask mode and has bottom-right
        # alignment for q_len != kv_len, which is required by Lance KV cache.
        return torch_npu.npu_fusion_attention(q, k, v, **kwargs)[0]

    flash_module = types.ModuleType("flash_attn")
    flash_module.__path__ = []
    flash_module.__spec__ = ModuleSpec("flash_attn", loader=None, is_package=True)
    flash_module.flash_attn_varlen_func = flash_attn_varlen_func
    flash_module.__version__ = "npu-shim"
    sys.modules["flash_attn"] = flash_module

    # Lance's vision module normally falls back to SDPA when the real package is
    # absent.  Keep a rotary submodule available for dependency probes without
    # claiming that the complete CUDA package is installed.
    layers_module = types.ModuleType("flash_attn.layers")
    layers_module.__path__ = []
    layers_module.__spec__ = ModuleSpec("flash_attn.layers", loader=None, is_package=True)
    rotary_module = types.ModuleType("flash_attn.layers.rotary")
    rotary_module.__spec__ = ModuleSpec("flash_attn.layers.rotary", loader=None)

    def apply_rotary_emb(x: Any, cos: Any, sin: Any, interleaved: bool = False, **_: Any) -> Any:
        rotary_dim = cos.shape[-1] * 2
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        if interleaved:
            x_even, x_odd = x_rot[..., 0::2], x_rot[..., 1::2]
            output = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)
        else:
            x_first, x_second = x_rot.chunk(2, dim=-1)
            output = torch.cat((x_first * cos - x_second * sin, x_first * sin + x_second * cos), dim=-1)
        return torch.cat((output, x_pass), dim=-1)

    rotary_module.apply_rotary_emb = apply_rotary_emb
    sys.modules["flash_attn.layers"] = layers_module
    sys.modules["flash_attn.layers.rotary"] = rotary_module


def _patch_transformers_flash_attn_probe() -> None:
    """Make Transformers expose the process-local FlashAttention shim.

    ``transformers.utils.is_flash_attn_2_available`` checks installed package
    metadata in addition to importability.  The Lance NPU adapter deliberately
    installs an in-memory compatibility module rather than a fake Python
    distribution, so the stock probe returns ``False`` and Lance binds both
    ``flash_attn_varlen_func`` and ``apply_rotary_emb`` to ``None``.  Override
    only the probe imported by the dedicated Lance process after the shim has
    been installed.
    """
    transformers_utils = importlib.import_module("transformers.utils")
    current = getattr(transformers_utils, "is_flash_attn_2_available", None)
    if getattr(current, "_lance_npu_compatible", False):
        return

    def is_flash_attn_2_available() -> bool:
        return True

    is_flash_attn_2_available._lance_npu_compatible = True
    transformers_utils.is_flash_attn_2_available = is_flash_attn_2_available


def _patch_distributed_backend(torch: Any) -> None:
    dist = torch.distributed
    original = dist.init_process_group
    if getattr(original, "_lance_npu_compatible", False):
        return

    def init_process_group(backend: Optional[str] = None, *args: Any, **kwargs: Any) -> Any:
        if backend == "nccl":
            backend = "hccl"
        return original(backend, *args, **kwargs)

    init_process_group._lance_npu_compatible = True
    dist.init_process_group = init_process_group


def _patch_autocast(torch: Any) -> None:
    original = torch.amp.autocast
    if getattr(original, "_lance_npu_compatible", False):
        return

    def autocast(device_type: str, *args: Any, **kwargs: Any) -> Any:
        return original("npu" if device_type == "cuda" else device_type, *args, **kwargs)

    autocast._lance_npu_compatible = True
    torch.amp.autocast = autocast


def _patch_cuda_namespace(torch: Any) -> None:
    """Guarantee the CUDA compatibility spellings used by upstream Lance."""
    mappings = {
        "is_available": torch.npu.is_available,
        "device_count": torch.npu.device_count,
        "set_device": torch.npu.set_device,
        "empty_cache": torch.npu.empty_cache,
        "ipc_collect": getattr(torch.npu, "ipc_collect", lambda: None),
    }
    for name, value in mappings.items():
        setattr(torch.cuda, name, value)


def _patch_device_mesh() -> None:
    """Translate the CUDA device spelling used by upstream FSDP mesh setup."""

    device_mesh = importlib.import_module("torch.distributed.device_mesh")
    original = device_mesh.init_device_mesh
    if getattr(original, "_lance_npu_compatible", False):
        return

    def init_device_mesh(device_type: str, *args: Any, **kwargs: Any) -> Any:
        return original("npu" if device_type == "cuda" else device_type, *args, **kwargs)

    init_device_mesh._lance_npu_compatible = True
    device_mesh.init_device_mesh = init_device_mesh


def enable_lance_ascend_runtime(execution_mode: str = "inference") -> LanceRuntimeInfo:
    """Install process-local compatibility hooks and return runtime metadata."""

    if execution_mode not in ("inference", "training"):
        raise ValueError("execution_mode must be 'inference' or 'training'")
    try:
        torch = importlib.import_module("torch")
        torch_npu = importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError(
            "Lance Ascend execution requires the MindSpeed-MM PyTorch/NPU environment "
            "(torch and torch_npu must both be installed)."
        ) from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("No available Ascend NPU was detected")

    # This is the compatibility mechanism already used for third-party models in
    # MindSpeed-MM.  It rewrites Tensor.cuda()/Module.cuda() and CUDA namespaces.
    importlib.import_module("torch_npu.contrib.transfer_to_npu")
    _patch_cuda_namespace(torch)
    _patch_distributed_backend(torch)
    _patch_autocast(torch)
    if execution_mode == "training":
        _patch_device_mesh()
    _install_flash_attn_shim(
        torch,
        torch_npu,
        allow_dropout=execution_mode == "training",
    )
    _patch_transformers_flash_attn_probe()

    return LanceRuntimeInfo(
        execution_mode=execution_mode,
        device_type="npu",
        distributed_backend="hccl",
        torch_version=str(getattr(torch, "__version__", "unknown")),
        torch_npu_version=str(getattr(torch_npu, "__version__", "unknown")),
        flash_attention_backend="torch_npu.npu_fusion_attention:TND",
    )


def attention_smoke_test() -> Dict[str, object]:
    """Numerically compare Lance's NPU KV-cache attention with a CPU reference."""
    runtime = enable_lance_ascend_runtime()
    torch = importlib.import_module("torch")
    flash_attn_varlen_func = sys.modules["flash_attn"].flash_attn_varlen_func
    torch.manual_seed(1234)
    q_lengths = (3, 2)
    kv_lengths = (5, 4)
    q_heads, kv_heads, head_dim = 16, 2, 128
    q_cpu = torch.randn((sum(q_lengths), q_heads, head_dim), dtype=torch.float32)
    k_cpu = torch.randn((sum(kv_lengths), kv_heads, head_dim), dtype=torch.float32)
    v_cpu = torch.randn((sum(kv_lengths), kv_heads, head_dim), dtype=torch.float32)
    q = q_cpu.to(device="npu", dtype=torch.bfloat16)
    k = k_cpu.to(device="npu", dtype=torch.bfloat16)
    v = v_cpu.to(device="npu", dtype=torch.bfloat16)
    cu_q = torch.tensor((0, 3, 5), device="npu", dtype=torch.int32)
    cu_k = torch.tensor((0, 5, 9), device="npu", dtype=torch.int32)
    actual = flash_attn_varlen_func(q, k, v, cu_q, cu_k, 3, 5, causal=True).float().cpu()

    references = []
    q_start = kv_start = 0
    groups = q_heads // kv_heads
    for q_length, kv_length in zip(q_lengths, kv_lengths):
        query = q_cpu[q_start:q_start + q_length].transpose(0, 1)
        key = k_cpu[kv_start:kv_start + kv_length].repeat_interleave(groups, dim=1).transpose(0, 1)
        value = v_cpu[kv_start:kv_start + kv_length].repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        row = torch.arange(q_length).unsqueeze(1)
        column = torch.arange(kv_length).unsqueeze(0)
        allowed = column <= row + (kv_length - q_length)
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        references.append(torch.matmul(torch.softmax(scores, dim=-1), value).transpose(0, 1))
        q_start += q_length
        kv_start += kv_length
    expected = torch.cat(references, dim=0)
    difference = (actual - expected).abs()
    max_error = float(difference.max().item())
    mean_error = float(difference.mean().item())
    tolerance = 0.08
    return {
        "runtime": runtime.to_dict(),
        "shape": list(actual.shape),
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "tolerance": tolerance,
        "status": "passed" if max_error <= tolerance else "failed",
    }
