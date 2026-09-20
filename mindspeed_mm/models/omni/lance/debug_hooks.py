"""Optional debug instrumentation for native Lance training.

Zero-behavior-change design: every hook is a no-op unless LANCE_DEBUG=1 is set
in the environment, so the production training path is untouched.

Output goes to stdout (rank-gated) AND a per-rank JSONL file so progress
survives the launcher tty. Default sink: <training.save>/debug_rank{RANK}.jsonl.

Environment switches
--------------------
LANCE_DEBUG=1              master switch (default off)
LANCE_DEBUG_EVERY=N        heavy-dump interval in forward calls (default 1)
LANCE_DEBUG_RANKS=0|all    comma list of ranks that print (default "0")
LANCE_DEBUG_LOG=path       JSONL sink override
LANCE_DEBUG_TOKENIZER=path optional HF tokenizer dir used to decode text ids
LANCE_DEBUG_DUMP_TOKENS=N  how many leading text token ids to dump (default 32)

What is covered (per operator request)
--------------------------------------
1. token composition entering forward: text ids, image latent tokens, MoT
   expert routing, latent geometry (token count -> implied pixel resolution)
2. attention-mask structure (packed documents / segments / attention blocks)
   plus per-step loss components (ce / mse / weights / totals)
3. per-training-step wall time (every step into JSONL + window statistics)
4. per-step device memory (allocated / reserved / max / HBM info) and an
   analytic activation budget derived from the actual batch shape:

   activation estimate (bf16 = 2 bytes), sequence length L, hidden H, layers N,
   MLP intermediate I, CE tokens C, MSE/VAE tokens V, patch dim P:
     - embedding staging buffer          L * H * 2
     - per-layer boundary (grad ckpt ON) N * L * H * 2
     - per-layer full activations (OFF)  N * (k_attn * L * H + k_mlp * L * I) * 2
       (k_attn ~= 34H standard Transformer accounting, k_mlp ~= 5 for
        gate/up/down MLPs; MoT routes each token to one expert, so the seq-level
        figure does not double, but compute does)
     - CE logits                         C * vocab * 2 (+ fp32 copy *4 in loss)
     - latent bridge tensors             ~6 * V * P * 2
       (clean / noise / noisy / velocity target / prediction / log-var scratch)
     - image token geometry: V = (H_px / 8 / patch_h) * (W_px / 8 / patch_w)
       with VAE spatial compression 8 and latent patch 2x2 (P = 48 * 4 = 192)
"""

import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional

import torch

_PREFIX = "[LANCE-DBG]"
_call_counter = {"forward": 0, "step": 0}
_window: List[float] = []
_LOG_PATH: Dict[int, str] = {}
_TOKENIZER = None


def _enabled() -> bool:
    return os.environ.get("LANCE_DEBUG", "0") == "1"


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _print_ranks():
    raw = os.environ.get("LANCE_DEBUG_RANKS", "0")
    if raw.strip().lower() == "all":
        return None
    try:
        return {int(piece) for piece in raw.split(",") if piece.strip()}
    except ValueError:
        return {0}


def _should_print() -> bool:
    ranks = _print_ranks()
    return ranks is None or _rank() in ranks


def _every() -> int:
    try:
        return max(1, int(os.environ.get("LANCE_DEBUG_EVERY", "1")))
    except ValueError:
        return 1


def _say(message: str) -> None:
    if _should_print():
        print("{} r{} {}".format(_PREFIX, _rank(), message), flush=True)


def _log_path(save_dir: Optional[str]) -> str:
    key = _rank()
    if key not in _LOG_PATH:
        override = os.environ.get("LANCE_DEBUG_LOG", "")
        if override:
            path = override.replace("{RANK}", str(key))
        else:
            base = save_dir or os.environ.get("LANCE_OUTPUT_DIR", "/tmp")
            path = os.path.join(base, "debug_rank{}.jsonl".format(key))
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except OSError:
            path = "/tmp/lance_debug_rank{}.jsonl".format(key)
        _LOG_PATH[key] = path
    return _LOG_PATH[key]


def _record(payload: Dict[str, Any], save_dir: Optional[str] = None) -> None:
    payload = dict(payload)
    payload.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
    payload.setdefault("rank", _rank())
    try:
        with open(_log_path(save_dir), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    except OSError as error:
        _say("jsonl write failed: {}".format(error))


def _n(tensor: Optional[torch.Tensor]) -> int:
    return 0 if tensor is None else int(tensor.numel())


def remember_batch(engine, batch) -> None:
    """Keep the CPU-side batch on the engine for the activation budget.

    Only stores a reference when LANCE_DEBUG=1; otherwise a strict no-op that
    never touches the batch (no extra lifetime, no behavior change).
    """
    if _enabled():
        try:
            engine._last_debug_batch = batch
        except Exception:
            pass


def _maybe_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    path = os.environ.get("LANCE_DEBUG_TOKENIZER", "")
    if not path:
        _TOKENIZER = False
        return None
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    except Exception as error:  # tokenizer is a convenience; never crash training
        _say("tokenizer load failed ({}): {}".format(path, error))
        _TOKENIZER = False
    return _TOKENIZER or None


def _fmt_ids(values: List[int]) -> str:
    limit = max(4, int(os.environ.get("LANCE_DEBUG_DUMP_TOKENS", "32")))
    head = " ".join(str(v) for v in values[:limit])
    return head + (" ... (+{} more)".format(len(values) - limit) if len(values) > limit else "")


# ---------------------------------------------------------------------------
# Hook 1 + 2a: what the forward actually consumes (tokens + attention mask)
# ---------------------------------------------------------------------------


def _segment_summary(packed) -> List[Dict[str, Any]]:
    documents = list(getattr(packed, "documents", ()) or ())
    segments = []
    for document in documents:
        for segment in getattr(document, "segments", ()):  # type: ignore[union-attr]
            segments.append(
                {
                    "sample_id": getattr(document, "sample_id", "?"),
                    "len": getattr(segment, "length", 0),
                    "mode": getattr(segment, "attention_mode", "?"),
                    "modality": getattr(segment, "modality", "?"),
                    "expert": getattr(segment, "expert", "?"),
                }
            )
    return segments


def hook_forward_inputs(model, batch) -> None:
    """Dump the packed-sequence composition and attention-mask structure."""
    if not _enabled():
        return
    try:
        _call_counter["forward"] += 1
        if _call_counter["forward"] % _every() != 1 and _every() != 1:
            return
        seq = int(batch.sequence_length)
        n_text = _n(batch.text_indexes)
        n_vit = _n(batch.vit_indexes)
        n_vae = _n(batch.vae_indexes)
        n_ce = _n(batch.ce_indexes)
        n_mse = _n(batch.mse_indexes)
        n_und = _n(batch.understanding_indexes)
        n_gen = _n(batch.generation_indexes)

        # Implied pixel geometry from the latent token count: VAE f8 + 2x2 patch.
        side = int(round((n_vae) ** 0.5)) if n_vae else 0
        pixel_note = ""
        if n_vae and side * side == n_vae:
            pixel_note = " (~{}x{} px @VAE-f8, patch 2x2)".format(side * 16, side * 16)

        _say(
            "forward#{} seq={} | text={} vit={} vae={}{} | ce={} mse={} | MoT und/gen={}/{}".format(
                _call_counter["forward"], seq, n_text, n_vit, n_vae, pixel_note,
                n_ce, n_mse, n_und, n_gen,
            )
        )

        text_ids = batch.token_ids[batch.text_indexes].tolist()
        _say("  text token ids: [{}]".format(_fmt_ids(text_ids)))
        tokenizer = _maybe_tokenizer()
        if tokenizer is not None and text_ids:
            try:
                _say("  decoded text: {!r}".format(tokenizer.decode(text_ids)))
            except Exception as error:
                _say("  decode failed: {}".format(error))
        if n_vae:
            _say(
                "  latents: clean{} logvar={} | pos[{},{}] | timesteps sentinel>0: {}".format(
                    tuple(batch.clean_latents.shape),
                    None if batch.latent_log_variance is None else tuple(batch.latent_log_variance.shape),
                    int(batch.latent_position_ids.min().item()),
                    int(batch.latent_position_ids.max().item()),
                    int((batch.timesteps > 0).sum().item()),
                )
            )
        if n_ce:
            _say(
                "  ce labels: [{}]".format(
                    _fmt_ids(batch.ce_labels.tolist())
                )
            )

        mask = batch.attention_mask
        if mask is None:
            _say("  attention_mask: None")
        elif hasattr(mask, "documents"):
            segments = _segment_summary(mask)
            _say(
                "  attention_mask: LancePackedSequence len={} docs={} segments={}".format(
                    getattr(mask, "length", "?"), len(getattr(mask, "documents", ())), len(segments)
                )
            )
            for piece in segments[:12]:
                _say("    seg {}".format(piece))
            if len(segments) > 12:
                _say("    ... (+{} more segments)".format(len(segments) - 12))
            try:
                blocks = mask.block_schedule()
                causal_blocks = sum(1 for block in blocks if getattr(block, "causal", False))
                _say(
                    "  attention blocks: {} (causal {}, full {})".format(
                        len(blocks), causal_blocks, len(blocks) - causal_blocks
                    )
                )
            except Exception as error:
                _say("  block schedule unavailable: {}".format(error))
        else:
            dense = torch.as_tensor(mask)
            _say(
                "  attention_mask: dense {} dtype={} density={:.4f}".format(
                    tuple(dense.shape), dense.dtype, float(dense.float().mean().item())
                )
            )

        payload = {
            "event": "forward_inputs",
            "call": _call_counter["forward"],
            "seq": seq,
            "text": n_text,
            "vit": n_vit,
            "vae": n_vae,
            "ce": n_ce,
            "mse": n_mse,
            "mot_und": n_und,
            "mot_gen": n_gen,
            "text_ids": text_ids[:256],
            "segments": _segment_summary(mask) if hasattr(mask or object(), "documents") else None,
        }
        _record(payload)
    except Exception as error:  # debug code must never break training
        _say("hook_forward_inputs error: {!r}".format(error))


# ---------------------------------------------------------------------------
# Hook 2b: loss computation internals
# ---------------------------------------------------------------------------


def hook_losses(batch, ce_loss, mse_loss, ce_weight, mse_weight, total_loss) -> None:
    """Dump the joint CE/MSE loss decomposition for this forward."""
    if not _enabled():
        return
    try:
        def _scalar(value):
            return None if value is None else float(value.detach().float().item())

        _say(
            "loss: total={:.6E} ce={:.6E} (weight={:.1f} tok) mse={:.6E} (weight={:.1f} tok)".format(
                _scalar(total_loss) or float("nan"),
                _scalar(ce_loss) or 0.0,
                ce_weight if isinstance(ce_weight, float) else _scalar(ce_weight) or 0.0,
                _scalar(mse_loss) or 0.0,
                mse_weight if isinstance(mse_weight, float) else _scalar(mse_weight) or 0.0,
            )
        )
        _record(
            {
                "event": "losses",
                "call": _call_counter["forward"],
                "total": _scalar(total_loss),
                "ce": _scalar(ce_loss),
                "mse": _scalar(mse_loss),
                "ce_weight": _scalar(ce_weight),
                "mse_weight": _scalar(mse_weight),
            }
        )
    except Exception as error:
        _say("hook_losses error: {!r}".format(error))


# ---------------------------------------------------------------------------
# Hooks 3 + 4: per-step timing, memory, and the analytic activation budget
# ---------------------------------------------------------------------------


def _accel_module():
    module = getattr(torch, "npu", None)
    if module is not None and hasattr(module, "memory_allocated"):
        return module
    module = getattr(torch, "cuda", None)
    if module is not None and hasattr(module, "memory_allocated"):
        return module
    return None


def _memory_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    module = _accel_module()
    if module is None:
        return {"available": False}
    try:
        out["allocated_GB"] = round(module.memory_allocated() / 2**30, 3)
        out["reserved_GB"] = round(module.memory_reserved() / 2**30, 3)
        out["max_allocated_GB"] = round(module.max_memory_allocated() / 2**30, 3)
        out["max_reserved_GB"] = round(module.max_memory_reserved() / 2**30, 3)
    except Exception as error:
        out["error"] = repr(error)
    try:  # device-level (shared across ranks); meaningful on the printing rank
        free, total = module.mem_get_info()
        out["hbm_free_GB"] = round(free / 2**30, 3)
        out["hbm_total_GB"] = round(total / 2**30, 3)
    except Exception:
        pass
    return out


def _activation_budget(engine, batch) -> Dict[str, Any]:
    """Analytic activation memory for the actual batch, in bytes (bf16)."""
    try:
        model = engine.model
        config = getattr(getattr(model, "language_model", None), "config", None)
        if config is None:
            config = getattr(model, "config", None)
        hidden = int(getattr(config, "hidden_size", 0) or 0)
        layers = int(getattr(config, "num_hidden_layers", 0) or 0)
        inter = int(getattr(config, "intermediate_size", 0) or 0)
        vocab = int(
            getattr(model, "effective_vocab_size", None)
            or getattr(config, "vocab_size", 0)
            or 0
        )
        patch_dim = int(getattr(config, "patch_latent_dim", 0) or 192)
        grad_ckpt = bool(getattr(engine.args.model, "gradient_checkpointing", False))
        seq = int(batch.sequence_length)
        n_ce = _n(batch.ce_indexes)
        n_vae = _n(batch.vae_indexes)
        n_text = _n(batch.text_indexes)

        rows: List[List[Any]] = []
        rows.append(["hidden staging [seq,H]", seq * hidden * 2])
        if grad_ckpt:
            rows.append(["layer boundaries (ckpt ON) [N,L,H]", layers * seq * hidden * 2])
        else:
            rows.append(
                [
                    "layer activations (ckpt OFF, ~=34H+5I per layer)",
                    layers * (34 * seq * hidden + 5 * seq * inter) * 2,
                ]
            )
        if n_ce:
            rows.append(["CE logits bf16 [C,vocab]", n_ce * vocab * 2])
            rows.append(["CE logits fp32 copy [C,vocab]", n_ce * vocab * 4])
        if n_vae:
            rows.append(
                ["latent bridge tensors ~6x [V,P]", 6 * n_vae * patch_dim * 2]
            )
        total = sum(value for _, value in rows)
        return {
            "seq": seq,
            "text": n_text,
            "ce": n_ce,
            "vae": n_vae,
            "hidden": hidden,
            "layers": layers,
            "intermediate": inter,
            "vocab": vocab,
            "grad_ckpt": grad_ckpt,
            "rows": [[name, value, round(value / 2**30, 3)] for name, value in rows],
            "total_GB": round(total / 2**30, 3),
        }
    except Exception as error:
        return {"error": repr(error)}


def hook_step(engine, iteration: int, elapsed: float, loss, grad_norm) -> None:
    """Per-step wall time, loss, device memory, and activation budget."""
    if not _enabled():
        return
    try:
        _call_counter["step"] += 1
        save_dir = getattr(engine.args.training, "save", None) or None
        try:
            batch = engine._last_debug_batch
        except AttributeError:
            batch = None

        def _scalar(value):
            try:
                return None if value is None else float(value.detach().float().item())
            except Exception:
                return None

        metrics = getattr(engine, "last_metrics", {}) or {}
        record = {
            "event": "step",
            "iteration": int(iteration),
            "elapsed_s": round(float(elapsed), 4),
            "loss": _scalar(loss),
            "grad_norm": grad_norm if grad_norm is None else float(grad_norm),
            "ce": metrics.get("ce"),
            "mse": metrics.get("mse"),
            "ce_tokens": metrics.get("ce_tokens"),
            "mse_tokens": metrics.get("mse_tokens"),
            "memory": _memory_snapshot(),
        }
        if batch is not None:
            record["activation_budget"] = _activation_budget(engine, batch)
        _record(record, save_dir)

        _say(
            "step {} t={:.2f}s loss={:.4E} ce={:.4E} mse={:.4E} mem alloc/max {}/{} GB".format(
                iteration,
                elapsed,
                record["loss"] if record["loss"] is not None else float("nan"),
                metrics.get("ce") or 0.0,
                metrics.get("mse") or 0.0,
                record["memory"].get("allocated_GB", "?"),
                record["memory"].get("max_allocated_GB", "?"),
            )
        )

        # Window statistics on the same cadence the regular logger uses.
        _window.append(float(elapsed))
        log_interval = int(getattr(engine.args.training, "log_interval", 10) or 10)
        if len(_window) >= log_interval and iteration % log_interval == 0:
            ordered = sorted(_window)
            _say(
                "timing window ({} steps): mean={:.2f}s min={:.2f}s p50={:.2f}s p95={:.2f}s max={:.2f}s -> {:.1f} steps/h".format(
                    len(_window),
                    statistics.fmean(_window),
                    ordered[0],
                    ordered[len(ordered) // 2],
                    ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
                    ordered[-1],
                    3600.0 / statistics.fmean(_window),
                )
            )
            _window.clear()
    except Exception as error:
        _say("hook_step error: {!r}".format(error))
