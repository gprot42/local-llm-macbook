# SPDX-License-Identifier: Apache-2.0
"""Construct an mlx_lm TextModel from mlx_vlm-loaded model weights.

PATCHED for Gemma 4 (and other multimodal models that are NOT Qwen 3.5).

Upstream vllm-mlx 0.3.x hardcodes ``from mlx_lm.models.qwen3_5 import
TextModel, TextModelArgs`` at module level. For non-Qwen MLLMs this raises
``ZeroDivisionError`` inside ``TextModelArgs.__post_init__`` (head_dim
derivation), the fast TextModel path silently fails, and the engine falls
back to the slower mlx_vlm MLLM wrapper. The slow path is what was causing
streams to emit ~140 tokens and then loop on whitespace/end-of-turn,
producing the "empty-delta" stalls we now abort with a graceful stop.

This patch dispatches on ``text_config["model_type"]``:
  - ``gemma4_text``  → ``mlx_lm.models.gemma4_text.{Model, ModelArgs}``
  - anything else     → ``mlx_lm.models.qwen3_5.{TextModel, TextModelArgs}``
                        (original behavior preserved)

When mlx_vlm loads a model, it strips MTP weights in sanitize().
This module builds a parallel mlx_lm TextModel that:
1. Shares backbone + lm_head weights with the vlm model (zero-copy)
2. Loads MTP weights from safetensors on disk
3. Provides full mlx_lm API: return_hidden, n_confirmed, mtp_forward, make_mtp_cache
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

logger = logging.getLogger(__name__)


def _import_text_model_classes(model_type: str):
    """Return (TextModel, TextModelArgs) for the given model_type.

    Returns None if no compatible mlx_lm fast-path exists for this model.
    """
    if model_type == "gemma4_text":
        try:
            from mlx_lm.models.gemma4_text import (
                Model as TextModel,
                ModelArgs as TextModelArgs,
            )
            return TextModel, TextModelArgs
        except ImportError as e:
            logger.info(
                "mlx_lm.models.gemma4_text not available (%s); "
                "fast TextModel path disabled for gemma4_text",
                e,
            )
            return None

    # Default: Qwen 3.5 — original upstream behavior.
    # qwen3_5_moe.py does NOT export TextModel/TextModelArgs; the dense
    # qwen3_5 module handles both dense and MoE natively.
    try:
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
        return TextModel, TextModelArgs
    except ImportError as e:
        logger.info(
            "mlx_lm.models.qwen3_5 not available (%s); "
            "fast TextModel path disabled for model_type=%r",
            e,
            model_type,
        )
        return None


def build_text_model(vlm_model: Any, model_path: str | Path) -> Any | None:
    """Build an mlx_lm TextModel from a vlm-loaded model's weights.

    Args:
        vlm_model: The mlx_vlm-loaded model (has .language_model attribute)
        model_path: Path to the model directory (contains config.json + safetensors)

    Returns:
        mlx_lm TextModel with MTP support, or None on failure.
    """
    if vlm_model is None:
        return None

    model_path = Path(model_path) if model_path else None
    if model_path is None or not (model_path / "config.json").exists():
        return None

    try:
        config = json.loads((model_path / "config.json").read_text())
        text_config = config.get("text_config", config)

        # Dispatch on text_config.model_type so non-Qwen MLLMs (gemma4, etc.)
        # don't blow up inside Qwen 3.5's __post_init__.
        model_type = text_config.get("model_type") or config.get("model_type", "")
        imported = _import_text_model_classes(model_type)
        if imported is None:
            return None
        TextModel, TextModelArgs = imported

        # Build args via from_dict — it filters kwargs by signature, so
        # passing the full text_config is safe even if it has extra keys.
        args = TextModelArgs.from_dict(text_config)
        text_model = TextModel(args)

        # Collect all weights first: backbone from vlm + MTP from safetensors
        vlm_lm = vlm_model.language_model
        vlm_weights = mlx.utils.tree_flatten(vlm_lm.parameters())
        mtp_weights = _load_mtp_weights(model_path)

        all_weight_names = set(name for name, _ in vlm_weights)
        all_weight_names.update(name for name, _ in mtp_weights)

        # Quantize the TextModel skeleton to match source weights.
        # Mirror mlx_lm.utils._quantize semantics:
        #   - per-layer dict overrides come from config["quantization"][path]
        #   - otherwise quantize layers that have a matching `.scales` weight
        #
        # Per-layer keys in the source config use the full vlm namespace
        # (e.g. "language_model.model.layers.0.mlp.gate_proj"), but the
        # TextModel skeleton uses the stripped namespace, so we look up
        # both forms.
        quantization = text_config.get("quantization", config.get("quantization", None))
        if quantization is not None:
            model_quant_predicate = getattr(text_model, "quant_predicate", None)

            def _class_predicate(path, module):
                # Per-layer override from config["quantization"] (dict value).
                qcfg = quantization.get(path)
                if qcfg is None:
                    qcfg = quantization.get(f"language_model.{path}")
                if isinstance(qcfg, dict):
                    return qcfg

                if not hasattr(module, "to_quantized"):
                    return False

                # Only quantize layers we have scales for in the source.
                if f"{path}.scales" not in all_weight_names:
                    return False

                # Defer to the model's own quant_predicate if it provides
                # a finer-grained answer (e.g. gemma4_text returns a dict
                # for router.proj layers).
                if model_quant_predicate is not None:
                    model_decision = model_quant_predicate(path, module)
                    if isinstance(model_decision, dict):
                        return model_decision
                    if model_decision is False:
                        return False

                return True

            nn.quantize(
                text_model,
                group_size=quantization.get("group_size", 64),
                bits=quantization.get("bits", 8),
                mode=quantization.get("mode", "affine"),
                class_predicate=_class_predicate,
            )

        # Transfer backbone + lm_head weights from vlm language_model (zero-copy).
        # strict=False because TextModel has MTP params that vlm doesn't have yet.
        text_model.load_weights(vlm_weights, strict=False)

        logger.info(
            "Transferred %d weight arrays from vlm language_model (model_type=%s)",
            len(vlm_weights),
            model_type,
        )

        # Load MTP weights from safetensors
        if mtp_weights:
            text_model.load_weights(mtp_weights, strict=False)
            logger.info("Loaded %d MTP weights from safetensors", len(mtp_weights))
        else:
            logger.info(
                "No MTP weights in %s (expected for %s)",
                model_path.name,
                model_type,
            )

        # Inject MTP if TextModel doesn't have native MTP support.
        # mlx_lm's qwen3_5.TextModel strips MTP weights in sanitize(),
        # so we inject MTP module + methods at runtime. gemma4_text has
        # num_kv_shared_layers but no MTP layers, so this is a no-op there.
        if not hasattr(text_model, "mtp") or text_model.mtp is None:
            num_mtp = text_config.get("mtp_num_hidden_layers", 0)
            if num_mtp == 0:
                num_mtp = text_config.get("num_nextn_predict_layers", 0)
            if num_mtp > 0:
                from .patches.qwen3_5_mtp import inject_mtp_support

                inject_mtp_support(text_model, model_path, config)

        # Materialize ALL parameters on the current thread before returning.
        # nn.quantize + load_weights leave the parameter tree as lazy graph
        # nodes whose stream affinity is set when first evaluated. If we
        # return without evaluating, the first request's worker thread will
        # try to materialize them — but that thread runs under a freshly
        # bound generation_stream (see vllm_mlx.mlx_streams), and the lazy
        # nodes reference a stream that lives only on the build thread.
        # The result is "There is no Stream(gpu, N) in current thread."
        #
        # Evaluating here pins the arrays to the engine's owner thread
        # (event-loop thread for SimpleEngine), matching the same invariant
        # that BatchedEngine documents in engine/batched.py:257-261.
        mx.eval(text_model.parameters())

        if hasattr(text_model, "mtp") and text_model.mtp is not None:
            mx.eval(text_model.mtp.parameters())
            num_mtp = text_config.get(
                "mtp_num_hidden_layers",
                text_config.get("num_nextn_predict_layers", 0),
            )
            logger.info("TextModel built with MTP support (%d layers)", num_mtp)
        else:
            logger.info("TextModel built without MTP (model_type=%s)", model_type)

        # Warmup forward pass. Some models (notably gemma4_text) hold lazy
        # internal state outside `parameters()` — RoPE frequency tables, SDPA
        # masks, RotatingKVCache buffers — that gets created on FIRST forward
        # and binds to whatever stream is current then. If first forward runs
        # on a worker thread after `_bind_worker_generation_streams()`, those
        # buffers are pinned to a stream that does not exist on the build
        # thread (and vice versa), producing
        # "There is no Stream(gpu, N) in current thread."
        #
        # Running a 1-token forward HERE on the build thread forces all of
        # that lazy state to materialize on the current thread under the
        # default stream, where it stays reachable from any subsequent
        # generation thread.
        try:
            from mlx_lm.models.cache import make_prompt_cache

            warmup_cache = make_prompt_cache(text_model, max_kv_size=None)
            warmup_tokens = mx.array([[1]])
            warmup_out = text_model(warmup_tokens, cache=warmup_cache)
            mx.eval(warmup_out)
            mx.eval([c.state for c in warmup_cache])
            # Free the warmup cache so it can't accidentally be reused.
            del warmup_cache, warmup_out, warmup_tokens
            logger.info("TextModel warmup forward complete (stream-pinned)")
        except Exception as e:
            # Don't fail the whole build if warmup hits an edge case — log and
            # carry on; the first real request will surface any deeper issue.
            logger.warning("TextModel warmup forward failed: %s", e)

        return text_model

    except ImportError as e:
        logger.error("Cannot import mlx_lm TextModel: %s", e)
        return None
    except Exception as e:
        logger.error("Failed to build TextModel from vlm: %s", e)
        return None


def _load_mtp_weights(model_path: Path) -> list[tuple[str, mx.array]]:
    """Load MTP weights from safetensors, stripping the language_model. prefix.

    mlx_vlm's sanitize() strips mtp.* keys during model loading,
    but the weights are still on disk in the safetensors files.
    """
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return []

    index = json.loads(index_file.read_text())
    weight_map = index.get("weight_map", {})

    # Find MTP keys and their shard files
    mtp_keys: dict[str, tuple[str, str]] = {}
    for key, shard in weight_map.items():
        if ".mtp." in key:
            # Strip "language_model." prefix to match mlx_lm namespace
            clean = (
                key.replace("language_model.", "", 1)
                if key.startswith("language_model.")
                else key
            )
            mtp_keys[key] = (clean, shard)

    if not mtp_keys:
        return []

    # Group by shard to minimize I/O
    shards: dict[str, list[tuple[str, str]]] = {}
    for orig, (clean, shard) in mtp_keys.items():
        shards.setdefault(shard, []).append((orig, clean))

    weights = []
    for shard_file, key_pairs in shards.items():
        shard_path = model_path / shard_file
        if not shard_path.exists():
            logger.warning("MTP shard not found: %s", shard_file)
            continue
        shard_data = mx.load(str(shard_path))
        for orig, clean in key_pairs:
            if orig in shard_data:
                weights.append((clean, shard_data[orig]))

    return weights
