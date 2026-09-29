# LEGACY (mlx-vlm < 0.5 only).
# apply_local_patches.sh skips this file when mlx-vlm ≥ 0.5, because those
# releases ship a full sample_utils with make_sampler / make_logits_processors.
# Overwriting 0.5+ with this stub breaks mlx_vlm.generate imports.
import mlx.core as mx


def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output. Shape [vocab], [B, vocab],
                or [B, draft_tokens, vocab] (MTP verification path).
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion. Shape [], [B], or [B, draft_tokens].
    """
    unbatched = logits.ndim == 1
    if unbatched:
        logits = logits[None]

    # MTP verification passes 3-D logits [B, draft_tokens, vocab].
    # Flatten the leading dims so the rest of the function sees [N, vocab], then
    # restore the original batch shape before returning.
    extra_shape = None
    if logits.ndim > 2:
        extra_shape = logits.shape[:-1]          # e.g. (1, 4)
        logits = logits.reshape(-1, logits.shape[-1])   # (4, vocab)

    if (
        logits.dtype == mx.bfloat16
    ):  # workaround for unable to load kernel contiguous_scan_inclusive_sum_bfloat16_bfloat16
        logits = logits.astype(mx.float32)

    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits / temperature, axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        mx.zeros_like(sorted_probs),
    )

    sampled_pos = mx.random.categorical(mx.log(top_probs))
    token = mx.take_along_axis(sorted_indices, sampled_pos[:, None], axis=-1).squeeze(
        -1
    )

    if extra_shape is not None:
        token = token.reshape(extra_shape)

    return token.squeeze(0) if unbatched else token
