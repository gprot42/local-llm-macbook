# Native 1.0.x support for Ternary Bonsai 2 (`model_type: prism_hadamard_qwen35`)?

### Summary

The 1.0.0 release notes list **Bonsai** among the models "being ported for 1.0.x".
I'd like to confirm whether that port is intended to cover the **2-bit Hadamard**
Bonsai pack specifically — `model_type: prism_hadamard_qwen35`
(`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`) — or only a different Bonsai quant, and
roughly where it sits on the 1.0.x roadmap. This is a status/clarification question,
not a bug report: the native engine already refuses the checkpoint cleanly and points
back to the 0.6 line.

### What the native binary does today (v1.0.5)

```
$ tensorfold info prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
prism-ml/Ternary-Bonsai-2-27B-mlx-2bit: no registered Zig family serves
model_type prism_hadamard_qwen35; the 0.6 line may: tensorfold@0.6
```

Clear and actionable — no complaint there. I just can't tell from the notes whether
`prism_hadamard_qwen35` is on the native roadmap or out of scope.

### The same checkpoint on the Python engine (v0.6.6)

v0.6.6 serves it fine, so this is purely about the native port:

```
$ tensorfold info prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
model_type        prism_hadamard_qwen35
family            Ternary Bonsai 2 (tensorfold.families.bonsai)
engine            MLX lane engine
kernels           qwen/prism/v1
num_hidden_layers 64
hidden_size       5120
vocab_size        248320
```

Verified on the hardware below: it serves, and a drafted reply
(`z-lab/Qwen3.8-27B-DFlash2`) is byte-identical to the same request with
`"draft": false`.

### Pack shape

- `model_type: prism_hadamard_qwen35` — ternary weights in a rotated (Hadamard) basis,
  2-bit, groups of 128; the version-1 Prism Hadamard contract.
- Qwen3.5-family dimensions (64 layers, hidden 5120), vocab 248320, untied embeddings,
  no MTP head. A vision tower ships in the pack but isn't loaded for text serving.

### Questions

1. Is `prism_hadamard_qwen35` (the 2-bit Hadamard Bonsai pack) in scope for the native
   1.0.x port, or does "Bonsai" in the 1.0.0 notes refer to a different quant/contract?
2. If it's in scope, is there a tracking issue or rough milestone to follow?
3. Anything pack-side that would help (e.g. a specific export/contract version the native
   `qwen/prism` kernels will expect)?

Happy to test a native build against this pack and report back — it reproduces on a
single command above.

### Environment

- TensorFold 1.0.5 (native, Homebrew) and 0.6.6 (Python, pip)
- Apple M5 Max, 128 GB, macOS 27.0.1
- Checkpoint: `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`; drafter `z-lab/Qwen3.8-27B-DFlash2`

_Related: #18 (MLX support request) — raising this separately since it's about one
specific `model_type` on the native engine rather than MLX support in general; happy to
fold it in there if you'd prefer._
