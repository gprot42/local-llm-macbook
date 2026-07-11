#!/usr/bin/env bash
# Graft stock Gemma 4 31B vision tower into the Heretic checkpoint.
#
# Upstream mlx-community Heretic 4-bit only ships language_model.* tensors, so
# mlx-vlm / vllm-mlx MLLM load fails with missing vision_tower parameters.
# We keep Heretic language weights (uncensored) and copy vision_tower +
# embed_vision from the stock IT 4-bit tree when available.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HERETIC_DIR="${1:-gemma-4-31b-heretic-mlx-4bit}"
STOCK_DIR="${2:-../gemma4-server-mlx-31b/gemma-4-31b-it-mlx-4bit}"
VENV_PY="$SCRIPT_DIR/venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv python not found at $VENV_PY" >&2
    exit 1
fi
if [[ ! -d "$HERETIC_DIR" ]]; then
    echo "ERROR: heretic model dir not found: $HERETIC_DIR" >&2
    exit 1
fi
if [[ ! -d "$STOCK_DIR" ]]; then
    echo "ERROR: stock model dir not found: $STOCK_DIR" >&2
    echo "Download stock 31B IT 4-bit first (sibling project gemma4-server-mlx-31b)." >&2
    exit 1
fi

"$VENV_PY" - "$HERETIC_DIR" "$STOCK_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

heretic = Path(sys.argv[1]).resolve()
stock = Path(sys.argv[2]).resolve()
h4 = heretic / "model-00004-of-00004.safetensors"
s4 = stock / "model-00004-of-00004.safetensors"
index_path = heretic / "model.safetensors.index.json"

if not h4.is_file() or not s4.is_file() or not index_path.is_file():
    raise SystemExit("missing heretic/stock shard4 or index")

idx = json.loads(index_path.read_text())
wm = idx.get("weight_map") or {}
vision_in_index = any(
    isinstance(k, str) and (k.startswith("vision_tower") or k.startswith("embed_vision"))
    for k in wm
)
if vision_in_index:
    # Confirm tensors actually exist in shard
    keys = list(mx.load(str(h4)).keys())
    if any(k.startswith("vision_tower") or k.startswith("embed_vision") for k in keys):
        print(f"→ Vision already present in {heretic.name} — nothing to do.")
        raise SystemExit(0)

print(f"→ Grafting vision from {stock.name} into {heretic.name} ...")
h_w = dict(mx.load(str(h4)))
s_w = dict(mx.load(str(s4)))
vision = {
    k: v
    for k, v in s_w.items()
    if k.startswith("vision_tower") or k.startswith("embed_vision")
}
if len(vision) < 100:
    raise SystemExit(f"stock shard4 has too few vision tensors: {len(vision)}")
overlap = set(h_w) & set(vision)
if overlap:
    raise SystemExit(f"unexpected key overlap with language weights: {list(overlap)[:5]}")

bak4 = heretic / "model-00004-of-00004.safetensors.text-only.bak"
if not bak4.exists():
    shutil.copy2(h4, bak4)
    print(f"   backed up text-only shard4 → {bak4.name}")
bak_idx = heretic / "model.safetensors.index.json.text-only.bak"
if not bak_idx.exists():
    shutil.copy2(index_path, bak_idx)

merged = {**h_w, **vision}
# mlx appends .safetensors if missing; use final name without double suffix
out_tmp = heretic / "model-00004-of-00004.grafting"
mx.save_safetensors(str(out_tmp), merged)
written = Path(str(out_tmp) + ".safetensors")
if not written.is_file():
    # some mlx versions write exactly the given path
    written = out_tmp if out_tmp.is_file() else None
if written is None or not written.is_file():
    raise SystemExit("mx.save_safetensors did not produce an output file")
written.replace(h4)
print(f"   wrote merged shard4 ({h4.stat().st_size / 1e9:.2f} GB), vision tensors={len(vision)}")

# Rebuild weight_map for vision keys
if bak_idx.exists():
    base = json.loads(bak_idx.read_text())
else:
    base = idx
wm = dict(base.get("weight_map") or {})
for k in vision:
    wm[k] = "model-00004-of-00004.safetensors"
total = sum((heretic / sh).stat().st_size for sh in sorted(set(wm.values())))
base["weight_map"] = wm
base.setdefault("metadata", {})
base["metadata"]["total_size"] = total
base["metadata"]["vision_graft"] = f"from {stock.name} vision_tower+embed_vision; language_model remains heretic"
index_path.write_text(json.dumps(base, indent=2) + "\n")
print(f"   index keys={len(wm)} total_size={total / 1e9:.2f} GB")

# Config: ensure vision_config + processor present
h_cfg_path = heretic / "config.json"
bak_cfg = heretic / "config.json.text-only.bak"
if not bak_cfg.exists():
    shutil.copy2(h_cfg_path, bak_cfg)
h_cfg = json.loads(h_cfg_path.read_text())
s_cfg = json.loads((stock / "config.json").read_text())
if not h_cfg.get("vision_config") and s_cfg.get("vision_config"):
    h_cfg["vision_config"] = s_cfg["vision_config"]
    print("   added vision_config from stock")
for key in (
    "vision_soft_tokens_per_image",
    "image_token_id",
    "boi_token_id",
    "eoi_token_id",
    "video_token_id",
    "audio_token_id",
    "audio_config",
):
    if key not in h_cfg and key in s_cfg:
        h_cfg[key] = s_cfg[key]
h_cfg["architectures"] = ["Gemma4ForConditionalGeneration"]
h_cfg_path.write_text(json.dumps(h_cfg, indent=4) + "\n")

proc_src = stock / "processor_config.json"
if proc_src.is_file():
    shutil.copy2(proc_src, heretic / "processor_config.json")
    print("   copied processor_config.json")

print("→ Vision graft complete (multimodal Heretic).")
PY
