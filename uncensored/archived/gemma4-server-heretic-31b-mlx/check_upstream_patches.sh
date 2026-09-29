#!/usr/bin/env bash
# check_upstream_patches.sh — Compare local patches/ vs installed wheels.
#
# Usage:
#   ./check_upstream_patches.sh           # check active venv
#   ./check_upstream_patches.sh --apply   # apply patches then re-check
#   ./check_upstream_patches.sh --fetch   # download pristine wheels to /tmp first
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APPLY=false
FETCH=false
for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=true ;;
        --fetch) FETCH=true ;;
        --help|-h)
            echo "Usage: $0 [--apply] [--fetch]"
            echo "  --apply   Run apply_local_patches.sh before checking"
            echo "  --fetch   Download pristine PyPI wheels for diff baseline"
            exit 0
            ;;
    esac
done

if [[ "$APPLY" == true ]]; then
    chmod +x apply_local_patches.sh
    ./apply_local_patches.sh
fi

VENV_PY=""
for cand in "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/venv/bin/python3"; do
    if [[ -x "$cand" ]]; then
        VENV_PY="$cand"
        break
    fi
done
if [[ -z "$VENV_PY" ]]; then
    echo "ERROR: venv not found. Run ./1_setup_download.sh first." >&2
    exit 1
fi

PIP=("$VENV_PY" -m pip)
SITE="$("$VENV_PY" -c 'import site; print(site.getsitepackages()[0])')"
WHEEL_CACHE="${TMPDIR:-/tmp}/gemma4_upstream_wheels"
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

pass=0
warn=0
fail=0

ok()   { echo "  PASS  $*"; pass=$((pass + 1)); }
warn() { echo "  WARN  $*"; warn=$((warn + 1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail + 1)); }

section() { echo ""; echo "=== $1 ==="; }

pkg_version() {
    "$VENV_PY" -c "from importlib.metadata import version; print(version('$1'))" 2>/dev/null \
        || echo "unknown"
}

ver_ge() {
    local have="$1" need="$2"
    "$VENV_PY" - "$have" "$need" <<'PY'
import sys
def parts(v: str):
    out = []
    for p in v.split("."):
        digits = "".join(c for c in p if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)
print("yes" if parts(sys.argv[1]) >= parts(sys.argv[2]) else "no")
PY
}

mlx_vlm_ge_050() {
    ver_ge "$(pkg_version mlx-vlm)" "0.5.0"
}

file_contains() {
    local path="$1"
    local pattern="$2"
    [[ -f "$path" ]] && grep -q "$pattern" "$path"
}

files_differ() {
    local a="$1"
    local b="$2"
    [[ -f "$a" && -f "$b" ]] && ! cmp -s "$a" "$b"
}

fetch_wheel_file() {
    local pkg="$1"
    local ver="$2"
    local relpath="$3"
    local cache_key="${pkg}-${ver}-${relpath//\//_}"
    mkdir -p "$WHEEL_CACHE"
    if [[ -f "$WHEEL_CACHE/$cache_key" ]]; then
        cat "$WHEEL_CACHE/$cache_key"
        return 0
    fi
    local tmp
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/wheel.XXXXXX")"
    if ! "${PIP[@]}" download "${pkg}==${ver}" --no-deps -q -d "$tmp" 2>/dev/null; then
        rm -rf "$tmp"
        return 1
    fi
    local whl
    whl="$(ls "$tmp"/*.whl 2>/dev/null | head -1)"
    if [[ -z "$whl" ]]; then
        rm -rf "$tmp"
        return 1
    fi
    if ! unzip -p "$whl" "$relpath" > "$WHEEL_CACHE/$cache_key" 2>/dev/null; then
        rm -f "$WHEEL_CACHE/$cache_key"
        rm -rf "$tmp"
        return 1
    fi
    rm -rf "$tmp"
    cat "$WHEEL_CACHE/$cache_key"
}

section "Installed versions"
echo "  vllm-mlx:  $(pkg_version vllm-mlx)"
echo "  mlx-vlm:   $(pkg_version mlx-vlm)"
echo "  mlx-lm:    $(pkg_version mlx-lm)"
echo "  mlx:       $(pkg_version mlx)"
echo "  transformers: $(pkg_version transformers)"

VLLM_VER="$(pkg_version vllm-mlx)"
MLX_VLM_VER="$(pkg_version mlx-vlm)"
VLLM_GE_040="$(ver_ge "$VLLM_VER" "0.4.0")"

section "Upstream fixes now in mlx-vlm (no local patch needed when ≥ 0.5.0)"
if [[ "$(mlx_vlm_ge_050)" == "yes" ]]; then
    if file_contains "$SITE/mlx_vlm/models/gemma4/language.py" 'mx.array(cache.offset)'; then
        ok "mlx-vlm ≥ 0.5: Gemma4 RoPE offset defensive copy present"
    else
        bad "mlx-vlm ≥ 0.5 but Gemma4 language.py missing mx.array(cache.offset)"
    fi
    _gen_stream_file=""
    for _candidate in \
        "$SITE/mlx_vlm/generate.py" \
        "$SITE/mlx_vlm/generate/common.py" \
        "$SITE/mlx_vlm/generate/dispatch.py"; do
        if [[ -f "$_candidate" ]] && file_contains "$_candidate" 'new_thread_local_stream'; then
            _gen_stream_file="$_candidate"
            break
        fi
    done
    if [[ -n "$_gen_stream_file" ]]; then
        ok "mlx-vlm ≥ 0.5: thread-local generation stream present ($_gen_stream_file)"
    else
        warn "mlx-vlm ≥ 0.5 but no new_thread_local_stream in generate module"
    fi
else
    warn "mlx-vlm < 0.5.0 — install mlx-vlm>=0.5.0 (see requirements.txt)"
fi

section "Upstream fixes now in vllm-mlx (merged in 0.4.0+)"
if [[ "$VLLM_GE_040" == "yes" ]]; then
    if file_contains "$SITE/vllm_mlx/text_model_from_vlm.py" 'gemma4_text'; then
        ok "vllm-mlx ≥ 0.4: gemma4_text TextModel dispatch present (no local overwrite needed)"
    else
        bad "vllm-mlx ≥ 0.4 but gemma4_text dispatch missing"
    fi
    if file_contains "$SITE/vllm_mlx/api/models.py" 'logit_bias'; then
        ok "vllm-mlx ≥ 0.4: ChatCompletionRequest.logit_bias present"
    else
        bad "vllm-mlx ≥ 0.4 but logit_bias field missing on request model"
    fi
    if file_contains "$SITE/vllm_mlx/server.py" '_attach_logit_bias_processor\|make_logits_processors'; then
        ok "vllm-mlx ≥ 0.4: server logit_bias wiring present"
    else
        # grep -E style may not work with file_contains - check separately
        if file_contains "$SITE/vllm_mlx/server.py" 'logit_bias'; then
            ok "vllm-mlx ≥ 0.4: server logit_bias wiring present"
        else
            bad "vllm-mlx ≥ 0.4 but server logit_bias wiring missing"
        fi
    fi
    # Local 0.3.x full-file patches must NOT be applied on top of 0.4.0
    for pair in \
        "patches/text_model_from_vlm.py|$SITE/vllm_mlx/text_model_from_vlm.py|text_model_from_vlm" \
        "patches/api/models.py|$SITE/vllm_mlx/api/models.py|api/models" \
        "patches/engine/simple.py|$SITE/vllm_mlx/engine/simple.py|engine/simple" \
        "patches/server.py|$SITE/vllm_mlx/server.py|server"; do
        IFS='|' read -r localf instf label <<<"$pair"
        if [[ -f "$localf" && -f "$instf" ]] && cmp -s "$localf" "$instf"; then
            bad "$label: installed file is the old 0.3.x patch — reinstall vllm-mlx 0.4.0"
        fi
    done
else
    warn "vllm-mlx < 0.4.0 — upgrade recommended (see requirements.txt); local 0.3.x patches still apply"
    if file_contains "$SITE/vllm_mlx/text_model_from_vlm.py" 'gemma4_text'; then
        ok "text_model_from_vlm: gemma4_text present (patched or backported)"
    else
        bad "text_model_from_vlm: gemma4_text missing — run ./apply_local_patches.sh"
    fi
    if file_contains "$SITE/vllm_mlx/api/models.py" 'logit_bias'; then
        ok "api/models: logit_bias present"
    else
        bad "api/models: logit_bias missing — run ./apply_local_patches.sh"
    fi
fi

if [[ "$(mlx_vlm_ge_050)" == "no" ]]; then
    section "Legacy mlx-vlm < 0.5 patches"
    if file_contains "$SITE/mlx_vlm/generate.py" '_refresh_generation_stream'; then
        ok "mlx_vlm generate stream patch present"
    else
        bad "mlx_vlm generate stream patch missing"
    fi
elif [[ -f patches/generate.py ]] && files_differ "patches/generate.py" "$SITE/mlx_vlm/generate.py"; then
    section "Legacy mlx-vlm patches"
    ok "mlx_vlm/generate.py: using upstream (mlx-vlm ≥ 0.5); legacy patches/generate.py not applied"
else
    section "Legacy mlx-vlm patches"
    ok "mlx_vlm/generate.py: using upstream (mlx-vlm ≥ 0.5)"
fi

section "vllm-mlx gemma4_mllm runtime patch (still applied)"
if file_contains "$SITE/vllm_mlx/patches/gemma4_mllm.py" 'mask trim for BatchedEngine'; then
    ok "gemma4_mllm.py: mask-trim patch present"
elif file_contains "$SITE/vllm_mlx/patches/gemma4_mllm.py" 'BatchKVCache support'; then
    warn "gemma4_mllm.py: legacy offset patch — run ./apply_local_patches.sh after mlx-vlm upgrade"
else
    bad "gemma4_mllm.py: unknown or missing patch — run ./apply_local_patches.sh"
fi

section "PyPI upstream status (informational)"
echo "  vllm-mlx:                 0.4.0 includes gemma4_text + logit_bias"
echo "  mlx-vlm:                  ≥0.5.0 has RoPE offset + thread-local streams"
echo "  mlx-vlm constraint:       transformers>=5.5,<5.13 (do not install 5.13+ yet)"
echo "  mlx-lm:                   gemma4_text model class available in 0.31+"
echo "  still local:              gemma4_mllm mask-trim + gemma4_mlx_kilo_proxy"

section "Summary"
echo "  PASS=$pass  WARN=$warn  FAIL=$fail"
if [[ "$fail" -gt 0 ]]; then
    echo ""
    echo "Fix: ./apply_local_patches.sh   or   pip install -r requirements.txt && ./apply_local_patches.sh"
    exit 1
fi
if [[ "$warn" -gt 0 ]]; then
    exit 2
fi
exit 0
