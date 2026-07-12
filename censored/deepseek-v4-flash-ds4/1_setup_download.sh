#!/usr/bin/env bash
# =============================================================================
# 1_setup_download.sh — Clone/build DwarfStar (antirez/ds4) + download GGUF
#
# Native Metal inference engine for DeepSeek V4 Flash (and PRO on huge RAM).
# Not a generic GGUF runner — only antirez's published GGUFs work.
#
# Engine: https://github.com/antirez/ds4
# Weights: https://huggingface.co/antirez/deepseek-v4-gguf
#
# Best: hard coding/agent work when quality > latency; long context; tool-calling
#       agents that speak OpenAI/Anthropic APIs (ds4-server)
# OK:   interactive CLI (./ds4), native ds4-agent, multi-turn chat
# Bad:  snappy tool loops (prefer Qwen 3.6); machines without enough RAM for the
#       chosen quant (use --ssd-streaming only if you accept slower generation)
#
# Model targets (first arg, passed to download_model.sh):
#   q2-imatrix      ~81 GB  — default, 96/128 GB Macs
#   q2-q4-imatrix   ~98 GB  — higher quality, 128 GB Macs
#   q4-imatrix      ~153 GB — needs ≥256 GB RAM (or SSD streaming)
#   pro-q2-imatrix  ~430 GB — PRO; 512 GB machines (or experimental streaming)
#   mtp             ~3.5 GB — optional speculative component for Flash quants
#
# Usage:
#   ./1_setup_download.sh                 # clone + make + q2-imatrix
#   ./1_setup_download.sh q2-q4-imatrix   # higher-quality Flash mix
#   ./1_setup_download.sh --skip-download # clone + build only
#   ./1_setup_download.sh --rebuild       # force make clean + make
#   ./1_setup_download.sh --force-download  # remove local GGUF and re-fetch
#   DS4_REPO_URL=... DS4_REF=main ./1_setup_download.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS4_DIR="${DS4_DIR:-$SCRIPT_DIR/ds4}"
DS4_REPO_URL="${DS4_REPO_URL:-https://github.com/antirez/ds4.git}"
DS4_REF="${DS4_REF:-main}"
CONFIG_FILE="$SCRIPT_DIR/.ds4_config"
VALIDATE_PY="$SCRIPT_DIR/validate_model.py"

MODEL_TARGET="q2-imatrix"
SKIP_DOWNLOAD=false
FORCE_REBUILD=false
FORCE_DOWNLOAD=false

# Retry budget for network flakiness on multi-GB curl transfers.
MAX_DOWNLOAD_ATTEMPTS="${MAX_DOWNLOAD_ATTEMPTS:-25}"
RETRY_BASE_SECONDS="${RETRY_BASE_SECONDS:-5}"
RETRY_MAX_SECONDS="${RETRY_MAX_SECONDS:-120}"

# Exact byte sizes (HF x-linked-size) — bash 3.2 compatible (no assoc arrays).
expected_bytes_for() {
    case "$1" in
        q2-imatrix)    echo 86720111488 ;;
        q2-q4-imatrix) echo 97591747456 ;;
        q4-imatrix)    echo 164633502592 ;;
        mtp)           echo 3807602400 ;;
        *)             echo "" ;;
    esac
}

model_file_for_target() {
    case "$1" in
        q2-imatrix)
            echo "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
            ;;
        q2-q4-imatrix)
            echo "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
            ;;
        q4-imatrix)
            echo "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
            ;;
        pro-q2-imatrix)
            echo "DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
            ;;
        mtp)
            echo "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
            ;;
        *)
            echo ""
            ;;
    esac
}

for arg in "$@"; do
    case "$arg" in
        q2-imatrix|q2-q4-imatrix|q4-imatrix|pro-q2-imatrix|pro-q4-layers00-30|pro-q4-layers31-output|pro-q4-split|mtp)
            MODEL_TARGET="$arg"
            ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
        --rebuild) FORCE_REBUILD=true ;;
        --force-download|--force) FORCE_DOWNLOAD=true ;;
        --help|-h)
            awk '/^# ===/{c++; if(c==2) exit} c==1{sub(/^# ?/,""); print}' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $arg"
            echo "       Run ./1_setup_download.sh --help"
            exit 1
            ;;
    esac
done

case "${MODEL_TARGET}" in
    q2-imatrix)     MODEL_DESC="~81 GB — Flash 2-bit imatrix (recommended for 128 GB)" ;;
    q2-q4-imatrix)  MODEL_DESC="~98 GB — Flash mixed q2/q4 imatrix (higher quality)" ;;
    q4-imatrix)     MODEL_DESC="~153 GB — Flash 4-bit imatrix (≥256 GB RAM)" ;;
    pro-q2-imatrix) MODEL_DESC="~430 GB — PRO 2-bit imatrix (512 GB / streaming)" ;;
    mtp)            MODEL_DESC="~3.5 GB — optional MTP speculative component" ;;
    *)              MODEL_DESC="see ds4/download_model.sh" ;;
esac

human_bytes() {
    local n="${1:-0}"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "n=int('${n}'); print(f'{n/1e9:.2f} GB')" 2>/dev/null && return
    fi
    echo "${n} B"
}

retry_delay() {
    local attempt="$1"
    local exp=$((attempt - 1))
    (( exp > 5 )) && exp=5
    local delay=$((RETRY_BASE_SECONDS * (1 << exp)))
    (( delay > RETRY_MAX_SECONDS )) && delay=$RETRY_MAX_SECONDS
    echo "$delay"
}

require_cmd() {
    local cmd="$1"
    local hint="${2:-}"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $cmd"
        [[ -n "$hint" ]] && echo "       $hint"
        exit 1
    fi
}

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "=== DwarfStar (ds4) Setup — DeepSeek V4 ==="
echo "→ Engine:  $DS4_REPO_URL @ $DS4_REF"
echo "→ Dir:     $DS4_DIR"
echo "→ Model:   $MODEL_TARGET  ($MODEL_DESC)"
echo ""

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: This stack targets macOS Metal. On Linux use make cuda-* inside ds4/."
    exit 1
fi

require_cmd git "Install Xcode CLT or git: xcode-select --install"
require_cmd curl "Install curl (macOS ships it)."
require_cmd make "Install Xcode Command Line Tools: xcode-select --install"
require_cmd cc "Install Xcode Command Line Tools: xcode-select --install"
require_cmd python3 "Install Python 3 (brew install python3)."

if ! xcode-select -p >/dev/null 2>&1; then
    echo "ERROR: Xcode Command Line Tools not found."
    echo "       Run: xcode-select --install"
    exit 1
fi

if [[ ! -f "$VALIDATE_PY" ]]; then
    echo "ERROR: missing validator: $VALIDATE_PY"
    exit 1
fi
chmod +x "$VALIDATE_PY" 2>/dev/null || true

# ── Clone ─────────────────────────────────────────────────────────────────────
if [[ ! -d "$DS4_DIR/.git" ]]; then
    if [[ -e "$DS4_DIR" ]]; then
        # Empty or non-git tree left from a failed clone — only auto-remove if
        # it looks like a broken clone (has no .git and no ds4.c).
        if [[ -f "$DS4_DIR/ds4.c" || -f "$DS4_DIR/Makefile" ]]; then
            echo "ERROR: $DS4_DIR exists but is not a git repo."
            echo "       Remove it or set DS4_DIR to a fresh path."
            exit 1
        fi
        echo "→ Removing incomplete non-git directory: $DS4_DIR"
        rm -rf "$DS4_DIR"
    fi
    echo "→ Cloning $DS4_REPO_URL → $DS4_DIR ..."
    clone_ok=false
    for attempt in $(seq 1 5); do
        if git clone --depth 1 --branch "$DS4_REF" "$DS4_REPO_URL" "$DS4_DIR"; then
            clone_ok=true
            break
        fi
        delay="$(retry_delay "$attempt")"
        echo "→ Clone attempt $attempt failed; retrying in ${delay}s ..."
        rm -rf "$DS4_DIR"
        sleep "$delay"
    done
    if [[ "$clone_ok" != true ]]; then
        echo "ERROR: git clone failed after retries."
        exit 1
    fi
else
    echo "→ Existing clone: $DS4_DIR"
    echo "→ Fetching updates (best-effort) ..."
    (
        cd "$DS4_DIR"
        # Preserve local object files / metal build; only update sources.
        git fetch --depth 1 origin "$DS4_REF" 2>/dev/null || true
        if git rev-parse --verify "origin/$DS4_REF" >/dev/null 2>&1; then
            # Prefer ff-only merge to avoid surprising force-checkouts of dirty trees.
            if git merge-base --is-ancestor HEAD "origin/$DS4_REF" 2>/dev/null; then
                git merge --ff-only "origin/$DS4_REF" 2>/dev/null \
                    || git checkout -B "$DS4_REF" "origin/$DS4_REF" 2>/dev/null \
                    || true
            else
                git checkout -B "$DS4_REF" "origin/$DS4_REF" 2>/dev/null || true
            fi
        fi
        echo "→ HEAD: $(git rev-parse --short HEAD) ($(git log -1 --pretty=%s))"
    )
fi

# ── Build Metal ───────────────────────────────────────────────────────────────
echo ""
echo "→ Building Metal binaries (make) ..."

build_ds4() {
    local jobs
    jobs="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    (
        cd "$DS4_DIR"
        if [[ "$FORCE_REBUILD" == true ]]; then
            make clean || true
        fi
        make -j"$jobs"
    )
}

if ! build_ds4; then
    echo "→ Build failed — cleaning and retrying once ..."
    (
        cd "$DS4_DIR"
        make clean || true
    )
    FORCE_REBUILD=false
    if ! build_ds4; then
        echo "ERROR: make failed after clean retry."
        echo "       Check Xcode CLT / Metal SDK, then re-run with --rebuild."
        exit 1
    fi
fi

for bin in ds4 ds4-server ds4-agent ds4-bench ds4-eval; do
    if [[ ! -x "$DS4_DIR/$bin" ]]; then
        echo "ERROR: expected binary missing: $DS4_DIR/$bin"
        exit 1
    fi
done
echo "→ Built: ds4, ds4-server, ds4-agent, ds4-bench, ds4-eval"
"$DS4_DIR/ds4" --help 2>&1 | head -5 || true
echo ""

# ── Model path helpers ────────────────────────────────────────────────────────
gguf_path_for_target() {
    local f
    f="$(model_file_for_target "$1")"
    if [[ -z "$f" ]]; then
        echo ""
        return
    fi
    echo "$DS4_DIR/gguf/$f"
}

model_is_complete() {
    local path="$1"
    [[ -e "$path" ]] || return 1
    python3 "$VALIDATE_PY" -q "$path" >/dev/null 2>&1
}

report_partial_progress() {
    local target="$1"
    local path part size expected pct
    path="$(gguf_path_for_target "$target")"
    [[ -n "$path" ]] || return 0
    part="${path}.part"
    expected="$(expected_bytes_for "$target")"
    if [[ -f "$path" ]]; then
        size="$(stat -f%z "$path" 2>/dev/null || echo 0)"
        if [[ -n "$expected" && "$size" -lt "$expected" ]]; then
            pct="$(python3 -c "print(f'{100*int($size)/int($expected):.1f}')" 2>/dev/null || echo '?')"
            echo "→ Incomplete GGUF at $path ($(human_bytes "$size") / $(human_bytes "$expected"), ${pct}%)"
        fi
        return 0
    fi
    if [[ -f "$part" ]]; then
        size="$(stat -f%z "$part" 2>/dev/null || echo 0)"
        if [[ -n "$expected" && "$expected" -gt 0 ]]; then
            pct="$(python3 -c "print(f'{100*int($size)/int($expected):.1f}')" 2>/dev/null || echo '?')"
            echo "→ Resumable partial: $part"
            echo "   Progress: $(human_bytes "$size") / $(human_bytes "$expected") (${pct}%)"
        else
            echo "→ Resumable partial: $part ($(human_bytes "$size"))"
        fi
    fi
}

ensure_disk_space() {
    local target="$1"
    local expected free_kb need
    expected="$(expected_bytes_for "$target")"
    [[ -n "$expected" ]] || return 0

    # Free space on the volume holding gguf/ (or ds4 root).
    mkdir -p "$DS4_DIR/gguf"
    free_kb="$(df -k "$DS4_DIR/gguf" | awk 'NR==2{print $4}')"
    # Need remaining bytes + 5 GB headroom (or full size if nothing downloaded yet).
    local path part have=0
    path="$(gguf_path_for_target "$target")"
    part="${path}.part"
    if [[ -f "$path" ]]; then
        have="$(stat -f%z "$path" 2>/dev/null || echo 0)"
    elif [[ -f "$part" ]]; then
        have="$(stat -f%z "$part" 2>/dev/null || echo 0)"
    fi
    need=$((expected - have + 5 * 1024 * 1024 * 1024))
    (( need < 0 )) && need=0
    local free_b=$((free_kb * 1024))
    if (( free_b < need )); then
        echo "ERROR: not enough free disk space under $DS4_DIR/gguf"
        echo "       Need ~$(human_bytes "$need") more; free $(human_bytes "$free_b")"
        echo "       Free space and re-run (download is resumable)."
        exit 1
    fi
    echo "→ Disk: free $(human_bytes "$free_b") (need up to ~$(human_bytes "$need") more for $target)"
}

force_remove_model() {
    local target="$1"
    local path part f
    path="$(gguf_path_for_target "$target")"
    [[ -n "$path" ]] || return 0
    part="${path}.part"
    f="$(model_file_for_target "$target")"
    echo "→ --force-download: removing local $f (and .part) ..."
    rm -f "$path" "$part"
    # Also drop default symlink if it pointed at this file
    if [[ -L "$DS4_DIR/ds4flash.gguf" ]]; then
        local link_tgt
        link_tgt="$(readlink "$DS4_DIR/ds4flash.gguf" 2>/dev/null || true)"
        if [[ "$link_tgt" == *"$f"* ]]; then
            rm -f "$DS4_DIR/ds4flash.gguf"
        fi
    fi
}

# ── Download GGUF (resumable + retried) ───────────────────────────────────────
if [[ "$SKIP_DOWNLOAD" == true ]]; then
    echo "→ Skipping weight download (--skip-download)."
    if [[ -L "$DS4_DIR/ds4flash.gguf" || -f "$DS4_DIR/ds4flash.gguf" ]]; then
        echo "→ Current default model link:"
        ls -lh "$DS4_DIR/ds4flash.gguf" 2>/dev/null || true
        if model_is_complete "$DS4_DIR/ds4flash.gguf"; then
            python3 "$VALIDATE_PY" "$DS4_DIR/ds4flash.gguf" || true
        else
            echo "→ WARNING: ds4flash.gguf incomplete or invalid — re-run without --skip-download"
            python3 "$VALIDATE_PY" "$DS4_DIR/ds4flash.gguf" 2>&1 || true
            report_partial_progress "$MODEL_TARGET"
        fi
    else
        echo "→ No ds4flash.gguf yet — re-run without --skip-download when ready."
        report_partial_progress "$MODEL_TARGET"
    fi
else
    if [[ "$FORCE_DOWNLOAD" == true ]]; then
        force_remove_model "$MODEL_TARGET"
    fi

    path="$(gguf_path_for_target "$MODEL_TARGET")"
    if [[ -n "$path" ]] && model_is_complete "$path"; then
        echo "→ $MODEL_TARGET already complete — skipping download"
        python3 "$VALIDATE_PY" "$path"
        # Ensure symlink exists for main Flash targets
        if [[ "$MODEL_TARGET" != mtp && "$MODEL_TARGET" != pro-q4-* ]]; then
            f="$(model_file_for_target "$MODEL_TARGET")"
            if [[ -n "$f" ]]; then
                ln -sfn "gguf/$f" "$DS4_DIR/ds4flash.gguf" 2>/dev/null \
                    || ln -sfn "$path" "$DS4_DIR/ds4flash.gguf"
            fi
        fi
    else
        report_partial_progress "$MODEL_TARGET"
        ensure_disk_space "$MODEL_TARGET"
        echo ""

        DOWNLOAD_PY="$SCRIPT_DIR/download_gguf.py"
        remote_file="$(model_file_for_target "$MODEL_TARGET")"
        # Prefer our resilient Range-resume downloader for known single-file Flash/MTP
        # targets. Upstream curl dies on "Connection reset by peer" without auto-resume.
        use_resilient=false
        if [[ -n "$path" && -n "$remote_file" && -f "$DOWNLOAD_PY" ]]; then
            case "$MODEL_TARGET" in
                q2-imatrix|q2-q4-imatrix|q4-imatrix|mtp) use_resilient=true ;;
            esac
        fi

        if [[ "$use_resilient" == true ]]; then
            echo "→ Downloading $MODEL_TARGET via download_gguf.py (resumable + auto-retry) ..."
            echo "   Partial progress is kept across connection resets."
            echo ""
            chmod +x "$DOWNLOAD_PY" 2>/dev/null || true
            # Build argv without empty-array expansion (bash 3.2 + set -u breaks "${arr[@]}").
            dl_cmd=(
                python3 "$DOWNLOAD_PY"
                --repo "antirez/deepseek-v4-gguf"
                --remote "$remote_file"
                --dest "$path"
                --validate-script "$VALIDATE_PY"
            )
            if [[ "$MODEL_TARGET" != mtp ]]; then
                dl_cmd+=(--link "$DS4_DIR/ds4flash.gguf")
            fi
            if [[ "$FORCE_DOWNLOAD" == true ]]; then
                dl_cmd+=(--force)
            fi

            set +e
            "${dl_cmd[@]}"
            rc=$?
            set -e
            if [[ $rc -ne 0 ]] || ! model_is_complete "$path"; then
                echo ""
                echo "ERROR: resilient download failed or incomplete."
                report_partial_progress "$MODEL_TARGET"
                echo "       Re-run the same command — it will resume the .part file:"
                echo "         ./1_setup_download.sh $MODEL_TARGET"
                exit 1
            fi
            python3 "$VALIDATE_PY" "$path"
            echo "→ Download complete for $MODEL_TARGET"
        else
            echo "→ Downloading $MODEL_TARGET via ds4/download_model.sh ..."
            echo "   (resumable; auto-retries on network errors up to $MAX_DOWNLOAD_ATTEMPTS attempts)"
            echo ""

            export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

            # Prepend resilient curl wrapper for upstream download_model.sh (PRO / splits).
            CURL_WRAP_DIR="$SCRIPT_DIR/bin"
            if [[ -x "$CURL_WRAP_DIR/curl-resilient" ]]; then
                chmod +x "$CURL_WRAP_DIR/curl-resilient" 2>/dev/null || true
                ln -sfn curl-resilient "$CURL_WRAP_DIR/curl"
                export DS4_REAL_CURL
                DS4_REAL_CURL="$(command -v curl)"
                if [[ "$DS4_REAL_CURL" == "$CURL_WRAP_DIR/curl" || "$DS4_REAL_CURL" == "$CURL_WRAP_DIR/curl-resilient" ]]; then
                    DS4_REAL_CURL="$(PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "^${CURL_WRAP_DIR}$" | paste -sd: -)" command -v curl || true)"
                fi
                if [[ -z "${DS4_REAL_CURL:-}" ]]; then
                    DS4_REAL_CURL="/usr/bin/curl"
                fi
                export PATH="$CURL_WRAP_DIR:$PATH"
                echo "→ curl wrapper: retries via $CURL_WRAP_DIR/curl → $DS4_REAL_CURL"
            fi

            download_ok=false
            for attempt in $(seq 1 "$MAX_DOWNLOAD_ATTEMPTS"); do
                set +e
                (
                    cd "$DS4_DIR"
                    ./download_model.sh "$MODEL_TARGET"
                )
                rc=$?
                set -e

                if [[ $rc -eq 0 ]]; then
                    if [[ -n "$path" ]]; then
                        if model_is_complete "$path"; then
                            download_ok=true
                            break
                        fi
                        echo "→ download_model.sh exited 0 but GGUF not complete yet"
                        report_partial_progress "$MODEL_TARGET"
                        if [[ -f "$path" ]] && ! model_is_complete "$path"; then
                            echo "→ Moving incomplete $path → ${path}.part for resume"
                            rm -f "${path}.part"
                            mv "$path" "${path}.part"
                        fi
                    else
                        download_ok=true
                        break
                    fi
                else
                    echo "→ Download attempt $attempt/$MAX_DOWNLOAD_ATTEMPTS failed (exit $rc)"
                    report_partial_progress "$MODEL_TARGET"
                fi

                if [[ $attempt -ge $MAX_DOWNLOAD_ATTEMPTS ]]; then
                    break
                fi
                delay="$(retry_delay "$attempt")"
                echo "→ Retrying in ${delay}s (partial progress is kept) ..."
                sleep "$delay"
            done

            if [[ "$download_ok" != true ]]; then
                echo ""
                echo "ERROR: model download failed or incomplete after $MAX_DOWNLOAD_ATTEMPTS attempts."
                report_partial_progress "$MODEL_TARGET"
                echo "       Check network / disk, then re-run: ./1_setup_download.sh $MODEL_TARGET"
                exit 1
            fi

            if [[ -n "$path" ]]; then
                python3 "$VALIDATE_PY" "$path"
            fi
            echo "→ Download complete for $MODEL_TARGET"
        fi
    fi
fi

# API model id used by ds4-server (Flash alias; PRO alias if you load PRO)
API_MODEL_ID="deepseek-v4-flash"
if [[ "$MODEL_TARGET" == pro-* ]]; then
    API_MODEL_ID="deepseek-v4-pro"
fi

cat > "$CONFIG_FILE" << EOF
# Written by 1_setup_download.sh — do not edit manually
DS4_DIR="${DS4_DIR}"
DS4_REPO_URL="${DS4_REPO_URL}"
DS4_REF="${DS4_REF}"
MODEL_TARGET="${MODEL_TARGET}"
API_MODEL_ID="${API_MODEL_ID}"
EOF

chmod +x "$SCRIPT_DIR/2_start_ds4.sh" "$VALIDATE_PY" 2>/dev/null || true

echo ""
echo "✅  Setup complete!"
echo ""
echo "  Engine:       $DS4_DIR"
echo "  Model target: $MODEL_TARGET"
echo "  Default GGUF: $DS4_DIR/ds4flash.gguf"
echo "  API model id: $API_MODEL_ID"
echo ""
echo "  Start OpenAI/Anthropic server:"
echo "    ./2_start_ds4.sh"
echo ""
echo "  One-shot CLI (no server):"
echo "    cd ds4 && ./ds4 -p 'Say hello in one sentence.' --nothink"
echo ""
echo "  Native coding agent:"
echo "    cd ds4 && ./ds4-agent --chdir \"$DS4_DIR\""
echo ""
