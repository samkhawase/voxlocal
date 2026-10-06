#!/usr/bin/env bash
#
# voxlocal — interactive voice agent launcher.
#
# Handles the things that make interactive use annoying:
#   * finds the binary (it is NOT in ./target — see .cargo/config.toml)
#   * preflights models and audio devices before you start talking
#   * degrades gracefully to --text / --wav when there is no usable microphone
#   * exits cleanly on Ctrl-C or SIGTERM instead of dumping a Rust backtrace
#
# Usage:
#   ./run.sh                          interactive mic loop
#   ./run.sh --text "..."             one-shot text turn
#   ./run.sh --wav /tmp/q.wav         full loop from a file
#   ./run.sh --selftest               pipeline check
#   ./run.sh --wav-regression         run samples/* sample through the full loop
#   ./run.sh --check                  preflight only, start nothing
#   ./run.sh --no-tts                 interactive without Piper/speakers

set -uo pipefail

cd "$(dirname "$0")" || exit 1

say()  { printf '\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33m!  %s\033[0m\n' "$1"; }
die()  { printf '\033[31mx  %s\033[0m\n' "$1" >&2; exit 1; }

# --- locate the binary -------------------------------------------------------
# target-dir is redirected to /tmp/vx to keep espeak-ng's build path short, so
# the default ./target/release/voxlocal does not exist.
BIN=""
if command -v cargo >/dev/null 2>&1; then
  BIN="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
          | tr ',' '\n' | grep '"target_directory"' | cut -d'"' -f4)/release/voxlocal"
fi
if [ ! -x "$BIN" ]; then
  # Fall back to a conventional location, then try building.
  for c in ./target/release/voxlocal ./target/debug/voxlocal; do
    [ -x "$c" ] && BIN="$c" && break
  done
fi
if [ ! -x "$BIN" ]; then
  say "Not built yet — building (first run compiles whisper.cpp, espeak-ng and ONNX Runtime; expect several minutes)"
  cargo build --release || die "build failed. See README 'Build requirements' (needs cmake)."
  BIN="$(cargo metadata --format-version 1 --no-deps \
          | tr ',' '\n' | grep '"target_directory"' | cut -d'"' -f4)/release/voxlocal"
fi
[ -x "$BIN" ] || die "could not locate the voxlocal binary"

# --- preflight ---------------------------------------------------------------
MODELS="${VOXLOCAL_MODELS:-.models}"
preflight() {
  local missing=0
  for f in llm/smollm2-135m-q4km.gguf llm/tokenizer.json \
           embed/config.json embed/model.safetensors embed/tokenizer.json \
           stt/ggml-tiny.en.bin tts/en_US-lessac-medium.onnx \
           tts/en_US-lessac-medium.onnx.json; do
    [ -s "$MODELS/$f" ] || { warn "missing model: $MODELS/$f"; missing=1; }
  done
  [ "$missing" -eq 0 ] || die "run ./setup.sh to download models into $MODELS/"

  # Is there a usable input device? Non-fatal: we fall back to other modes.
  local mic
  mic="$(system_profiler SPAudioDataType 2>/dev/null | grep -ci 'input channels' || true)"
  if [ "${mic:-0}" -eq 0 ]; then
    warn "no audio input device found"
    return 1
  fi
  return 0
}

# --- signal handling ---------------------------------------------------------
# Without this, Ctrl-C during a 5 s capture kills the process mid-stream and the
# shell prints the abort noise. Trap it, tear down cleanly, exit 0.
cleanup() {
  # Restore the cursor in case anything left it hidden.
  printf '\033[?25h' 2>/dev/null || true
  printf '\n  Bye.\n'
  exit 0
}
trap cleanup INT TERM

say "voxlocal ($BIN)"
preflight || true

# --- --check: preflight only -------------------------------------------------
for a in "$@"; do
  if [ "$a" = "--check" ]; then
    say "Preflight OK — models present, input device present"
    say "Nothing was started. Drop --check to run interactively."
    exit 0
  fi
done

# --- regression: run every sample through the full loop ------------------------
# A committed sample WAV lets CI / mic-less machines exercise the entire
# capture -> STT -> RAG -> LLM -> TTS path deterministically.
run_regression() {
  local dir="$(dirname "$0")/samples"
  local wavs=("$dir"/*.wav)
  [ -e "${wavs[0]}" ] || die "no samples in $dir"

  say "Running ${#wavs[@]} sample(s) through the full pipeline"
  local fail=0
  for w in "${wavs[@]}"; do
    say "  $(basename "$w")"
    if ! out="$("$BIN" --wav "$w" 2>&1)"; then
      warn "%s: pipeline errored" "$w"
      fail=1; continue
    fi
    # The LLM should name the car part it recognised; a bare "I only have
    # information about car parts" means STT/RAG regressed to out-of-domain.
    if ! printf '%s' "$out" | grep -qi "brake"; then
      warn "%s: expected 'brake' in output" "$w"
      fail=1
    fi
    say "  ok"
  done
  [ "$fail" -eq 0 ] || die "regression failed"
  say "all samples passed"
}

# --- choose a mode -----------------------------------------------------------
INTERACTIVE=1
PASSTHRU=()
for a in "$@"; do
  case "$a" in
    --text|--wav|--selftest|--probe-audio) INTERACTIVE=0; PASSTHRU+=("$a") ;;
    --text=*|--wav=*) INTERACTIVE=0; PASSTHRU+=("$a") ;;
    --no-tts) PASSTHRU+=("$a") ;;
    --wav-regression) run_regression; exit $? ;;
    *) PASSTHRU+=("$a") ;;
  esac
done

if [ "$INTERACTIVE" -eq 1 ]; then
  # Not a terminal? Do not spin on piped stdin.
  if [ ! -t 0 ]; then
    warn "stdin is not a terminal — interactive mode needs one."
    warn "falling back to --selftest. Use --text/--wav to pick a specific case."
    # Append to PASSTHRU, which is what exec actually reads.
    PASSTHRU=(--selftest)
  elif ! preflight; then
    warn "no microphone — dropping to --selftest."
    warn "try:  ./run.sh --wav /tmp/q.wav     (see README for a 'say' one-liner)"
    PASSTHRU=(--selftest)
  else
    say "Starting interactive loop. Enter = speak, q = quit, Ctrl-C = stop."
    say "If it seems to hang on first record, approve the macOS microphone prompt."
    printf '\n'
  fi
fi

exec "$BIN" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
