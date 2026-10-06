#!/usr/bin/env bash
#
# voxlocal — fetch every model into .models/ and build the binary.
#
# Everything downloaded here is public and runs fully offline afterwards.
# No API tokens are required: all four artifacts are ungated on HuggingFace.
#
# IMPORTANT — keep the project path short (see .cargo/config.toml). piper-rs
# vendors espeak-ng, which has a hardcoded 160-byte N_PATH_HOME buffer; a long
# build directory truncates its phondata filenames and the build fails with
#   Failed to open: '.../phsource/vwl_en_us_nyc/a_rais'
# Upstream only fixed this for Linux (espeak-ng PR #2220), so macOS still needs
# the short CARGO_TARGET_DIR that .cargo/config.toml sets.

set -euo pipefail

MODEL_DIR="${VOXLOCAL_MODELS:-.models}"
HF="https://huggingface.co"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: '$1' not found. $2" >&2
    exit 1
  }
}

say "Checking prerequisites"
need curl "Install with: brew install curl"
need cargo "Install Rust from https://rustup.rs"

# whisper-rs and espeak-rs-sys both shell out to CMake to build their vendored
# C/C++ (whisper.cpp, espeak-ng). Without this the build dies immediately.
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required to build whisper.cpp and espeak-ng."
  echo "  macOS:   brew install cmake"
  echo "  Debian:  sudo apt-get install cmake"
  exit 1
fi
echo "  cargo $(cargo --version | awk '{print $2}')"
echo "  cmake   $(cmake --version | head -1 | awk '{print $3}')"

mkdir -p "$MODEL_DIR"/{llm,embed,stt,tts}

# fetch <url> <dest>
fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    echo "  have $(basename "$dest")"
    return 0
  fi
  echo "  get  $(basename "$dest")"
  # -f: fail on HTTP errors (a 401 HTML page would otherwise save as a "model").
  curl -sSfL --retry 3 --retry-delay 2 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

say "1/4  SmolLM2-135M-Instruct (Q4_K_M GGUF) — the router LLM"
fetch "$HF/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf" \
      "$MODEL_DIR/llm/smollm2-135m-q4km.gguf"
# Tokenizer comes from the *base* repo, not the GGUF one.
fetch "$HF/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json" \
      "$MODEL_DIR/llm/tokenizer.json"
fetch "$HF/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer_config.json" \
      "$MODEL_DIR/llm/tokenizer_config.json"

say "2/4  all-MiniLM-L6-v2 — the RAG embedding model"
for f in config.json model.safetensors tokenizer.json tokenizer_config.json; do
  fetch "$HF/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f" "$MODEL_DIR/embed/$f"
done

say "3/4  whisper-tiny.en — speech to text"
fetch "$HF/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" "$MODEL_DIR/stt/ggml-tiny.en.bin"

say "4/4  Piper en_US-lessac-medium — text to speech"
PIPER="$HF/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
fetch "$PIPER/en_US-lessac-medium.onnx"      "$MODEL_DIR/tts/en_US-lessac-medium.onnx"
fetch "$PIPER/en_US-lessac-medium.onnx.json" "$MODEL_DIR/tts/en_US-lessac-medium.onnx.json"

say "Model inventory"
du -sh "$MODEL_DIR"/*/ | sed 's/^/  /'
# Sanity-check the ones we can. A small config.json is legitimate; a small
# *safetensors* or *.bin is a truncated download and must be refetched.
for f in "$MODEL_DIR/embed/model.safetensors" \
         "$MODEL_DIR/llm/smollm2-135m-q4km.gguf" \
         "$MODEL_DIR/stt/ggml-tiny.en.bin" \
         "$MODEL_DIR/tts/en_US-lessac-medium.onnx"; do
  sz=$(wc -c <"$f")
  [ "$sz" -lt 1000000 ] && echo "  WARNING: $f is only ${sz} bytes — refetching" && rm -f "$f"
done

say "Building (release)"
# The target dir is pinned short by .cargo/config.toml; do not override it.
cargo build --release
# Ask cargo where the binary actually landed — it is NOT ./target because
# target-dir is redirected to keep espeak-ng's build path short.
BIN="$(cargo metadata --format-version 1 --no-deps \
        | tr ',' '\n' | grep '"target_directory"' | cut -d'"' -f4)/release/voxlocal"
echo "  built: $BIN"

say "Done"
cat <<EOF
Try it:

  # 1. pipeline self-check (no audio hardware needed)
  $BIN --selftest

  # 2. full loop from a WAV file — no microphone needed
  say -v Samantha -o /tmp/q.wav --data-format=LEI16@16000 "what is the price of an oil change"
  $BIN --wav /tmp/q.wav

  # 3. text only, no STT and no playback
  $BIN --text "book a brake pad appointment tomorrow at 9am"

  # 4. interactive: press Enter, then speak
  $BIN

  # (add --no-tts to any of the above to skip Piper and the speakers)

The first interactive run can look like a hang: macOS shows a GUI microphone
permission dialog and a CLI binary cannot answer it. Click Allow (System Settings >
Privacy & Security > Microphone). Audio output needs no permission.

Note the binary is under /tmp/vx, not ./target — see .cargo/config.toml.
EOF
