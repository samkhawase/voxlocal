# voxlocal

A 100% local voice agent CLI. Microphone → Whisper → text normalisation → in-memory RAG
→ SmolLM2-135M tool call → Piper TTS → speakers. No network calls at inference time,
no API keys, no cloud services.

```
mic (cpal) ──▶ Whisper tiny.en ──▶ regex normalise ──▶ MiniLM + cosine ──▶ SmolLM2-135M
                                                                              │
                              speakers ◀── rodio ◀── Piper ONNX ◀── mock executor ◀┘
```

## Quick start

```bash
./setup.sh                    # fetch models into .models/, then cargo build --release
./run.sh                      # interactive voice loop (does the above checks for you)
```

`run.sh` is the intended entry point. It finds the binary (which is **not** in
`./target`, see "Build requirements"), preflights the models and audio devices,
traps Ctrl-C so you don't get abort noise mid-capture, and falls back to
`--selftest` if there's no usable microphone.

```bash
./run.sh --check                              # preflight only, start nothing
./run.sh --text "book a brake pad tomorrow"   # one-shot text turn
./run.sh --wav /tmp/q.wav                     # full loop from a file
./run.sh --wav-regression                     # run bundled sample(s) through the loop
./run.sh --selftest                           # pipeline check
./run.sh --no-tts                             # interactive, no Piper/speakers
```

Without `run.sh`, set `BIN` yourself:

```bash
BIN=$(cargo metadata --format-version 1 --no-deps | tr ',' '\n' \
      | grep '"target_directory"' | cut -d'"' -f4)/release/voxlocal
$BIN --selftest
```

Model root is `.models/` by default; override with `VOXLOCAL_MODELS=/some/path`.

## Modes

| Flag | What it does |
|---|---|
| *(none)* | Interactive REPL. Enter starts a 5 s recording; `q` quits. Requires a TTY. |
| `--text "<utterance>"` | Text pipeline only (skips Whisper and the mic). |
| `--wav <path>` | Full loop from a WAV file: STT → RAG → LLM → TTS → playback. |
| `--selftest` | Runs four utterances through every stage and reports pass/fail. |
| `--wav-regression` | Runs `samples/` through the full pipeline; fails if any sample misroutes. |
| `--probe-audio` | Records from the mic and prints peak/RMS so you can check levels. |
| `--no-tts` | Skip Piper synthesis and playback. |

Interactive mode refuses to start on piped stdin (it prompts per turn, so it would
look frozen) and prints how to pick a non-interactive mode instead.

While recording, stderr prints one dot per ~half second so you can see the window
open and — more usefully — close.

## Testing it

Work from cheapest to most demanding — each level isolates one more stage.

**1. Pipeline logic, no audio hardware at all**
```bash
./run.sh --selftest
```
Four utterances through normalise → RAG → LLM → executor. Expect
`all 4 cases passed`. Fast (~2 s).

**2. Whisper, deterministic**
```bash
say -v Samantha -o /tmp/q.wav --data-format=LEI16@16000 "what is the price of an oil change"
./run.sh --wav /tmp/q.wav
```
Exercises STT → RAG → LLM → TTS → speakers with reproducible audio. `say` must produce
16 kHz mono — `afinfo /tmp/q.wav` should report `1 ch, 16000 Hz`. Expect
`[stt] "What is the price of an oil change?"` and `[reply] oil change is $79.`

**3. Mic levels, before committing to the interactive loop**
```bash
./run.sh --probe-audio
```
Reports `peak` and `rms`. A quiet room gives `rms≈0.0007`; speech should push `peak`
to ~0.07. If `rms` stays near zero your mic is muted or the wrong input is selected.

**4. Interactive**
```bash
./run.sh
```
Press Enter, then speak while the dots tick. Click **Allow** if macOS prompts for
microphone access. Type `q` + Enter to quit, or Ctrl-C.

**Useful variations**
```bash
./run.sh --no-tts --text "..."     # is it the LLM or the speakers? skip Piper
./run.sh --text "hi"               # out-of-domain: expect "no context used"
./run.sh --text "what do you sell" # same — RAG floor rejects it
./run.sh --wav bad.wav             # error handling on a non-RIFF file
./run.sh --check                   # preflight only
```

`--text "hi"` and `--text "what do you sell"` are the two most useful negative tests:
they should both print `(nothing above 0.35 similarity — no context used)` and the
"did not find anything" reply, rather than routing on irrelevant context.

Every run prints a per-stage latency budget — use it to spot which stage regressed.

## Regression testing

`run.sh --wav-regression` runs every `samples/*.wav` through the full pipeline
(mic → Whisper → RAG → LLM → Piper) and fails if any sample misroutes (e.g. the
LLM fails to name the car part it recognised). It needs no microphone, so it's
suitable for CI.

The bundled sample is generated with macOS `say` (12 kHz mono Int16 WAV):

```bash
say -v Samantha -o samples/oil_change.wav "Book me an appointment for new brake pads"
```

Add more by dropping another `.wav` into `samples/`.

## Measured latency

Apple M5 Pro, release build, Metal for both Whisper and candle. Live microphone input,
`MacBook Pro Microphone` (48 kHz stereo, downmixed and resampled to 16 kHz mono):

```
  latency budget:
    capture     5099.2 ms   <- fixed 5 s recording window, not compute
    stt           60.9 ms
    normalize      4.0 ms
    rag           10.8 ms
    llm           70.5 ms
    tts           75.2 ms
    play        1133.8 ms   <- audio duration, not latency
```

| Stage | Typical |
|---|---|
| Whisper STT (1.8 s utterance) | 35–61 ms |
| Normalise (regex) | <4 ms |
| RAG (embed query + cosine over 10 docs) | 5–11 ms |
| SmolLM2 prefill + ≤48 greedy tokens | 65–95 ms |
| Piper synthesis | 70–75 ms |
| **Compute total (excl. capture + playback)** | **~175–235 ms** |

So the **compute** path lands around 180–230 ms rather than the 150 ms target. The
dominant cost is LLM decode, and closing the gap needs a smaller/quantised model or
speculative decoding rather than more tuning. Playback is wall-clock audio duration and
is not latency. `capture` is the deliberate 5 s recording window, also not compute — the
figure that matters for responsiveness is the time from *end of speech* to first audio.

Every run prints this breakdown, so you can measure on your own hardware.

## Models

All four are public on HuggingFace; `setup.sh` pulls them unauthenticated.

| Role | Artifact | Size |
|---|---|---|
| LLM | `SmolLM2-135M-Instruct-Q4_K_M.gguf` (bartowski GGUF) + base tokenizer | 114 MB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 97 MB |
| STT | `ggml-tiny.en.bin` (whisper.cpp) | 81 MB |
| TTS | Piper `en_US-lessac-medium` ONNX | 61 MB |

## Build requirements

* Rust 1.98+
* **CMake** — `whisper-rs` and `espeak-rs-sys` each vendor C/C++ (whisper.cpp and
  espeak-ng) and build it via CMake. Without it the build fails immediately.

### Keep the project path short

`.cargo/config.toml` pins `target-dir = "/tmp/vx"`. This is not cosmetic. espeak-ng
hardcodes `N_PATH_HOME` at 160 bytes and truncates the phondata path at
`sizeof(path_home)+20`; with a long build directory the filename
`phsource/vwl_en_us_nyc/a_raised` truncates to `a_rais` and the build dies with:

```
Failed to open: '.../phsource/vwl_en_us_nyc/a_rais'
make[2]: *** [espeak-ng-data/phondata] Error 1
```

This is not arch-specific (it was first reported on Linux) and upstream fixed it only
for Linux — espeak-ng PR #2220 guards the bump behind `#if defined(__linux__)`, so macOS
still needs a short path. See espeak-ng#2048 and piper-rs#20. Budget: the full path to
`debug/build/espeak-rs-sys-<hash>/out/build/espeak-ng-data` should stay under ~130 chars.

**Consequence:** the binary lands in `/tmp/vx/release/`, not `./target/release/`.
Anything that assumes the default layout (`cargo run --release`, CI path filters,
IDE run configs) will not find it. Resolve the real path with `cargo metadata` as shown
in Quick start, or symlink it back:

```bash
mkdir -p target/release && ln -sf /tmp/vx/release/voxlocal target/release/voxlocal
```

Also note `/tmp` is cleared on reboot, so expect a rebuild after one.

## Version pins and why

* **`cpal = "0.17"`, not 0.18.** `rodio 0.22` depends on `cpal ^0.17`. Pinning 0.18 would
  put two cpal versions in the graph and two CoreAudio backends in the process.
* **`candle 0.11`.** `BertModel::forward` is `(input_ids, token_type_ids, attention_mask)`
  and `Tensor::new` takes shape from the array, so both differ from older candle examples.
  There is no `Config::default()` shortcut for MiniLM — it deserialises to BERT-base
  (768/12); the real `config.json` must be read.
* **`tokenizers = "0.22"`.** candle 0.11 has no `tokenizers` dependency at all, so this
  is our choice rather than something to reconcile.
* **`whisper-rs` with `metal`.** Metal is opt-in; without the feature whisper.cpp sets
  `GGML_METAL=OFF`.
* **Do not call `FullParams::set_n_threads(0)`.** whisper.cpp reads that as "no worker
  threads" and aborts the process with `Rust cannot catch foreign exceptions`. Leaving
  it unset gives the library default of `min(4, hardware_concurrency)`.

## Known limitations

**Microphone permission on macOS.** Verified working: the interactive loop was
exercised end-to-end on a MacBook Pro Microphone (capture → STT → RAG → LLM → TTS →
playback). Note that the *first* attempt appears to hang. That is macOS's TCC gate: it
shows a GUI dialog, and a plain CLI binary has no bundle identifier to attach it to, so
the prompt sits unanswered until you click **Allow** (System Settings → Privacy &
Security → Microphone). Audio *output* needs no grant and works immediately.

**The 135M router needs supervision.** SmolLM2-135M is small enough that it does not
reliably respect a closed tool vocabulary or copy arguments correctly. Observed
failures, all now corrected in `snap_tool` / `route` / `time_from_query`:

* copies the `service` argument from the few-shot example regardless of the query
  → `service` is taken from the retrieved RAG document instead
* invents tools that were never offered (e.g. `schedule_oil_change`)
  → snapped to the allowed set, falling back to keyword rules
* echoes `time` from the example on unrelated queries
  → re-derived from the query's own words

This is a deliberate correctness-over-purity trade. The JSON repair pass in
`parse_tool_call` / `repair_truncated` also exists because a 48-token cap can truncate
output mid-string. If you want the model's output taken literally, delete those three
functions — the pipeline still runs, and routing accuracy degrades noticeably.

**A failed turn never ends the session.** The 135M router will occasionally emit
unparseable output. In interactive mode that costs you one turn and prints `[error]`,
rather than propagating out of `main` and killing the REPL.

**`piper-rs` pins `ort = "=2.0.0-rc.12"`, a release candidate.** It builds and runs, but
you are on an RC ONNX Runtime. `sherpa-onnx 1.13.8` (prebuilt `osx-arm64`, returns raw
`&[f32]` PCM) is the cleaner long-term TTS choice if you want to drop espeak-ng and the
RC pin.

**RAG is brute force with a relevance floor.** Exact cosine over an in-memory corpus,
capped at 2 hits and filtered by `RAG_MIN_SIM = 0.35`. Without that floor an unrelated
utterance still splices near-zero-scoring documents into the prompt and misleads the
router — `"hi"` scores 0.14 against every doc. On this corpus genuine questions score
>=0.55, so 0.35 separates cleanly. It does not scale past a few thousand documents
without an index.

## Layout

```
Cargo.toml          pinned deps
.cargo/config.toml  short target-dir (required by espeak-ng, see above)
setup.sh            model fetch + build
run.sh              interactive launcher: preflight, binary discovery, clean Ctrl-C
src/main.rs         the driver: load models, pick a mode, run the REPL loop
src/capture.rs      Stage 1 — mic capture (cpal); also owns TARGET_SR + resample()
src/stt.rs          Stage 2 — Whisper STT
src/normalize.rs    Stage 3 — regex text normalisation
src/rag.rs          Stage 4 — MiniLM embeddings, RAG_MIN_SIM = 0.35, corpus, price lookup
src/router.rs       Stage 5 — SmolLM2 tool router + mock tool executor
src/tts.rs          Stage 6 — Piper TTS
src/playback.rs     Stage 7 — audio playback (rodio)
src/pipeline.rs     WAV decode, run_turn, --wav / --selftest / --probe-audio / warmup
src/util.rs         Budget / Paths / ms / now / is_tty
.models/            downloaded weights
```
