# voxlocal: per-stage processing logs

## Objective

Add clear processing logs for every pipeline stage. Every stage log must use this exact prefix shape:

```text
[Stage 1: Voice capture]: <message>
```

The implementation must preserve the current pipeline behavior, command-line modes, latency budget, and existing error handling.

## Stage names

Use these exact names everywhere:

1. `Voice capture`
2. `Speech to text`
3. `Text normalization`
4. `Retrieval augmented generation`
5. `LLM tool routing`
6. `Text to speech`
7. `Audio playback`

Keep capitalization and spacing consistent. Do not invent alternate names such as `STT`, `RAG`, or `TTS` in the stage prefix.

## Logging rules

1. Add one shared helper in `src/util.rs` for formatting stage logs.
2. The helper should accept a complete stage label and message, or accept a stage number/name and construct the label. Pick the simpler API that keeps call sites readable.
3. Use the exact output format:

   ```text
   [Stage N: Name]: message
   ```

4. Use `println!` for normal processing logs so they appear in normal command output and are captured by regression scripts.
5. Continue using `eprintln!` for errors and the existing latency budget output unless a specific stage error log is being added.
6. Do not log model prompt contents, full audio buffers, embeddings, or other large payloads.
7. Log useful summaries: counts, rates, selected documents, similarity scores, tool names, reply text, and elapsed time.
8. Keep the existing `[error]`, `[heard]`, `[clean]`, `[rag]`, `[llm]`, `[tool]`, `[reply]`, and latency output unless it is directly replaced by an equivalent stage log. Avoid breaking scripts that rely on current output.
9. Add logs at operation boundaries: one before an expensive operation and one after it when the result is meaningful.
10. Include elapsed time in completion logs where the existing `Budget` already measures that stage.
11. Do not change model loading, inference settings, routing rules, thresholds, audio formats, or reply text.

## Atomic implementation steps

### Step 1: Inspect the current logging and call boundaries

1. Open `src/util.rs` and locate `Budget`, `now`, and `ms`.
2. Open `src/capture.rs`, `src/stt.rs`, `src/normalize.rs`, `src/rag.rs`, `src/router.rs`, `src/tts.rs`, `src/playback.rs`, and `src/pipeline.rs`.
3. Confirm the main per-turn call sequence in `pipeline::run_turn`:

   ```text
   normalize -> embed/search -> route -> execute -> speak -> play
   ```

4. Confirm the microphone-specific sequence in `src/main.rs`:

   ```text
   capture -> transcribe -> run_turn
   ```

5. Confirm WAV input uses `pipeline::run_wav` and text input enters `pipeline::run_turn` directly.
6. Do not edit files during this inspection step.

### Step 2: Add the shared stage logging helper

1. Edit `src/util.rs`.
2. Add a small public helper for the standard stage format.
3. Keep the helper independent of pipeline state and model types.
4. Make the helper accept a message without requiring callers to allocate a formatted `String` unnecessarily, for example by accepting `impl AsRef<str>` or by using a simple `&str` API.
5. Do not modify `Budget` behavior in this step.

### Step 3: Add Stage 1 logs for voice capture

1. Edit `src/capture.rs`.
2. At the beginning of `record_until`, log that recording is starting and include the requested duration.
3. After obtaining the default input configuration, log the source sample rate, channel count, and sample format.
4. After capture completes, log the number of collected samples.
5. After downmixing and resampling, log the resulting mono sample count and target rate.
6. Before returning capture errors, add a Stage 1 error-context log only if it does not duplicate or obscure the existing error returned to the caller.
7. Keep the existing dot progress display and capture timing behavior.
8. Do not log individual samples.

Suggested output shape:

```text
[Stage 1: Voice capture]: starting 5.0 second recording
[Stage 1: Voice capture]: input format: 48000 Hz, 2 channels, F32
[Stage 1: Voice capture]: captured 240000 source samples
[Stage 1: Voice capture]: converted to 80000 mono samples at 16000 Hz
```

### Step 4: Add Stage 2 logs for speech to text

1. Edit `src/stt.rs`.
2. At the start of `Stt::transcribe`, log the input sample count and expected sample rate assumption.
3. After Whisper returns, log the transcript text and its character count.
4. If the transcript is empty, log that no speech text was produced.
5. Do not print Whisper internals, token IDs, or audio data.
6. Preserve the current returned `Result` values and error messages.

Suggested output shape:

```text
[Stage 2: Speech to text]: transcribing 80000 samples at 16000 Hz
[Stage 2: Speech to text]: produced 42 characters: "what is the price of an oil change"
```

### Step 5: Add Stage 3 logs for text normalization

1. Edit the normalization call site in `src/pipeline.rs`, or edit `src/normalize.rs` only if that gives a cleaner single call boundary. Prefer logging from `run_turn` so all modes receive the same log.
2. Before `Normalizer::apply`, log the raw transcript.
3. After `Normalizer::apply`, log the normalized transcript.
4. Include the existing measured normalization duration in the completion log.
5. Preserve the existing `[heard]` and `[clean]` lines unless replacing them with equivalent output would be demonstrably safe. The preferred approach is to retain them and add the stage logs.

Suggested output shape:

```text
[Stage 3: Text normalization]: input: "um what's the price of 1,200 dollars"
[Stage 3: Text normalization]: output: "what's the price of 1200 dollars" (0.8 ms)
```

### Step 6: Add Stage 4 logs for retrieval augmented generation

1. Edit `src/pipeline.rs` around the embedding and `rag.search` calls.
2. Before embedding, log that the normalized query is being embedded.
3. After search, log the number of documents considered or the corpus size if that value is available without changing public APIs.
4. When hits exist, log each selected document title and similarity score. The existing top-two limit must remain unchanged.
5. Log the context character count after the existing 320-character limit is applied.
6. When no hit passes `RAG_MIN_SIM`, log the threshold and that the fallback reply path is being used.
7. Include the existing measured RAG duration in the completion log.
8. Do not log full document bodies unless they are already printed by existing output.

Suggested output shape:

```text
[Stage 4: Retrieval augmented generation]: embedding query
[Stage 4: Retrieval augmented generation]: selected 1 document: oil change (0.812)
[Stage 4: Retrieval augmented generation]: using 96 context characters (10.7 ms)
```

No-hit shape:

```text
[Stage 4: Retrieval augmented generation]: no document exceeded similarity threshold 0.35; using fallback reply
```

### Step 7: Add Stage 5 logs for LLM routing and tool execution

1. Edit `src/pipeline.rs` around `llm.route` and `execute_tool`.
2. Add a log before routing with the query, context character count, and maximum generation token count.
3. Preserve the existing raw model output log. Add a Stage 5 summary after routing with the validated tool name and arguments.
4. If the router repaired truncated or malformed JSON, expose that fact in a concise Stage 5 log. If this requires changing `router::Llm::route` to return metadata, make the smallest API change possible and update all call sites.
5. If tool snapping changes an invalid model tool name, log the final selected tool. Do not expose internal prompt text.
6. Before `execute_tool`, log the tool being executed.
7. After execution, log the generated reply and the existing LLM/tool elapsed time.
8. Keep the allowed tool set, argument repair, service override, and time extraction behavior unchanged.

Suggested output shape:

```text
[Stage 5: LLM tool routing]: generating up to 48 tokens with 96 context characters
[Stage 5: LLM tool routing]: validated tool: check_price {service: "oil change"}
[Stage 5: LLM tool routing]: executing check_price
[Stage 5: LLM tool routing]: reply prepared: "oil change is $79." (72.4 ms)
```

### Step 8: Add Stage 6 logs for text to speech

1. Edit `src/pipeline.rs` around `tts.speak`, and edit `src/tts.rs` only if a lower-level log is needed.
2. Before synthesis, log the reply character count.
3. After synthesis, log the generated sample count and sample rate.
4. Include the existing measured TTS duration.
5. When `tts` is `None`, log that Stage 6 was skipped because TTS is disabled or unavailable.
6. Keep the existing optional TTS behavior and error propagation unchanged.

Suggested output shape:

```text
[Stage 6: Text to speech]: synthesizing 18 reply characters
[Stage 6: Text to speech]: generated 35200 samples at 22050 Hz (75.2 ms)
```

Skipped shape:

```text
[Stage 6: Text to speech]: skipped; TTS disabled
```

### Step 9: Add Stage 7 logs for audio playback

1. Edit `src/playback.rs` or its call site in `src/pipeline.rs`. Prefer the call site for the stage boundary and keep `play` focused on playback.
2. Before playback, log the input sample count and sample rate.
3. If resampling is needed, log the destination device rate.
4. After `sleep_until_end`, log playback completion and the existing measured playback duration.
5. Preserve device channel handling, resampling, blocking playback, and error behavior.

Suggested output shape:

```text
[Stage 7: Audio playback]: playing 35200 samples at 22050 Hz
[Stage 7: Audio playback]: resampling for output device at 48000 Hz
[Stage 7: Audio playback]: playback complete (1133.8 ms)
```

### Step 10: Add mode-specific input logs

1. Edit `src/pipeline.rs` and `src/main.rs` only where needed.
2. For `--text`, log that Stage 1 was skipped because text input was supplied.
3. For `--wav`, log that the WAV file replaced microphone capture and include the decoded sample count.
4. For `--selftest`, log the case number before each call to `run_turn`; the individual stages must still log normally.
5. For `--probe-audio`, use Stage 1 for capture and Stage 2 for transcription. Keep the existing peak and RMS diagnostics.
6. For `--no-tts`, emit the Stage 6 skipped log and do not invoke playback.
7. Do not add logs to `run.sh` unless an execution path cannot otherwise identify the selected mode. The requested stage logs belong to the Rust program.

### Step 11: Review output consistency

1. Search all new stage logs with:

   ```bash
   rg -n "Stage [1-7]:" src
   ```

2. Confirm every stage prefix exactly matches the names in this plan.
3. Confirm no stage log uses a different bracket style, lowercase stage name, or missing colon.
4. Confirm normal logs use `stdout` and errors remain on `stderr`.
5. Confirm no log prints model prompts, embeddings, raw audio, or large document bodies.
6. Confirm the existing latency budget still prints once per completed turn.

### Step 12: Verify the implementation

Run the existing project checks after implementation:

```bash
cargo build --release
cargo clippy --release --all-targets
./run.sh --selftest
./run.sh --text "what is the price of brake pads"
./run.sh --text "hi" --no-tts
./run.sh --wav-regression
```

For each command, inspect that:

1. The expected Stage 1 through Stage 7 logs appear when that stage runs.
2. Text mode identifies Stage 1 as skipped.
3. `--no-tts` identifies Stages 6 and 7 as skipped or bypassed according to the final implementation.
4. The no-match RAG path logs the threshold and fallback reply.
5. Selftest still completes all four cases.
6. WAV regression still passes.
7. Existing reply text and routing behavior are unchanged.
8. The latency budget remains present and contains the same stage measurements.

Do not add new tests unless the implementation exposes a small pure helper that requires a focused unit test to make the logging behavior deterministic. Prefer the existing commands above for verification.

## Completion criteria

The work is complete when:

- Every processing stage emits logs with the exact requested format.
- Logs appear for interactive, text, WAV, selftest, probe, regression, and no-TTS paths where applicable.
- Logs explain the input, main processing action, and result for each stage.
- Existing output, replies, routing, timing, and error behavior remain compatible.
- Build, clippy, selftest, text mode, and WAV regression all succeed.
