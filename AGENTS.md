# Repository Guidelines

## Project Structure & Module Organization

This repository is a Rust 2021 CLI for a local voice-agent pipeline. Application code is in `src/`, with one module per pipeline concern: audio capture and playback, Whisper speech-to-text, normalization, in-memory RAG, routing, TTS, shared utilities, and orchestration in `pipeline.rs`. `src/main.rs` wires the stages and command-line modes together. `samples/` contains WAV fixtures for end-to-end regression checks. `.models/` holds downloaded local models and should not be committed. `run.sh` is the normal launcher; `setup.sh` downloads models and builds the release binary. `.cargo/config.toml` intentionally places Cargo output in `/tmp/vx`.

## Build, Test, and Development Commands

- `./setup.sh` — verify `curl`, Rust, and CMake; download models into `.models/`; build release artifacts.
- `cargo fmt --all` — format Rust sources.
- `cargo check` — type-check quickly without producing a release binary.
- `./run.sh --selftest` — exercise all pipeline stages with built-in utterances.
- `./run.sh --wav-regression` — run every `samples/*.wav` fixture through the full pipeline.
- `./run.sh --text "book a brake pad tomorrow"` — run a one-shot text request without a microphone.
- `./run.sh --check` — perform model and audio-device preflight only.

Use `VOXLOCAL_MODELS=/path/to/models` to select a model directory. Keep the configured short target path; overriding it can break the vendored espeak-ng build on macOS.

## Coding Style & Naming Conventions

Use standard `rustfmt` formatting, four-space indentation, and idiomatic Rust naming: `snake_case` for functions/modules, `CamelCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Prefer `anyhow::Result` with contextual errors at I/O and model boundaries. Keep pipeline stages small and preserve the existing stage-timing/logging conventions.

## Testing Guidelines

There are currently no standalone Rust test modules. Treat `cargo check`, `--selftest`, and `--wav-regression` as the validation suite. Add focused unit tests beside the relevant module for pure parsing, normalization, or WAV-processing logic; use descriptive names such as `rejects_non_riff_wav`.

## Commit & Pull Request Guidelines

No Git history is available in this checkout to infer established conventions. Use concise imperative commits (for example, `Improve WAV validation`) and keep unrelated changes separate. Pull requests should explain behavior changes, list validation commands and results, note model or hardware assumptions, and include representative CLI output when changing user-visible pipeline behavior.

## Security & Configuration Tips

Keep inference and model files local; do not add credentials, downloaded model binaries, generated build output, or personal audio recordings to commits. Review changes to model URLs and shell scripts carefully because setup and runtime execute external tools and filesystem operations.
