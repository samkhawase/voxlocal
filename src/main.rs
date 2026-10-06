//! voxlocal — a 100% local, low-latency voice agent CLI.
//!
//! Pipeline (all inference on-device; nothing leaves the machine):
//!
//!   mic (cpal) -> 16 kHz mono f32
//!     -> Whisper tiny.en (whisper-rs / whisper.cpp)      : speech-to-text
//!     -> regex normalisation                            : strip fillers, expand numbers/dates
//!     -> all-MiniLM-L6-v2 + exact cosine search (candle): in-memory RAG
//!     -> SmolLM2-135M-Instruct Q4_K_M GGUF (candle)      : ChatML -> strict JSON tool call
//!     -> mock tool executor                              : parse + dispatch
//!     -> Piper ONNX (piper-rs)                           : text-to-speech
//!     -> rodio -> speakers                               : playback
//!
//! Every stage reports its own wall-clock time so the latency budget is measurable
//! rather than aspirational. See [`util::Budget`]. This file is only the driver:
//! each stage lives in its own module (`capture`, `stt`, `normalize`, `rag`,
//! `router`, `tts`, `playback`), and `pipeline` wires them into turns and modes.
//!
//! Usage:
//!   voxlocal                    # interactive REPL: press Enter, speak, get a spoken reply
//!   voxlocal --text "..."       # skip the mic, run the text pipeline end-to-end
//!   voxlocal --selftest         # run every stage against the bundled models

mod capture;
mod normalize;
mod pipeline;
mod playback;
mod rag;
mod router;
mod stt;
mod tts;
mod util;

use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use cpal::traits::{DeviceTrait, HostTrait};

use capture::{record_until, RECORD_SECS};
use normalize::Normalizer;
use pipeline::run_turn;
use rag::{autoshop_corpus, Embedder, Rag};
use router::Llm;
use stt::Stt;
use tts::Tts;
use util::{is_tty, ms, now, stage_log, Budget, Paths};

/// Probe Metal without allowing an unavailable/broken backend to terminate the
/// process or print its internal panic diagnostic. The bool reports whether the
/// fallback was caused by a panic rather than a normal initialization error.
fn probe_metal() -> (Option<Device>, bool) {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| Device::new_metal(0)));
    std::panic::set_hook(previous_hook);

    match result {
        Ok(Ok(device)) => (Some(device), false),
        Ok(Err(_)) => (None, false),
        Err(_) => (None, true),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let opt = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let paths = Paths::discover()?;
    println!(
        "voxlocal — local voice agent (models: {})",
        paths.llm.parent().unwrap().display()
    );

    // Candle: prefer Metal on Apple Silicon, fall back to CPU.
    // candle-core 0.11 exposes no `has_metal()`, so probe `new_metal` directly
    // and treat any failure (no GPU, headless) as "use CPU".
    let (metal, recovered_from_panic) = probe_metal();
    let device = match metal {
        Some(d) => {
            println!("  device:  metal");
            d
        }
        None => {
            if recovered_from_panic {
                println!("  device:  Metal initialization panicked; system recovered and is continuing on CPU");
            }
            println!("  device:  cpu");
            Device::Cpu
        }
    };

    // --- load models (warm-up, not part of the per-turn budget) ---
    println!("\n  Initializing local voice pipeline...");
    whisper_rs::install_logging_hooks();

    let t = now();
    let stt = Stt::load(&paths.stt).context("load whisper")?;
    println!("  ✓ Whisper STT loaded       {:>5.0} ms", ms(t.elapsed()));

    let t = now();
    let embedder = Embedder::load(&paths.embed_dir, &device)?;
    println!("  ✓ embeddings ready         {:>5.0} ms", ms(t.elapsed()));

    let corpus = autoshop_corpus();
    let t = now();
    let rag = Rag::build(&embedder, &corpus)?;
    println!(
        "  ✓ retrieval ready          {:>5.0} ms · {} documents",
        ms(t.elapsed()),
        corpus.len()
    );

    let t = now();
    let mut llm = Llm::load(&paths.llm, &paths.llm_tok, &device)?;
    println!("  ✓ LLM router ready        {:>5.0} ms", ms(t.elapsed()));

    let t = now();
    let norm = Normalizer::new()?;
    println!("  ✓ text normalizer ready   {:>5.0} ms", ms(t.elapsed()));

    // Piper pulls in ONNX Runtime + espeak-ng; heavy, and optional for text mode.
    let mut tts = if flag("--no-tts") {
        println!("  · TTS disabled by flag");
        None
    } else {
        let started = now();
        match Tts::load(&paths.tts_onnx, &paths.tts_cfg) {
            Ok(t) => {
                println!(
                    "  ✓ TTS ready               {:>5.0} ms",
                    ms(started.elapsed())
                );
                Some(t)
            }
            Err(e) => {
                eprintln!("  tts:     disabled ({e:#})");
                None
            }
        }
    };

    let t = now();
    pipeline::warmup(&stt, &embedder, &mut llm);
    println!("  ✓ warmup complete         {:>5.0} ms", ms(t.elapsed()));
    println!(
        "  Pipeline ready · {} · {} documents",
        if matches!(&device, Device::Cpu) {
            "CPU"
        } else {
            "Metal"
        },
        corpus.len()
    );

    // --- one-shot text mode ---
    if let Some(text) = opt("--text") {
        let mut b = Budget::default();
        run_turn(&norm, &rag, &embedder, &mut llm, tts.as_mut(), text, &mut b)?;
        return Ok(());
    }

    // --- WAV mode: STT -> normalise -> RAG -> LLM -> TTS, with no mic needed ---
    if let Some(wav) = opt("--wav") {
        pipeline::run_wav(&wav, &stt, &norm, &rag, &embedder, &mut llm, tts.as_mut())?;
        return Ok(());
    }

    // --- audio-only probe: proves cpal capture -> whisper without needing a human ---
    if flag("--probe-audio") {
        pipeline::run_probe_audio(&stt)?;
        return Ok(());
    }

    // --- selftest: exercise every stage against the bundled models ---
    if flag("--selftest") {
        pipeline::run_selftest(&norm, &rag, &embedder, &mut llm)?;
        return Ok(());
    }

    // --- interactive mode ---
    // Interactive needs a real terminal: the loop prompts per turn, so piped stdin
    // would either spin or look frozen. --wav/--text/--selftest cover automation.
    if !cpal::default_host().default_input_device().is_some() {
        bail!("no input device found — use --text, --wav or --selftest instead");
    }
    if !is_tty() {
        bail!("interactive mode needs a terminal; use --text, --wav or --selftest for piped input");
    }
    println!("\n  Press Enter, then speak while the dots tick. Type q + Enter to quit.\n");

    // Surface the input device now rather than after the user has spoken.
    match cpal::default_host().default_input_device() {
        Some(d) => match d.description() {
            Ok(desc) => println!("  mic:     {}", desc.name()),
            Err(_) => println!("  mic:     (present, name unavailable)"),
        },
        None => eprintln!("  mic:     NONE FOUND — audio capture will fail"),
    }

    loop {
        print!("  [enter] ");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        if line.trim().eq_ignore_ascii_case("q") {
            break;
        }

        eprintln!("  listening {RECORD_SECS}s (the dots are seconds — speak now)");
        let t = now();
        let cap = match record_until(RECORD_SECS) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  capture failed: {e:#}");
                continue;
            }
        };
        let cap_ms = ms(t.elapsed());
        stage_log(
            1,
            "Voice capture",
            format!(
                "capture complete: {} samples ({cap_ms:.0} ms)",
                cap.samples.len()
            ),
        );
        println!(
            "  captured {} samples ({:.0} ms)",
            cap.samples.len(),
            cap_ms
        );

        let mut b = Budget::default();
        b.add("capture", Duration::from_secs_f32(cap_ms / 1000.0));

        // Timed here rather than inside run_turn so the budget reflects a real
        // end-to-end turn. Guarded like run_turn: a Whisper failure should cost
        // one turn, not the session.
        let t = now();
        stage_log(
            2,
            "Speech to text",
            format!(
                "transcribing {} samples at {} Hz",
                cap.samples.len(),
                capture::TARGET_SR
            ),
        );
        let raw = match stt.transcribe(&cap.samples) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  [stt]    failed: {e:#}");
                continue;
            }
        };
        b.add("stt", t.elapsed());

        if raw.trim().is_empty() {
            println!("  [stt]    (nothing intelligible — try again, and speak during the window)");
            continue;
        }
        // Never let one bad utterance end the session: a 135M router will
        // occasionally emit unparseable output, and that should cost you one turn,
        // not the whole REPL.
        if let Err(e) = run_turn(&norm, &rag, &embedder, &mut llm, tts.as_mut(), raw, &mut b) {
            eprintln!("  [error]  {e:#}");
        }
    }

    Ok(())
}
