// ---------------------------------------------------------------------------
// Pipeline glue — WAV decode, run_turn, mock tools, non-interactive modes
// ---------------------------------------------------------------------------

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::capture::{record_until, resample, RECORD_SECS, TARGET_SR};
use crate::normalize::Normalizer;
use crate::playback::play;
use crate::rag::{Embedder, Rag, RAG_MIN_SIM};
use crate::router::{execute_tool, Llm, MAX_NEW_TOKENS};
use crate::stt::Stt;
use crate::tts::Tts;
use crate::util::{ms, now, stage_log, Budget};

// ---------------------------------------------------------------------------
// WAV decode
// ---------------------------------------------------------------------------

/// Read a PCM WAV file into mono f32 at [`crate::capture::TARGET_SR`].
///
/// Handles the common cases only: integer PCM 8/16/32-bit and IEEE float 32.
/// Anything else is an error rather than a silent misread.
pub fn read_wav_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("{} is not a RIFF/WAVE file", path.display());
    }

    let mut pos = 12usize;
    let (mut fmt, mut data) = (None, None);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = pos + 8;
        let end = (body + size).min(bytes.len());

        match id {
            b"fmt " => {
                let tag = u16::from_le_bytes(bytes[body..body + 2].try_into()?);
                let ch = u16::from_le_bytes(bytes[body + 2..body + 4].try_into()?) as usize;
                let rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into()?);
                let bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into()?);
                fmt = Some((tag, ch, rate, bits));
            }
            b"data" => data = Some(&bytes[body..end]),
            _ => {}
        }
        // Chunks are word-aligned.
        pos = end + (size & 1);
    }

    let (tag, ch, rate, bits) = fmt.context("WAV has no fmt chunk")?;
    let data = data.context("WAV has no data chunk")?;
    if ch == 0 || rate == 0 {
        bail!("WAV reports {ch} channels at {rate} Hz");
    }

    // Decode to interleaved f32, then downmix and resample.
    let interleaved: Vec<f32> = match (tag, bits) {
        (1, 16) => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0)
            .collect(),
        (1, 32) => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| (i32::from_le_bytes(*c) as f64 / 2_147_483_648.0) as f32)
            .collect(),
        (3, 32) => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        (1, 8) => data
            .iter()
            .map(|&b| (f32::from(b) - 128.0) / 128.0)
            .collect(),
        _ => bail!("unsupported WAV: format tag {tag}, {bits}-bit"),
    };

    let mono: Vec<f32> = interleaved
        .chunks(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect();
    Ok(resample(&mono, rate, TARGET_SR))
}

/// Runs the text half of the pipeline: normalise -> RAG -> route -> tool -> speak.
pub fn run_turn(
    norm: &Normalizer,
    rag: &Rag,
    embedder: &Embedder,
    llm: &mut Llm,
    tts: Option<&mut Tts>,
    transcript_raw: String,
    budget: &mut Budget,
) -> Result<()> {
    stage_log(1, "Voice capture", "skipped; text input was supplied");
    println!("\n  [heard]  {transcript_raw}");

    let t = now();
    stage_log(
        3,
        "Text normalization",
        format!("input: {transcript_raw:?}"),
    );
    let transcript = norm.apply(&transcript_raw);
    budget.add("normalize", t.elapsed());
    println!("  [clean]  {transcript}");
    stage_log(
        3,
        "Text normalization",
        format!("output: {transcript:?} ({:.1} ms)", ms(t.elapsed())),
    );

    // --- RAG ---
    let t = now();
    stage_log(4, "Retrieval augmented generation", "embedding query");
    let qv = embedder.embed_one(&transcript).context("embed query")?;
    let hits = rag.search(&qv, 2);
    budget.add("rag", t.elapsed());

    if hits.is_empty() {
        // Nothing in the knowledge base is close enough to answer this. Saying so
        // is better than routing on irrelevant context.
        let reply =
            "I did not find anything in our service list that matches that. Could you rephrase?";
        println!("  [rag]    (nothing above {RAG_MIN_SIM:.2} similarity — no context used)");
        stage_log(4, "Retrieval augmented generation", format!("no document exceeded similarity threshold {RAG_MIN_SIM:.2}; using fallback reply ({:.1} ms)", ms(t.elapsed())));
        println!("  [reply]  {reply}");
        if let Some(tts) = tts {
            stage_log(
                6,
                "Text to speech",
                format!("synthesizing {} reply characters", reply.chars().count()),
            );
            let t = now();
            let (pcm, sr) = tts.speak(reply)?;
            budget.add("tts", t.elapsed());
            stage_log(
                6,
                "Text to speech",
                format!(
                    "generated {} samples at {sr} Hz ({:.1} ms)",
                    pcm.len(),
                    ms(t.elapsed())
                ),
            );
            let t = now();
            play(&pcm, sr)?;
            budget.add("play", t.elapsed());
        } else {
            stage_log(6, "Text to speech", "skipped; TTS disabled");
        }
        budget.print();
        return Ok(());
    }
    let ctx: Vec<String> = hits
        .iter()
        .map(|(d, _)| format!("{}: {}", d.title, d.body))
        .collect();
    println!(
        "  [rag]    {}",
        hits.iter()
            .map(|(d, s)| format!("{} ({:.3})", d.title, s))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    for (doc, score) in &hits {
        stage_log(
            4,
            "Retrieval augmented generation",
            format!("selected document: {} ({score:.3})", doc.title),
        );
    }
    let context = ctx.join(" ");
    // Keep the context tight; a 135M router degrades fast with long prompts.
    let context: String = context.chars().take(320).collect();
    stage_log(
        4,
        "Retrieval augmented generation",
        format!(
            "using {} context characters ({:.1} ms)",
            context.chars().count(),
            ms(t.elapsed())
        ),
    );

    // --- LLM tool call ---
    let t = now();
    stage_log(
        5,
        "LLM tool routing",
        format!(
            "generating up to {MAX_NEW_TOKENS} tokens with {} context characters",
            context.chars().count()
        ),
    );
    let (call, raw) = llm.route(
        &transcript,
        &context,
        hits.first().map(|(d, _)| *d),
        MAX_NEW_TOKENS,
    )?;
    budget.add("llm", t.elapsed());
    println!("  [llm]    {raw}");
    println!("  [tool]   {} {:?}", call.tool, call.args);
    stage_log(
        5,
        "LLM tool routing",
        format!("validated tool: {} {:?}", call.tool, call.args),
    );

    // --- Mock execute ---
    stage_log(5, "LLM tool routing", format!("executing {}", call.tool));
    let reply = execute_tool(&call);
    println!("  [reply]  {reply}");
    stage_log(
        5,
        "LLM tool routing",
        format!("reply prepared: {reply:?} ({:.1} ms)", ms(t.elapsed())),
    );

    // --- TTS + playback ---
    if let Some(tts) = tts {
        stage_log(
            6,
            "Text to speech",
            format!("synthesizing {} reply characters", reply.chars().count()),
        );
        let t = now();
        let (pcm, sr) = tts.speak(&reply)?;
        budget.add("tts", t.elapsed());
        stage_log(
            6,
            "Text to speech",
            format!(
                "generated {} samples at {sr} Hz ({:.1} ms)",
                pcm.len(),
                ms(t.elapsed())
            ),
        );

        let t = now();
        play(&pcm, sr)?;
        budget.add("play", t.elapsed());
    } else {
        stage_log(6, "Text to speech", "skipped; TTS disabled");
    }
    budget.print();
    Ok(())
}

// ---------------------------------------------------------------------------
// Non-interactive modes
// ---------------------------------------------------------------------------

/// Pay every backend's lazy-init cost before the first user turn.
///
/// The first candle/whisper call is far more expensive than the rest, so it is
/// paid here rather than inside a turn. Each model is warmed separately so a
/// crash in one backend does not take the others down with it.
pub fn warmup(stt: &Stt, embedder: &Embedder, llm: &mut Llm) {
    let _ = embedder.embed_one("warmup");
    llm.warmup();
    // Silence warm-up: whisper allocates its compute buffers on the first decode.
    let _ = stt.warmup(&vec![0.0f32; TARGET_SR as usize]);
}

/// Full loop driven by a file: STT -> normalise -> RAG -> LLM -> TTS.
///
/// This makes the whole pipeline testable without a microphone grant.
pub fn run_wav(
    wav: &str,
    stt: &Stt,
    norm: &Normalizer,
    rag: &Rag,
    embedder: &Embedder,
    llm: &mut Llm,
    tts: Option<&mut Tts>,
) -> Result<()> {
    let pcm = read_wav_mono_f32(&PathBuf::from(wav))?;
    stage_log(
        1,
        "Voice capture",
        format!(
            "WAV input replaced microphone capture: {} decoded samples",
            pcm.len()
        ),
    );
    eprintln!("  wav:     {} samples @ {} Hz", pcm.len(), TARGET_SR);
    let mut b = Budget::default();
    let t = now();
    let raw = stt.transcribe(&pcm)?;
    b.add("stt", t.elapsed());
    println!("  [stt]    {raw:?}  ({:.0} ms)", ms(t.elapsed()));
    if raw.trim().is_empty() {
        println!("  (nothing intelligible)");
        return Ok(());
    }
    run_turn(norm, rag, embedder, llm, tts, raw, &mut b)
}

/// Audio-only probe: proves cpal capture -> whisper without needing a human.
pub fn run_probe_audio(stt: &Stt) -> Result<()> {
    println!(
        "\n  probe-audio: recording {}s of real mic input\n",
        RECORD_SECS
    );
    let t = now();
    let cap = record_until(RECORD_SECS)?;
    println!(
        "  captured {} samples @ {} Hz ({:.0} ms)",
        cap.samples.len(),
        TARGET_SR,
        ms(t.elapsed())
    );
    // Report signal statistics: proves we captured audio, not zeros.
    let peak = cap.samples.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let rms = {
        let sum: f32 = cap.samples.iter().map(|v| v * v).sum();
        (sum / cap.samples.len() as f32).sqrt()
    };
    println!("  peak={peak:.4}  rms={rms:.4}");
    if rms < 1e-4 {
        eprintln!("  WARNING: near-silent capture — check mic permissions / input gain");
    }
    let t = now();
    let text = stt.transcribe(&cap.samples)?;
    println!("  transcript: {text:?} ({:.0} ms)", ms(t.elapsed()));
    Ok(())
}

/// Selftest: exercise every stage against the bundled models.
pub fn run_selftest(
    norm: &Normalizer,
    rag: &Rag,
    embedder: &Embedder,
    llm: &mut Llm,
) -> Result<()> {
    let cases = [
        "um what's the price of an oil change",
        "book me a brake pad appointment tomorrow at 9am",
        "how much is a battery replacement on 2026-03-14",
        "I need 1,200 dollars of work done, is it free uh",
    ];
    println!("\n  selftest: {} cases\n", cases.len());
    let mut failures = 0;
    for (i, c) in cases.iter().enumerate() {
        stage_log(5, "LLM tool routing", format!("selftest case {}", i + 1));
        println!("  > {c}");
        let mut b = Budget::default();
        match run_turn(norm, rag, embedder, llm, None, c.to_string(), &mut b) {
            Ok(()) => {}
            Err(e) => {
                failures += 1;
                eprintln!("    FAILED: {e:#}");
            }
        }
        println!();
    }
    if failures == 0 {
        println!("  selftest: all {} cases passed", cases.len());
    } else {
        println!("  selftest: {failures} case(s) failed");
    }
    Ok(())
}
