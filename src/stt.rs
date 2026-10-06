// ---------------------------------------------------------------------------
// Stage 2 — speech-to-text (whisper-rs)
// ---------------------------------------------------------------------------

use std::path::Path;

use crate::capture::TARGET_SR;
use crate::util::stage_log;
use anyhow::{anyhow, Context, Result};

pub struct Stt {
    ctx: whisper_rs::WhisperContext,
}

impl Stt {
    pub fn load(path: &Path) -> Result<Self> {
        let params = whisper_rs::WhisperContextParameters::default();
        let ctx = whisper_rs::WhisperContext::new_with_params(
            path.to_str()
                .ok_or_else(|| anyhow!("non-utf8 whisper path"))?,
            params,
        )
        .map_err(|e| anyhow!("whisper init failed: {e}"))?;
        Ok(Self { ctx })
    }

    /// Transcribe 16 kHz mono f32 to lowercase-free raw text.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        self.transcribe_impl(audio, true)
    }

    /// Pay Whisper's first-decode initialization cost without presenting the
    /// synthetic warmup as a user speech-to-text stage.
    pub fn warmup(&self, audio: &[f32]) -> Result<String> {
        self.transcribe_impl(audio, false)
    }

    fn transcribe_impl(&self, audio: &[f32], log_stage: bool) -> Result<String> {
        if log_stage {
            stage_log(
                2,
                "Speech to text",
                format!("transcribing {} samples at {TARGET_SR} Hz", audio.len()),
            );
        }
        let mut state = self.ctx.create_state().context("create whisper state")?;

        let mut params =
            whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
        // NOTE: do not call set_n_threads(0). whisper.cpp treats 0 as "no worker
        // threads", which aborts the process ("Rust cannot catch foreign
        // exceptions") inside the Metal/BLAS dispatch. Leave it at the library
        // default of min(4, hardware_concurrency).
        params.set_language(Some("en"));
        params.set_translate(false);
        params.set_print_realtime(false);
        params.set_print_progress(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_single_segment(true);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.0);

        state
            .full(params, audio)
            .map_err(|e| anyhow!("whisper decode failed: {e}"))?;

        let n = state.full_n_segments();
        let mut out = String::new();
        for i in 0..n {
            if let Some(seg) = state.get_segment(i) {
                // to_str_lossy returns a Result in whisper-rs 0.16.
                if let Ok(s) = seg.to_str_lossy() {
                    out.push_str(&s);
                }
            }
        }
        let out = out.trim().to_string();
        if log_stage {
            if out.is_empty() {
                stage_log(2, "Speech to text", "no speech text was produced");
            } else {
                stage_log(
                    2,
                    "Speech to text",
                    format!("produced {} characters: {out:?}", out.chars().count()),
                );
            }
        }
        Ok(out)
    }
}
