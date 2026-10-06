// ---------------------------------------------------------------------------
// Stage 6 — Text-to-speech (piper-rs)
// ---------------------------------------------------------------------------

use std::path::Path;

use anyhow::{anyhow, Result};

pub struct Tts {
    piper: piper_rs::Piper,
}

impl Tts {
    pub fn load(onnx: &Path, cfg: &Path) -> Result<Self> {
        let piper =
            piper_rs::Piper::new(onnx, cfg).map_err(|e| anyhow!("piper init failed: {e}"))?;
        Ok(Self { piper })
    }

    /// Synthesise `text` to mono f32 PCM.
    ///
    /// `length_scale > 1` slows speech down; Piper uses the inverse convention
    /// of "duration factor", so a value >1 makes it longer.
    pub fn speak(&mut self, text: &str) -> Result<(Vec<f32>, u32)> {
        self.piper
            .create(text, false, None, None, None, None)
            .map_err(|e| anyhow!("piper synth failed: {e}"))
    }
}
