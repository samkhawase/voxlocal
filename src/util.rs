// ---------------------------------------------------------------------------
// Utilities — per-stage budget, model-path resolution, tty probe
// ---------------------------------------------------------------------------

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};

/// Monotonic-ish wall clock, used only for reporting.
#[inline]
pub fn now() -> Instant {
    Instant::now()
}

/// Milliseconds since a start point, as an f32.
pub fn ms(d: Duration) -> f32 {
    d.as_secs_f32() * 1000.0
}

/// Print a normal processing message using the stable stage-log format.
pub fn stage_log(stage: u8, name: &str, message: impl AsRef<str>) {
    println!("[Stage {stage}: {name}]: {}", message.as_ref());
}

/// Accumulates per-stage timings for one turn so we can print a latency budget.
#[derive(Default)]
pub struct Budget {
    rows: Vec<(&'static str, f32)>,
}

impl Budget {
    pub fn add(&mut self, stage: &'static str, d: Duration) {
        self.rows.push((stage, ms(d)));
    }

    pub fn print(&self) {
        if self.rows.is_empty() {
            return;
        }
        let total: f32 = self.rows.iter().map(|(_, v)| *v).sum();
        let w = self
            .rows
            .iter()
            .map(|(k, _)| k.len())
            .max()
            .unwrap_or(6)
            .max(6);
        eprintln!("  latency budget:");
        for (stage, v) in &self.rows {
            eprintln!("    {:<w$}  {:>7.1} ms", stage, v, w = w);
        }
        eprintln!(
            "    {:<w$}  {:>7.1} ms  <- total compute",
            "TOTAL",
            total,
            w = w
        );
    }
}

/// Resolution of every model path, rooted at `.models/` next to the binary's crate root.
pub struct Paths {
    pub llm: PathBuf,
    pub llm_tok: PathBuf,
    pub embed_dir: PathBuf,
    pub stt: PathBuf,
    pub tts_onnx: PathBuf,
    pub tts_cfg: PathBuf,
}

impl Paths {
    pub fn discover() -> Result<Self> {
        // Prefer $VOXLOCAL_MODELS, then ./models, then ./.models.
        let root = std::env::var("VOXLOCAL_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                [PathBuf::from(".models"), PathBuf::from("models")]
                    .into_iter()
                    .find(|p| p.is_dir())
                    .unwrap_or_else(|| PathBuf::from(".models"))
            });

        let p = Self {
            llm: root.join("llm/smollm2-135m-q4km.gguf"),
            llm_tok: root.join("llm/tokenizer.json"),
            embed_dir: root.join("embed"),
            stt: root.join("stt/ggml-tiny.en.bin"),
            tts_onnx: root.join("tts/en_US-lessac-medium.onnx"),
            tts_cfg: root.join("tts/en_US-lessac-medium.onnx.json"),
        };

        for (label, path) in [
            ("LLM GGUF", &p.llm),
            ("LLM tokenizer", &p.llm_tok),
            ("MiniLM config.json", &p.embed_dir.join("config.json")),
            (
                "MiniLM model.safetensors",
                &p.embed_dir.join("model.safetensors"),
            ),
            ("MiniLM tokenizer.json", &p.embed_dir.join("tokenizer.json")),
            ("Whisper ggml", &p.stt),
            ("Piper onnx", &p.tts_onnx),
            ("Piper onnx.json", &p.tts_cfg),
        ] {
            if !path.exists() {
                bail!(
                    "missing {label}: {}\nRun ./setup.sh to download all models into .models/",
                    path.display()
                );
            }
        }
        Ok(p)
    }
}

/// True when stdin is a terminal, not a pipe or file.
///
/// Interactive mode prompts once per turn; with piped stdin that either spins on
/// EOF or looks frozen, so we refuse up front instead.
pub fn is_tty() -> bool {
    // Avoid a libc dependency: use the documented shell-free probe.
    unsafe { libc_isatty(0) == 1 }
}

extern "C" {
    #[link_name = "isatty"]
    fn libc_isatty(fd: i32) -> i32;
}
