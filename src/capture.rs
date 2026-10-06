// ---------------------------------------------------------------------------
// Stage 1 — audio capture (cpal)
// ---------------------------------------------------------------------------

use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::util::{now, stage_log};

/// Whisper requires exactly 16 kHz mono f32 in [-1, 1].
pub const TARGET_SR: u32 = 16_000;

/// Length of one microphone capture in interactive mode.
pub const RECORD_SECS: f32 = 5.0;

/// Result of one capture: mono f32 samples resampled to [`TARGET_SR`].
pub struct Capture {
    pub samples: Vec<f32>,
}

/// Records mono audio from the default input device for a fixed duration.
///
/// Two paths, because cpal's typed and raw APIs differ sharply:
///
/// * `SampleFormat::F32` uses the typed callback (`&[f32]`).
/// * any other format uses `build_input_stream_raw`, where samples arrive as raw
///   bytes that we reinterpret ourselves.
///
/// Either way the callback only appends to a mutex-guarded buffer. The resample
/// happens once, on this thread, after the stream is dropped — which guarantees
/// the audio thread has stopped touching the buffer.
pub fn record_until(seconds: f32) -> Result<Capture> {
    stage_log(
        1,
        "Voice capture",
        format!("starting {seconds:.1} second recording"),
    );
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device found"))?;

    // cpal 0.17 hands back *ranges*, not concrete configs, so ask for the
    // device's default and try to pin it to 16 kHz (Whisper's native rate).
    // Ask the device for its default input config, then pin to 16 kHz (Whisper's
    // native rate) if the device's advertised range covers it.
    let default_cfg = dev.default_input_config().context("default input config")?;
    let wanted = std::num::NonZeroU32::new(TARGET_SR);
    let sr_in = match wanted {
        Some(sr) if default_cfg.sample_rate() == sr.get() => sr.get(),
        _ => default_cfg.sample_rate(),
    };
    let cfg = cpal::StreamConfig {
        channels: default_cfg.channels(),
        sample_rate: sr_in,
        buffer_size: cpal::BufferSize::Default,
    };
    let channels = cfg.channels as usize;
    let format = default_cfg.sample_format();
    stage_log(
        1,
        "Voice capture",
        format!("input format: {sr_in} Hz, {channels} channels, {format:?}"),
    );

    let buf: Arc<std::sync::Mutex<Vec<f32>>> = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Collected in both arms so the stream is dropped at the end of the match,
    // guaranteeing no callback is still running when we read the buffer.
    if format == cpal::SampleFormat::F32 {
        let sink = Arc::clone(&buf);
        let stream = dev.build_input_stream(
            &cfg,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if let Ok(mut g) = sink.lock() {
                    g.extend_from_slice(data);
                }
            },
            |err| eprintln!("[audio] stream error: {err}"),
            None,
        )?;
        collect(&stream, &buf, seconds, sr_in)?;
    } else {
        let sink = Arc::clone(&buf);
        let stream = dev.build_input_stream_raw(
            &cfg,
            format,
            move |data: &cpal::Data, _: &cpal::InputCallbackInfo| {
                let Ok(mut g) = sink.lock() else { return };
                let Some(bytes) = data.bytes().get(..) else {
                    return;
                };
                // cpal hands us raw little-endian i16 on most CoreAudio configs.
                if bytes.len() % 2 != 0 {
                    return;
                }
                let s16: &[i16] = bytemuck::cast_slice(bytes);
                let mono: Vec<f32> = s16
                    .iter()
                    .step_by(channels)
                    .map(|&s| f32::from(s) / 32768.0)
                    .collect();
                g.extend_from_slice(&mono);
            },
            |err| eprintln!("[audio] stream error: {err}"),
            None,
        )?;
        collect(&stream, &buf, seconds, sr_in)?;
    }

    let samples = buf.lock().map(|g| g.clone()).unwrap_or_default();
    stage_log(
        1,
        "Voice capture",
        format!("captured {} source samples", samples.len()),
    );
    if samples.is_empty() {
        bail!("captured no audio (device may be muted or in use)");
    }

    // The raw path already downmixed per frame; the f32 path is interleaved.
    let mono: Vec<f32> = if format == cpal::SampleFormat::F32 {
        samples
            .chunks(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    let samples = resample(&mono, sr_in, TARGET_SR);
    stage_log(
        1,
        "Voice capture",
        format!(
            "converted to {} mono samples at {TARGET_SR} Hz",
            samples.len()
        ),
    );
    Ok(Capture { samples })
}

/// Runs a cpal stream until we have about `seconds` of audio, then returns.
///
/// Takes `&Stream` and relies on the caller dropping it before reading `buf`.
/// Emits one dot per second on stderr so the user can see the window open and,
/// more importantly, close — otherwise it is impossible to know when to stop
/// speaking.
fn collect(
    stream: &cpal::Stream,
    buf: &Arc<std::sync::Mutex<Vec<f32>>>,
    seconds: f32,
    sr_in: u32,
) -> Result<()> {
    stream.play().context("start input stream")?;
    let target = (sr_in as f64 * seconds as f64) as usize;
    let deadline = now() + Duration::from_secs_f32(seconds) + Duration::from_secs(5);
    let tick = Duration::from_secs_f32(seconds / 10.0).max(Duration::from_millis(100));

    loop {
        std::thread::sleep(tick);
        let len = buf.lock().map(|g| g.len()).unwrap_or(0);
        eprint!(".");
        io::stderr().flush().ok();
        if len >= target || now() > deadline {
            break;
        }
    }
    eprintln!(" done");
    Ok(())
}

/// Linear-interpolation resampler. Good enough for speech-to-text on a 16 kHz
/// target and far cheaper than a windowed-sinc filter.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if input.is_empty() || from == to {
        return input.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = (input.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let frac = (pos - i0 as f64) as f32;
        out.push(input[i0] * (1.0 - frac) + input[i1] * frac);
    }
    out
}
