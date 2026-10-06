// ---------------------------------------------------------------------------
// Stage 7 — audio playback (rodio)
// ---------------------------------------------------------------------------

use anyhow::{anyhow, Result};

use crate::capture::resample;
use crate::util::stage_log;

/// Play f32 mono PCM through the default output device.
pub fn play(samples: &[f32], sample_rate: u32) -> Result<()> {
    stage_log(
        7,
        "Audio playback",
        format!("playing {} samples at {sample_rate} Hz", samples.len()),
    );
    let mut builder = rodio::DeviceSinkBuilder::open_default_sink()
        .map_err(|e| anyhow!("open output sink: {e}"))?;
    // rodio logs a banner when the sink drops; we drop it on purpose every turn.
    builder.log_on_drop(false);
    let sink = builder;
    let cfg = sink.config();
    let device_rate = cfg.sample_rate();
    let device_channels = cfg.channel_count();

    // Resample if the device is not at the model's native rate.
    // rodio's mixer also converts mono -> device channel count.
    let owned;
    let data: &[f32] = if device_rate.get() != sample_rate {
        stage_log(
            7,
            "Audio playback",
            format!("resampling for output device at {} Hz", device_rate.get()),
        );
        owned = resample(samples, sample_rate, device_rate.get());
        &owned
    } else {
        samples
    };

    let source = rodio::buffer::SamplesBuffer::new(device_channels, device_rate, data.to_vec());
    let player = rodio::Player::connect_new(sink.mixer());
    player.append(source);
    player.sleep_until_end();
    stage_log(7, "Audio playback", "playback complete");
    Ok(())
}
