//! Small audio DSP helpers shared by capture backends.
//!
//! These are pure functions so they are unit-testable on every platform;
//! capture backends use them to produce the mono 16 kHz stream the VAD/STT
//! pipeline expects, regardless of the device's native sample rate and
//! channel count.

/// Average interleaved multi-channel frames down to mono.
///
/// `samples` is interleaved (`channels` values per frame); a zero channel
/// count yields an empty vector.
pub fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channels = usize::from(channels.max(1));
    if samples.is_empty() || channels == 0 {
        return Vec::new();
    }

    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Linear-interpolation resampler from `from_hz` to `to_hz`.
///
/// Handles the common hardware mismatch (e.g. 48 kHz device → 16 kHz
/// pipeline). A length change without a rate change is a no-op pass-through;
/// equal rates always return a copy of the input.
pub fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if samples.is_empty() || from_hz == 0 || to_hz == 0 {
        return Vec::new();
    }
    if from_hz == to_hz {
        return samples.to_vec();
    }

    let ratio = f64::from(from_hz) / f64::from(to_hz);
    let out_len = ((samples.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);

    // Last output sample maps at or before the final input sample, so the
    // interpolation never reads out of bounds.
    for index in 0..out_len {
        let position = index as f64 * ratio;
        let base = position.floor() as usize;
        let frac = (position - base as f64) as f32;
        let next = (base + 1).min(samples.len() - 1);
        let a = samples[base];
        let b = samples[next];
        out.push(a + (b - a) * frac);
    }

    out
}

/// Convert raw interleaved device frames to mono `to_hz` samples in one step.
pub fn convert_to_mono_rate(
    samples: &[f32],
    channels: u16,
    from_hz: u32,
    to_hz: u32,
) -> Vec<f32> {
    resample_linear(&downmix_to_mono(samples, channels), from_hz, to_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo_averages_channels() {
        let stereo = [0.0, 1.0, 0.5, 0.5, -1.0, 1.0];
        assert_eq!(
            downmix_to_mono(&stereo, 2),
            vec![0.5, 0.5, 0.0]
        );
    }

    #[test]
    fn downmix_mono_is_identity() {
        let mono = [0.25, -0.5, 1.0];
        assert_eq!(downmix_to_mono(&mono, 1), mono.to_vec());
    }

    #[test]
    fn downmix_handles_partial_final_frame() {
        // 5 values at 2 channels: the trailing odd sample is dropped.
        let partial = [0.0, 1.0, 0.5, 0.5, 0.25];
        assert_eq!(downmix_to_mono(&partial, 2), vec![0.5, 0.5]);
    }

    #[test]
    fn downmix_empty_input() {
        assert!(downmix_to_mono(&[], 2).is_empty());
    }

    #[test]
    fn resample_downsamples_by_integer_factor() {
        // 4 samples @ 48k → 16k should produce ~1 sample per 3 input samples.
        let input = [0.0_f32, 3.0, 6.0, 9.0];
        let out = resample_linear(&input, 48, 16);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn resample_upsamples_interpolates() {
        let input = [0.0_f32, 1.0];
        let out = resample_linear(&input, 16, 32);
        // Doubling the rate doubles the sample count.
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn resample_same_rate_is_copy() {
        let input = [0.1_f32, 0.2, 0.3];
        assert_eq!(resample_linear(&input, 16_000, 16_000), input.to_vec());
    }

    #[test]
    fn resample_empty_and_degenerate_inputs() {
        assert!(resample_linear(&[], 48_000, 16_000).is_empty());
        assert!(resample_linear(&[1.0], 0, 16_000).is_empty());
        assert!(resample_linear(&[1.0], 48_000, 0).is_empty());
    }

    #[test]
    fn convert_end_to_end() {
        // Three stereo 48k frames → three mono samples → one 16k sample.
        let frames = [0.0_f32, 1.0, 0.4, 0.8, 0.2, 0.4];
        let out = convert_to_mono_rate(&frames, 2, 48_000, 16_000);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }
}
