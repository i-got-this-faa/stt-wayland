//! Voice activity detection and speech chunking via webrtc-vad.
//!
//! # Pipeline
//!
//! Consumes mono **f32 @ 16 kHz** frames from PipeWire. Each closed speech
//! utterance is emitted as a `Vec<f32>` on a **bounded** `mpsc` (backpressure
//! to the STT thread). In-progress speech sets a shared `speech_pending` flag
//! for the overlay dots indicator.
//!
//! # Frame size requirements (webrtc-vad)
//!
//! WebRTC VAD (`fvad` / `webrtc-vad` crate) accepts **only** frames of
//! **10, 20, or 30 ms** of signed **16-bit** PCM at 8 / 16 / 32 / 48 kHz.
//! Invalid lengths return an error from `is_voice_segment`.
//!
//! At the 16 kHz mono rate used by sttd:
//!
//! | Frame duration | Samples per frame |
//! |----------------|-------------------|
//! | 10 ms          | 160               |
//! | 20 ms          | 320               |
//! | 30 ms          | 480               |
//!
//! This module defaults to **10 ms (160 samples)** for lowest end-of-utterance
//! latency. f32 samples are converted to i16 **only** for VAD decisions; emitted
//! chunks keep the original f32 PCM for whisper-rs.
//!
//! # Silence gap
//!
//! After speech has been observed, a run of non-speech frames whose total
//! duration exceeds `silence_gap_ms` (config default **500 ms**) closes the
//! current chunk and sends it on the channel. Live mode is rolling (each VAD
//! close is one STT unit). Edit mode may also accumulate a full session buffer;
//! call [`Chunker::flush_all`] on Stop to force-close any open utterance.
//!
//! # Feature `stt-engine`
//!
//! When built **with** `stt-engine`, decisions come from `webrtc_vad::Vad`.
//! Without it, a simple RMS energy gate is used so the chunker still compiles
//! and is testable in minimal builds.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;

use anyhow::{bail, Result};
use tracing::{debug, trace, warn};

use crate::config::AudioConfig;

// ---------------------------------------------------------------------------
// Frame constants (16 kHz)
// ---------------------------------------------------------------------------

/// Capture / VAD sample rate expected by this module.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// 10 ms frame at 16 kHz — default webrtc-vad frame length.
pub const FRAME_SAMPLES_10MS: usize = 160;
/// 20 ms frame at 16 kHz.
pub const FRAME_SAMPLES_20MS: usize = 320;
/// 30 ms frame at 16 kHz.
pub const FRAME_SAMPLES_30MS: usize = 480;

/// Supported webrtc-vad frame durations in milliseconds.
pub const VALID_FRAME_MS: &[u32] = &[10, 20, 30];

/// Default VAD frame duration (ms).
pub const DEFAULT_FRAME_MS: u32 = 10;

/// Samples for a given frame duration at 16 kHz.
#[inline]
pub fn frame_samples(frame_ms: u32) -> Result<usize> {
    match frame_ms {
        10 => Ok(FRAME_SAMPLES_10MS),
        20 => Ok(FRAME_SAMPLES_20MS),
        30 => Ok(FRAME_SAMPLES_30MS),
        other => bail!(
            "webrtc-vad frame_ms must be 10, 20, or 30 (got {other}); \
             at 16 kHz that is 160 / 320 / 480 samples"
        ),
    }
}

// ---------------------------------------------------------------------------
// Channel helpers (bounded mpsc → STT thread)
// ---------------------------------------------------------------------------

/// Sender half of the VAD → STT chunk channel.
pub type ChunkSender = SyncSender<Vec<f32>>;
/// Receiver half of the VAD → STT chunk channel.
pub type ChunkReceiver = Receiver<Vec<f32>>;

/// Open a bounded channel for completed speech chunks.
///
/// Capacity is the number of `Vec<f32>` chunks (not samples). When full,
/// [`Chunker::push_samples`] blocks on `send` (backpressure) unless
/// non-blocking send is configured.
pub fn chunk_channel(capacity: usize) -> (ChunkSender, ChunkReceiver) {
    let cap = capacity.max(1);
    mpsc::sync_channel(cap)
}

// ---------------------------------------------------------------------------
// Speech-pending indicator (overlay dots)
// ---------------------------------------------------------------------------

/// Shared flag: `true` while the chunker is inside an open speech utterance.
#[derive(Clone, Debug, Default)]
pub struct SpeechPending(Arc<AtomicBool>);

impl SpeechPending {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn set(&self, pending: bool) {
        self.0.store(pending, Ordering::Relaxed);
    }

    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub fn as_arc(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
    }
}

// ---------------------------------------------------------------------------
// f32 ↔ i16 adapter for webrtc-vad
// ---------------------------------------------------------------------------

/// Convert a mono f32 sample in ≈[-1, 1] to PCM s16.
#[inline]
pub fn f32_to_i16(sample: f32) -> i16 {
    let x = (sample * 32768.0).round().clamp(-32768.0, 32767.0);
    x as i16
}

/// Convert a full f32 frame to i16 for VAD.
fn frame_f32_to_i16(src: &[f32], dst: &mut [i16]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f32_to_i16(*s);
    }
}

// ---------------------------------------------------------------------------
// Backend: webrtc-vad or energy gate
// ---------------------------------------------------------------------------

#[cfg(feature = "stt-engine")]
struct VadBackend {
    vad: webrtc_vad::Vad,
    i16_buf: Vec<i16>,
}

#[cfg(feature = "stt-engine")]
impl VadBackend {
    fn new(frame_len: usize) -> Self {
        // Aggressive: fewer false positives from keyboard / fan noise.
        let vad = webrtc_vad::Vad::new_with_rate_and_mode(
            webrtc_vad::SampleRate::Rate16kHz,
            webrtc_vad::VadMode::Aggressive,
        );
        Self {
            vad,
            i16_buf: vec![0i16; frame_len],
        }
    }

    fn is_speech(&mut self, frame: &[f32]) -> bool {
        if self.i16_buf.len() != frame.len() {
            self.i16_buf.resize(frame.len(), 0);
        }
        frame_f32_to_i16(frame, &mut self.i16_buf);
        match self.vad.is_voice_segment(&self.i16_buf) {
            Ok(v) => v,
            Err(()) => {
                warn!(
                    n = frame.len(),
                    "vad: invalid frame length for webrtc-vad (need 160/320/480 @ 16 kHz)"
                );
                false
            }
        }
    }

    fn reset(&mut self) {
        self.vad.reset();
        // new_with_rate_and_mode defaults are lost on reset — restore 16 kHz + Aggressive.
        self.vad.set_sample_rate(webrtc_vad::SampleRate::Rate16kHz);
        self.vad.set_mode(webrtc_vad::VadMode::Aggressive);
    }
}

/// Energy-based stand-in when `stt-engine` is disabled.
#[cfg(not(feature = "stt-engine"))]
struct VadBackend {
    /// RMS threshold on f32 samples (tuned for speech-ish levels).
    threshold: f32,
}

#[cfg(not(feature = "stt-engine"))]
impl VadBackend {
    fn new(_frame_len: usize) -> Self {
        Self { threshold: 0.01 }
    }

    fn is_speech(&mut self, frame: &[f32]) -> bool {
        if frame.is_empty() {
            return false;
        }
        let mut acc = 0.0f32;
        for &s in frame {
            acc += s * s;
        }
        let rms = (acc / frame.len() as f32).sqrt();
        rms >= self.threshold
    }

    fn reset(&mut self) {}
}

// ---------------------------------------------------------------------------
// Chunker
// ---------------------------------------------------------------------------

/// How completed chunks are delivered from [`Chunker::push_samples`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SendPolicy {
    /// Block when the channel is full (true backpressure). Default.
    #[default]
    Blocking,
    /// Drop the oldest behavior: if full, drop this chunk and log a warning.
    /// (Does not remove queued items — only skips the new send.)
    DropIfFull,
}

/// Rolling VAD chunker: frames in, silence-gap-closed `Vec<f32>` out.
///
/// * **Live** — each silence-closed chunk is sent on the mpsc; call
///   [`flush`](Chunker::flush) on Stop for a partial tail.
/// * **Edit** — same rolling chunks (STT still runs per chunk), or enable
///   [`set_accumulate_session`](Chunker::set_accumulate_session) and take the
///   full buffer via [`take_session_buffer`](Chunker::take_session_buffer) /
///   [`flush_all`](Chunker::flush_all) on Stop.
pub struct Chunker {
    sample_rate: u32,
    frame_ms: u32,
    frame_len: usize,
    /// Silence duration that closes an utterance (ms).
    silence_gap_ms: u32,
    /// Samples of non-speech accumulated while a chunk is open.
    silence_samples: u32,
    /// Incomplete frame leftover between pushes.
    partial: Vec<f32>,
    /// Open utterance being assembled (f32 mono).
    current: Vec<f32>,
    /// True once at least one speech frame entered `current`.
    in_speech: bool,
    backend: VadBackend,
    chunk_tx: Option<ChunkSender>,
    send_policy: SendPolicy,
    speech_pending: SpeechPending,
    /// When true, all samples that enter any utterance (and optional session
    /// path) are also copied into `session`.
    accumulate_session: bool,
    session: Vec<f32>,
}

impl Chunker {
    /// Build a chunker from audio config (default 10 ms frames, blocking send).
    pub fn from_config(cfg: &AudioConfig) -> Result<Self> {
        Self::new(
            cfg.sample_rate,
            cfg.silence_gap_ms,
            DEFAULT_FRAME_MS,
            None,
            SpeechPending::new(),
        )
    }

    /// Full constructor.
    ///
    /// `chunk_tx`: if `Some`, closed chunks are sent there. You can still also
    /// observe returns from [`push_samples`](Self::push_samples) / flush.
    pub fn new(
        sample_rate: u32,
        silence_gap_ms: u32,
        frame_ms: u32,
        chunk_tx: Option<ChunkSender>,
        speech_pending: SpeechPending,
    ) -> Result<Self> {
        if sample_rate != SAMPLE_RATE_HZ {
            bail!(
                "vad Chunker requires {SAMPLE_RATE_HZ} Hz mono (got {sample_rate}); \
                 webrtc-vad is configured for 16 kHz"
            );
        }
        let frame_len = frame_samples(frame_ms)?;
        let backend = VadBackend::new(frame_len);

        debug!(
            sample_rate,
            silence_gap_ms,
            frame_ms,
            frame_len,
            "vad: chunker init"
        );

        Ok(Self {
            sample_rate,
            frame_ms,
            frame_len,
            silence_gap_ms,
            silence_samples: 0,
            partial: Vec::with_capacity(frame_len),
            current: Vec::new(),
            in_speech: false,
            backend,
            chunk_tx,
            send_policy: SendPolicy::Blocking,
            speech_pending,
            accumulate_session: false,
            session: Vec::new(),
        })
    }

    /// Attach (or replace) the bounded chunk sender.
    pub fn set_chunk_sender(&mut self, tx: Option<ChunkSender>) {
        self.chunk_tx = tx;
    }

    pub fn set_send_policy(&mut self, policy: SendPolicy) {
        self.send_policy = policy;
    }

    /// When `true`, all audio that is part of speech utterances is also
    /// appended to an internal session buffer (Edit-mode full capture).
    pub fn set_accumulate_session(&mut self, on: bool) {
        self.accumulate_session = on;
        if !on {
            self.session.clear();
        }
    }

    pub fn speech_pending(&self) -> &SpeechPending {
        &self.speech_pending
    }

    pub fn silence_gap_ms(&self) -> u32 {
        self.silence_gap_ms
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn frame_ms(&self) -> u32 {
        self.frame_ms
    }

    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Whether an utterance is currently open (speech seen, not yet gap-closed).
    pub fn is_in_speech(&self) -> bool {
        self.in_speech
    }

    /// Samples currently buffered in the open utterance (not yet emitted).
    pub fn pending_samples(&self) -> usize {
        self.current.len()
    }

    /// Borrow the open utterance PCM (for Live in-progress preview STT).
    pub fn peek_current(&self) -> &[f32] {
        &self.current
    }

    /// Push mono f32 @ 16 kHz. May close zero or more chunks via silence gap.
    ///
    /// Returns the **last** chunk closed during this call (if any). All closed
    /// chunks are also sent on the bounded channel when configured.
    pub fn push_samples(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        if samples.is_empty() {
            return None;
        }

        // Append to partial, then process whole frames.
        self.partial.extend_from_slice(samples);

        let mut last_closed: Option<Vec<f32>> = None;
        let frame_len = self.frame_len;
        while self.partial.len() >= frame_len {
            // Take ownership of one frame without leaving a hole in `partial`.
            let frame: Vec<f32> = self.partial.drain(..frame_len).collect();
            if let Some(chunk) = self.process_frame(&frame) {
                last_closed = Some(chunk);
            }
        }
        last_closed
    }

    /// Force-close the current utterance (e.g. Pause edge or end of gap logic
    /// without waiting). Does **not** clear the session accumulator.
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        self.close_current()
    }

    /// Force-close any open utterance and, if session accumulation is on,
    /// return the entire session buffer (including the just-closed tail).
    ///
    /// Intended for **Stop**: Live callers can ignore the session and only
    /// use the channel / returned last chunk; Edit callers take the full buffer.
    ///
    /// When `accumulate_session` is false, this is equivalent to [`flush`](Self::flush)
    /// plus a reset of VAD hangover state.
    pub fn flush_all(&mut self) -> Option<Vec<f32>> {
        let tail = self.close_current();
        self.backend.reset();
        self.partial.clear();
        self.silence_samples = 0;

        if self.accumulate_session {
            if self.session.is_empty() {
                return tail;
            }
            let full = std::mem::take(&mut self.session);
            // Prefer full session; tail was already appended into session on close.
            let _ = tail;
            Some(full)
        } else {
            tail
        }
    }

    /// Take and clear the session accumulator without closing the open chunk.
    pub fn take_session_buffer(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.session)
    }

    /// Clear open utterance, partial frame, session, and speech-pending.
    pub fn reset(&mut self) {
        self.partial.clear();
        self.current.clear();
        self.session.clear();
        self.silence_samples = 0;
        self.in_speech = false;
        self.speech_pending.set(false);
        self.backend.reset();
    }

    // -- internal ------------------------------------------------------------

    fn process_frame(&mut self, frame: &[f32]) -> Option<Vec<f32>> {
        let speech = self.backend.is_speech(frame);
        self.apply_frame(frame, speech)
    }

    /// Apply one complete frame with an explicit speech decision.
    ///
    /// Used by the VAD backend path and by unit tests to exercise silence-gap
    /// logic without depending on webrtc-vad's spectral heuristics.
    fn apply_frame(&mut self, frame: &[f32], speech: bool) -> Option<Vec<f32>> {
        debug_assert_eq!(frame.len(), self.frame_len);

        if speech {
            if !self.in_speech {
                self.in_speech = true;
                self.speech_pending.set(true);
                trace!("vad: speech start");
            }
            self.current.extend_from_slice(frame);
            self.silence_samples = 0;
            return None;
        }

        // Non-speech frame
        if !self.in_speech {
            // Leading silence — drop (do not start a chunk).
            return None;
        }

        // Hangover: keep trailing silence inside the chunk for whisper.
        self.current.extend_from_slice(frame);
        self.silence_samples = self
            .silence_samples
            .saturating_add(self.frame_len as u32);

        let gap_samples = silence_gap_to_samples(self.silence_gap_ms, self.sample_rate);
        if self.silence_samples >= gap_samples {
            trace!(
                chunk_samples = self.current.len(),
                silence_ms = self.silence_gap_ms,
                "vad: silence gap — close chunk"
            );
            self.close_current()
        } else {
            None
        }
    }

    fn close_current(&mut self) -> Option<Vec<f32>> {
        if self.current.is_empty() {
            self.in_speech = false;
            self.speech_pending.set(false);
            self.silence_samples = 0;
            return None;
        }

        let chunk = std::mem::take(&mut self.current);
        self.in_speech = false;
        self.silence_samples = 0;
        self.speech_pending.set(false);

        if self.accumulate_session {
            self.session.extend_from_slice(&chunk);
        }

        debug!(n_samples = chunk.len(), "vad: emit chunk");
        self.deliver(chunk)
    }

    fn deliver(&mut self, chunk: Vec<f32>) -> Option<Vec<f32>> {
        if let Some(tx) = self.chunk_tx.as_ref() {
            match self.send_policy {
                SendPolicy::Blocking => {
                    if let Err(e) = tx.send(chunk.clone()) {
                        warn!("vad: chunk channel disconnected: {e}");
                    }
                }
                SendPolicy::DropIfFull => match tx.try_send(chunk.clone()) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        warn!(
                            n_samples = chunk.len(),
                            "vad: chunk channel full — dropping closed chunk (backpressure)"
                        );
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        warn!("vad: chunk channel disconnected");
                    }
                },
            }
        }
        Some(chunk)
    }
}

#[inline]
fn silence_gap_to_samples(gap_ms: u32, sample_rate: u32) -> u32 {
    // gap_ms * sample_rate / 1000 — use u64 to avoid overflow on large gaps.
    ((gap_ms as u64) * (sample_rate as u64) / 1000) as u32
}

// ---------------------------------------------------------------------------
// VadEngine — thin facade kept for main.rs scaffold wiring
// ---------------------------------------------------------------------------

/// High-level VAD engine used by the daemon scaffold.
///
/// Wraps a [`Chunker`] plus optional channel endpoints created from
/// [`AudioConfig::stt_channel_capacity`].
pub struct VadEngine {
    chunker: Chunker,
    /// Receiver side for the STT thread (moved out via [`take_receiver`](Self::take_receiver)).
    chunk_rx: Option<ChunkReceiver>,
}

impl VadEngine {
    pub fn new(cfg: &AudioConfig) -> Result<Self> {
        let (tx, rx) = chunk_channel(cfg.stt_channel_capacity);
        let speech_pending = SpeechPending::new();
        let mut chunker = Chunker::new(
            cfg.sample_rate,
            cfg.silence_gap_ms,
            DEFAULT_FRAME_MS,
            Some(tx),
            speech_pending,
        )?;
        // Default Live-style rolling; enable session accumulate when entering Edit.
        chunker.set_accumulate_session(false);

        Ok(Self {
            chunker,
            chunk_rx: Some(rx),
        })
    }

    pub fn silence_gap_ms(&self) -> u32 {
        self.chunker.silence_gap_ms()
    }

    pub fn sample_rate(&self) -> u32 {
        self.chunker.sample_rate()
    }

    pub fn speech_pending(&self) -> &SpeechPending {
        self.chunker.speech_pending()
    }

    pub fn chunker(&self) -> &Chunker {
        &self.chunker
    }

    pub fn chunker_mut(&mut self) -> &mut Chunker {
        &mut self.chunker
    }

    /// Take the STT-side receiver (once).
    pub fn take_receiver(&mut self) -> Option<ChunkReceiver> {
        self.chunk_rx.take()
    }

    /// Push interleaved mono f32 samples; may return a closed chunk.
    pub fn push_samples(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        self.chunker.push_samples(samples)
    }

    /// Force-close any in-progress chunk (e.g. on Stop).
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        self.chunker.flush()
    }

    /// Force-close and optionally return full session buffer (Edit / Stop).
    pub fn flush_all(&mut self) -> Option<Vec<f32>> {
        self.chunker.flush_all()
    }

    /// Enable/disable full-session accumulation (Edit mode).
    pub fn set_accumulate_session(&mut self, on: bool) {
        self.chunker.set_accumulate_session(on);
    }

    pub fn reset(&mut self) {
        self.chunker.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    /// Synthetic “speech”: loud sine-ish energy so energy-gate and real VAD
    /// both tend to fire (real VAD may still be picky — we mostly test framing).
    fn loud(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE_HZ as f32;
                (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.5
            })
            .collect()
    }

    #[test]
    fn frame_samples_table() {
        assert_eq!(frame_samples(10).unwrap(), 160);
        assert_eq!(frame_samples(20).unwrap(), 320);
        assert_eq!(frame_samples(30).unwrap(), 480);
        assert!(frame_samples(15).is_err());
    }

    #[test]
    fn f32_to_i16_range() {
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), 32767);
        assert_eq!(f32_to_i16(-1.0), -32768);
    }

    #[test]
    fn silence_gap_samples() {
        assert_eq!(silence_gap_to_samples(500, 16_000), 8000);
        assert_eq!(silence_gap_to_samples(10, 16_000), 160);
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let err = Chunker::new(48_000, 500, 10, None, SpeechPending::new());
        assert!(err.is_err());
    }

    #[test]
    fn partial_frames_held() {
        let mut c = Chunker::new(16_000, 500, 10, None, SpeechPending::new()).unwrap();
        // 100 samples < 160 — no frame processed, no panic.
        assert!(c.push_samples(&loud(100)).is_none());
        assert_eq!(c.partial.len(), 100);
    }

    #[test]
    fn flush_empty_is_none() {
        let mut c = Chunker::new(16_000, 500, 10, None, SpeechPending::new()).unwrap();
        c.push_samples(&silence(FRAME_SAMPLES_10MS * 5));
        assert!(c.flush().is_none());
        assert!(!c.speech_pending().get());
    }

    #[test]
    fn speech_then_gap_emits_chunk() {
        // Deterministic path: drive apply_frame so we do not depend on
        // webrtc-vad accepting a synthetic sine as speech.
        let pending = SpeechPending::new();
        let (tx, rx) = chunk_channel(4);
        // 50 ms gap @ 10 ms frames = 5 silent frames after speech.
        let mut c = Chunker::new(16_000, 50, 10, Some(tx), pending.clone()).unwrap();
        let frame = loud(FRAME_SAMPLES_10MS);

        for _ in 0..10 {
            assert!(c.apply_frame(&frame, true).is_none());
        }
        assert!(c.is_in_speech());
        assert!(pending.get());

        let mut closed = None;
        for _ in 0..5 {
            closed = c.apply_frame(&silence(FRAME_SAMPLES_10MS), false);
        }
        assert!(closed.is_some(), "expected chunk after silence gap");
        let chunk = closed.unwrap();
        // 10 speech + 5 hangover silence frames
        assert_eq!(chunk.len(), FRAME_SAMPLES_10MS * 15);
        assert!(!pending.get());
        assert!(!c.is_in_speech());

        let from_rx = rx.try_recv().expect("chunk also on channel");
        assert_eq!(from_rx.len(), chunk.len());
    }

    #[test]
    fn flush_all_with_session() {
        let mut c = Chunker::new(16_000, 5000, 10, None, SpeechPending::new()).unwrap();
        c.set_accumulate_session(true);

        let frame = loud(FRAME_SAMPLES_10MS);
        for _ in 0..10 {
            c.apply_frame(&frame, true);
        }
        let out = c.flush_all().expect("session audio");
        assert_eq!(out.len(), FRAME_SAMPLES_10MS * 10);
        assert!(c.take_session_buffer().is_empty());
    }

    #[test]
    fn leading_silence_dropped() {
        let mut c = Chunker::new(16_000, 50, 10, None, SpeechPending::new()).unwrap();
        for _ in 0..20 {
            assert!(c.apply_frame(&silence(FRAME_SAMPLES_10MS), false).is_none());
        }
        assert_eq!(c.pending_samples(), 0);
        assert!(!c.speech_pending().get());
    }

    #[test]
    fn multiple_utterances_rolling() {
        let mut c = Chunker::new(16_000, 30, 10, None, SpeechPending::new()).unwrap();
        let speech = loud(FRAME_SAMPLES_10MS);
        let sil = silence(FRAME_SAMPLES_10MS);

        // Utterance 1
        c.apply_frame(&speech, true);
        let mut n = 0;
        for _ in 0..3 {
            if c.apply_frame(&sil, false).is_some() {
                n += 1;
            }
        }
        assert_eq!(n, 1);

        // Utterance 2
        c.apply_frame(&speech, true);
        c.apply_frame(&speech, true);
        let second = c.flush().expect("open tail");
        assert_eq!(second.len(), FRAME_SAMPLES_10MS * 2);
    }

    #[test]
    fn channel_backpressure_capacity() {
        let (tx, rx) = chunk_channel(1);
        tx.send(vec![1.0]).unwrap();
        // Second try_send should be Full
        match tx.try_send(vec![2.0]) {
            Err(TrySendError::Full(_)) => {}
            other => panic!("expected Full, got {other:?}"),
        }
        let _ = rx.recv().unwrap();
    }

    #[test]
    fn vad_engine_from_config() {
        let cfg = AudioConfig::default();
        let mut eng = VadEngine::new(&cfg).unwrap();
        assert_eq!(eng.sample_rate(), 16_000);
        assert_eq!(eng.silence_gap_ms(), 500);
        assert!(eng.take_receiver().is_some());
        assert!(eng.take_receiver().is_none());
    }
}
