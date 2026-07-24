//! Speech-to-text engine (whisper-rs over whisper.cpp).
//!
//! # Architecture
//!
//! A **dedicated OS thread** owns the [`WhisperContext`] / [`WhisperState`] pair
//! (whisper contexts are not cheaply shared across threads). Callers talk to it
//! through bounded `mpsc` channels:
//!
//! - **In**: [`SttCommand`] jobs (`LiveChunk`, `LivePreview`, `EditFull`, …)
//! - **Out**: [`TranscriptResult`] `{ text, is_final }` for live/preview path
//!
//! VAD-closed chunks are expected on a bounded channel (backpressure when the
//! worker is slower than speech). Edit mode uses a request/reply oneshot-style
//! channel so the full buffer can be transcribed in one shot.
//!
//! # Models
//!
//! English-only GGML models (`base.en` / `small.en`) under
//! `~/.local/share/sttd/models/`. Residency: `preload` (default) or `lazy`.
//!
//! # Feature flag
//!
//! Real inference requires the `stt-engine` feature (links `whisper-rs`). Without
//! it the worker still starts and the API is complete, but load/transcribe
//! return clear errors. Build with cmake + clang for whisper.cpp.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvError, SendError, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::fs;
use std::io::{self, Read, Write};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use tracing::{debug, error, info, warn};

use crate::config::{ModelConfig, ModelResidency};

/// Whisper + webrtc-vad capture rate (mono f32).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Default Live-mode audio overlap for chunk continuity (~200 ms @ 16 kHz).
pub const DEFAULT_OVERLAP_MS: u32 = 200;

/// Default bounded channel capacity when not taken from config.
pub const DEFAULT_CHANNEL_CAPACITY: usize = 8;

/// Committed or partial transcript from the STT worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptResult {
    pub text: String,
    /// `true` for VAD-closed Live chunks and Edit full-buffer results.
    /// `false` for in-progress Live preview (overlay-only).
    pub is_final: bool,
}

/// Jobs accepted by the STT worker thread.
#[derive(Debug)]
pub enum SttCommand {
    /// Live mode: VAD-closed speech chunk → committed text (`is_final = true`).
    LiveChunk(Vec<f32>),
    /// Live mode: in-progress audio for overlay preview only (`is_final = false`).
    LivePreview(Vec<f32>),
    /// Edit mode: transcribe the entire buffer once; reply on `reply`.
    EditFull {
        samples: Vec<f32>,
        reply: mpsc::Sender<Result<TranscriptResult>>,
    },
    /// Load the model if not yet resident (lazy residency / explicit warm-up).
    EnsureLoaded {
        reply: mpsc::Sender<Result<()>>,
    },
    /// Clear Live overlap tail + prompt state (e.g. between Stop → Start).
    ResetSession,
    /// Stop the worker loop and drop the model.
    Shutdown,
}

/// Handle to the dedicated STT OS thread.
///
/// Send audio via [`Self::push_chunk`] / [`Self::try_push_chunk`] (bounded,
/// backpressured). Poll committed/preview text with [`Self::recv_result`] /
/// [`Self::try_recv_result`]. Edit mode uses [`Self::transcribe_full`].
pub struct SttEngine {
    cmd_tx: SyncSender<SttCommand>,
    result_rx: Receiver<TranscriptResult>,
    join: Option<JoinHandle<()>>,
    model_path: PathBuf,
    residency: ModelResidency,
    loaded: Arc<AtomicBool>,
}

impl SttEngine {
    /// Spawn the worker with default channel capacity and overlap.
    ///
    /// With `ModelResidency::Preload`, blocks until the model is loaded (or
    /// returns a clear error if the file is missing / unloadable).
    pub fn new(cfg: &ModelConfig) -> Result<Self> {
        Self::spawn(cfg, DEFAULT_CHANNEL_CAPACITY, DEFAULT_OVERLAP_MS)
    }

    /// Spawn the worker thread.
    ///
    /// * `channel_capacity` — bound for command + result channels (backpressure).
    /// * `overlap_ms` — Live-mode audio tail kept from the previous chunk.
    pub fn spawn(cfg: &ModelConfig, channel_capacity: usize, overlap_ms: u32) -> Result<Self> {
        let model_path = cfg.model_path();
        let capacity = channel_capacity.max(1);
        let overlap_samples = ms_to_samples(overlap_ms, SAMPLE_RATE_HZ);
        let threads = resolve_threads(cfg.threads);
        let language = if cfg.language.is_empty() {
            "en".to_string()
        } else {
            cfg.language.clone()
        };
        let residency = cfg.residency;
        let loaded = Arc::new(AtomicBool::new(false));

        info!(
            path = %model_path.display(),
            residency = ?residency,
            capacity,
            overlap_ms,
            threads,
            language = %language,
            "stt: spawning worker thread"
        );

        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<SttCommand>(capacity);
        let (result_tx, result_rx) = mpsc::sync_channel::<TranscriptResult>(capacity);

        let worker_cfg = WorkerConfig {
            model_path: model_path.clone(),
            residency,
            threads,
            language,
            overlap_samples,
            preload: residency == ModelResidency::Preload,
            loaded: Arc::clone(&loaded),
        };

        let join = thread::Builder::new()
            .name("sttd-whisper".into())
            .spawn(move || worker_main(worker_cfg, cmd_rx, result_tx))
            .context("spawn stt worker thread")?;

        let engine = Self {
            cmd_tx,
            result_rx,
            join: Some(join),
            model_path,
            residency,
            loaded,
        };

        if residency == ModelResidency::Preload {
            engine
                .ensure_loaded()
                .context("preload whisper model at startup")?;
        }

        Ok(engine)
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn residency(&self) -> ModelResidency {
        self.residency
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Acquire)
    }

    /// Load the model on the worker thread if not already resident.
    pub fn ensure_loaded(&self) -> Result<()> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.cmd_tx
            .send(SttCommand::EnsureLoaded { reply: reply_tx })
            .map_err(|_| anyhow!("stt worker thread is not running"))?;
        reply_rx
            .recv_timeout(Duration::from_secs(120))
            .map_err(|_| anyhow!("timed out waiting for whisper model load"))?
    }

    /// Live: enqueue a VAD-closed chunk (blocks when the channel is full).
    pub fn push_chunk(&self, samples: Vec<f32>) -> Result<()> {
        self.cmd_tx
            .send(SttCommand::LiveChunk(samples))
            .map_err(|_| anyhow!("stt worker thread is not running"))
    }

    /// Live: non-blocking enqueue; `Full` means backpressure (caller should drop or wait).
    pub fn try_push_chunk(&self, samples: Vec<f32>) -> Result<(), TrySendError<SttCommand>> {
        self.cmd_tx.try_send(SttCommand::LiveChunk(samples))
    }

    /// Live: in-progress audio for overlay preview (`is_final = false` results).
    pub fn push_preview(&self, samples: Vec<f32>) -> Result<()> {
        self.cmd_tx
            .send(SttCommand::LivePreview(samples))
            .map_err(|_| anyhow!("stt worker thread is not running"))
    }

    pub fn try_push_preview(&self, samples: Vec<f32>) -> Result<(), TrySendError<SttCommand>> {
        self.cmd_tx.try_send(SttCommand::LivePreview(samples))
    }

    /// Blocking receive of the next Live/preview transcript.
    pub fn recv_result(&mut self) -> Result<TranscriptResult, RecvError> {
        self.result_rx.recv()
    }

    /// Receive with timeout (e.g. event-loop polling).
    pub fn recv_result_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<TranscriptResult, mpsc::RecvTimeoutError> {
        self.result_rx.recv_timeout(timeout)
    }

    /// Non-blocking receive.
    pub fn try_recv_result(&mut self) -> Result<Option<TranscriptResult>, mpsc::RecvError> {
        match self.result_rx.try_recv() {
            Ok(r) => Ok(Some(r)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(RecvError),
        }
    }

    /// Edit mode: transcribe a full capture buffer on the worker thread.
    pub fn transcribe_full(&self, samples: Vec<f32>) -> Result<TranscriptResult> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.cmd_tx
            .send(SttCommand::EditFull {
                samples,
                reply: reply_tx,
            })
            .map_err(|_| anyhow!("stt worker thread is not running"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow!("stt worker dropped edit reply"))?
    }

    /// Convenience: full-buffer transcribe, text only (Edit path).
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        Ok(self.transcribe_full(samples.to_vec())?.text)
    }

    /// Clear Live overlap + prompt continuity (call on Stop / mode switch).
    pub fn reset_session(&self) -> Result<()> {
        self.cmd_tx
            .send(SttCommand::ResetSession)
            .map_err(|_| anyhow!("stt worker thread is not running"))
    }

    /// Signal shutdown and join the worker (also done in [`Drop`]).
    pub fn shutdown(mut self) -> Result<()> {
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> Result<()> {
        // Ignore send failure if the worker already exited.
        let _ = self.cmd_tx.send(SttCommand::Shutdown);
        if let Some(join) = self.join.take() {
            join.join()
                .map_err(|_| anyhow!("stt worker thread panicked"))?;
        }
        Ok(())
    }

    /// Clone of the command sender for VAD / audio threads.
    pub fn command_sender(&self) -> SyncSender<SttCommand> {
        self.cmd_tx.clone()
    }
}

impl Drop for SttEngine {
    fn drop(&mut self) {
        if self.join.is_some() {
            if let Err(err) = self.shutdown_inner() {
                warn!(error = %err, "stt: error while shutting down worker");
            }
        }
    }
}

// --- worker thread ------------------------------------------------------------

struct WorkerConfig {
    model_path: PathBuf,
    residency: ModelResidency,
    threads: i32,
    language: String,
    overlap_samples: usize,
    preload: bool,
    loaded: Arc<AtomicBool>,
}

fn worker_main(
    cfg: WorkerConfig,
    cmd_rx: Receiver<SttCommand>,
    result_tx: SyncSender<TranscriptResult>,
) {
    info!(
        path = %cfg.model_path.display(),
        residency = ?cfg.residency,
        "stt worker: started"
    );

    let mut worker = SttWorker::new(cfg);

    if worker.cfg.preload {
        if let Err(err) = worker.ensure_loaded() {
            error!(error = %err, "stt worker: preload failed (will retry on demand)");
            // Preload path on the handle also calls EnsureLoaded and surfaces the error.
        }
    }

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            SttCommand::Shutdown => {
                debug!("stt worker: shutdown");
                break;
            }
            SttCommand::ResetSession => {
                worker.reset_session();
            }
            SttCommand::EnsureLoaded { reply } => {
                let res = worker.ensure_loaded();
                let _ = reply.send(res);
            }
            SttCommand::LiveChunk(samples) => {
                match worker.transcribe_live(samples, true) {
                    Ok(result) => {
                        if send_result(&result_tx, result).is_err() {
                            warn!("stt worker: result consumer gone; exiting");
                            break;
                        }
                    }
                    Err(err) => {
                        error!(error = %err, "stt worker: live chunk failed");
                    }
                }
            }
            SttCommand::LivePreview(samples) => {
                match worker.transcribe_live(samples, false) {
                    Ok(result) => {
                        if send_result(&result_tx, result).is_err() {
                            warn!("stt worker: result consumer gone; exiting");
                            break;
                        }
                    }
                    Err(err) => {
                        error!(error = %err, "stt worker: live preview failed");
                    }
                }
            }
            SttCommand::EditFull { samples, reply } => {
                let res = worker.transcribe_edit(samples);
                let _ = reply.send(res);
            }
        }
    }

    worker.unload();
    info!("stt worker: exited");
}

fn send_result(
    tx: &SyncSender<TranscriptResult>,
    result: TranscriptResult,
) -> Result<(), SendError<TranscriptResult>> {
    tx.send(result)
}

/// Worker-local state; lives only on the `sttd-whisper` thread.
struct SttWorker {
    cfg: WorkerConfig,
    /// Tail of the last Live chunk, prepended to the next for continuity.
    overlap_tail: Vec<f32>,
    /// Last committed Live text (whisper initial_prompt).
    last_prompt: String,
    #[cfg(feature = "stt-engine")]
    ctx: Option<whisper_rs::WhisperContext>,
    #[cfg(feature = "stt-engine")]
    state: Option<whisper_rs::WhisperState>,
}

impl SttWorker {
    fn new(cfg: WorkerConfig) -> Self {
        Self {
            cfg,
            overlap_tail: Vec::new(),
            last_prompt: String::new(),
            #[cfg(feature = "stt-engine")]
            ctx: None,
            #[cfg(feature = "stt-engine")]
            state: None,
        }
    }

    fn reset_session(&mut self) {
        self.overlap_tail.clear();
        self.last_prompt.clear();
        debug!("stt worker: session reset");
    }

    fn unload(&mut self) {
        #[cfg(feature = "stt-engine")]
        {
            self.state = None;
            self.ctx = None;
        }
        self.cfg.loaded.store(false, Ordering::Release);
        self.reset_session();
    }

    fn ensure_loaded(&mut self) -> Result<()> {
        if self.cfg.loaded.load(Ordering::Acquire) {
            return Ok(());
        }
        self.load_model()
    }

    fn load_model(&mut self) -> Result<()> {
        let path = &self.cfg.model_path;

        if !path.exists() {
            match download_model(path) {
                Ok(true) => {
                    // Downloaded successfully — fall through to load.
                }
                Ok(false) => {
                    bail!(
                        "whisper model not found at {} — install base.en/small.en under \
                         ~/.local/share/sttd/models/ (e.g. ggml-base.en.bin)",
                        path.display()
                    );
                }
                Err(e) => {
                    bail!(
                        "whisper model not found at {} and auto-download failed: {e:#} \
                         — install base.en/small.en under ~/.local/share/sttd/models/ \
                         (e.g. ggml-base.en.bin)",
                        path.display()
                    );
                }
            }
        }

        #[cfg(feature = "stt-engine")]
        {
            info!(path = %path.display(), "stt worker: loading whisper model");

            // Keep whisper.cpp / ggml noise out of the daemon console when hooks work.
            whisper_rs::install_logging_hooks();

            let ctx = whisper_rs::WhisperContext::new_with_params(
                path,
                whisper_rs::WhisperContextParameters::default(),
            )
            .map_err(|e| {
                anyhow!(
                    "failed to load whisper model at {}: {e}",
                    path.display()
                )
            })?;

            let state = ctx
                .create_state()
                .map_err(|e| anyhow!("failed to create whisper state: {e}"))?;

            self.ctx = Some(ctx);
            self.state = Some(state);
            self.cfg.loaded.store(true, Ordering::Release);
            info!(path = %path.display(), "stt worker: model ready");
            Ok(())
        }

        #[cfg(not(feature = "stt-engine"))]
        {
            let _ = path;
            bail!(
                "sttd built without `stt-engine` feature (whisper-rs not linked); \
                 rebuild with default features and system cmake/clang"
            );
        }
    }

    /// Live path with optional short overlap from the previous chunk.
    fn transcribe_live(&mut self, samples: Vec<f32>, is_final: bool) -> Result<TranscriptResult> {
        if samples.is_empty() {
            return Ok(TranscriptResult {
                text: String::new(),
                is_final,
            });
        }

        self.ensure_loaded()?;

        let mut pcm = Vec::with_capacity(self.overlap_tail.len() + samples.len());
        pcm.extend_from_slice(&self.overlap_tail);
        pcm.extend_from_slice(&samples);

        let use_prompt = is_final && !self.last_prompt.is_empty();
        let text = self.run_whisper(&pcm, use_prompt)?;

        // Keep only the tail of *this* chunk so overlap size stays bounded.
        let n = self.cfg.overlap_samples.min(samples.len());
        if n > 0 {
            self.overlap_tail = samples[samples.len() - n..].to_vec();
        } else {
            self.overlap_tail.clear();
        }

        if is_final {
            if !text.is_empty() {
                // Prefer last ~200 chars as prompt context for the next chunk.
                self.last_prompt = truncate_prompt(&text, 200);
            }
            debug!(
                n_samples = samples.len(),
                n_with_overlap = pcm.len(),
                chars = text.len(),
                "stt worker: live final"
            );
        } else {
            debug!(
                n_samples = samples.len(),
                chars = text.len(),
                "stt worker: live preview"
            );
        }

        Ok(TranscriptResult { text, is_final })
    }

    /// Edit path: single full-buffer pass, no live overlap.
    fn transcribe_edit(&mut self, samples: Vec<f32>) -> Result<TranscriptResult> {
        if samples.is_empty() {
            return Ok(TranscriptResult {
                text: String::new(),
                is_final: true,
            });
        }

        self.ensure_loaded()?;
        let text = self.run_whisper(&samples, false)?;
        info!(
            n_samples = samples.len(),
            chars = text.len(),
            "stt worker: edit full buffer"
        );
        Ok(TranscriptResult {
            text,
            is_final: true,
        })
    }

    #[cfg(feature = "stt-engine")]
    fn run_whisper(&mut self, pcm: &[f32], use_prompt: bool) -> Result<String> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| anyhow!("whisper state missing after load"))?;

        let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy {
            best_of: 1,
        });

        params.set_n_threads(self.cfg.threads);
        // English-only models (base.en / small.en); language from config, default "en".
        let lang = self.cfg.language.as_str();
        if lang == "auto" {
            params.set_detect_language(true);
            params.set_language(None);
        } else {
            params.set_language(Some(lang));
            params.set_detect_language(false);
        }
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Independent VAD chunks; optional prompt for cross-chunk continuity.
        params.set_no_context(!use_prompt);
        params.set_single_segment(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        if use_prompt && !self.last_prompt.is_empty() {
            params.set_initial_prompt(&self.last_prompt);
        }

        state
            .full(params, pcm)
            .map_err(|e| anyhow!("whisper full() failed: {e}"))?;

        let mut out = String::new();
        for segment in state.as_iter() {
            let piece = segment
                .to_str()
                .map(|s| s.trim())
                .unwrap_or("");
            if piece.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(piece);
        }

        Ok(collapse_ws(&out))
    }

    #[cfg(not(feature = "stt-engine"))]
    fn run_whisper(&mut self, _pcm: &[f32], _use_prompt: bool) -> Result<String> {
        bail!(
            "sttd built without `stt-engine` feature (whisper-rs not linked); \
             rebuild with default features"
        );
    }
}

// --- helpers ------------------------------------------------------------------

fn ms_to_samples(ms: u32, sample_rate: u32) -> usize {
    (u64::from(ms) * u64::from(sample_rate) / 1000) as usize
}

fn resolve_threads(configured: u32) -> i32 {
    if configured > 0 {
        return configured as i32;
    }
    thread::available_parallelism()
        .map(|n| n.get().min(8) as i32)
        .unwrap_or(4)
}

fn truncate_prompt(text: &str, max_chars: usize) -> String {
    let t = text.trim();
    if t.chars().count() <= max_chars {
        return t.to_string();
    }
    // Keep the tail (most recent speech) as decoder prompt.
    let mut acc = String::new();
    for ch in t.chars().rev().take(max_chars) {
        acc.push(ch);
    }
    acc.chars().rev().collect()
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Known whisper.cpp model filenames available for auto-download.
const KNOWN_MODELS: &[&str] = &[
    "ggml-tiny.en.bin",
    "ggml-base.en.bin",
    "ggml-small.en.bin",
    "ggml-medium.en.bin",
];

/// Hugging Face repo for whisper.cpp GGML models.
const HF_REPO: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Attempt to download a whisper model from Hugging Face.
///
/// Returns `Ok(true)` on success, `Ok(false)` if the model filename is not in
/// the known list (no download attempted), and `Err` on network/IO failure.
fn download_model(path: &Path) -> Result<bool> {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    if !KNOWN_MODELS.contains(&filename) {
        return Ok(false);
    }

    let url = format!("{HF_REPO}/{filename}");
    info!(%url, "downloading whisper model");

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating model directory {}", parent.display()))?;
    }

    let response = ureq::get(&url)
        .call()
        .with_context(|| format!("HTTP request failed for {url}"))?;

    let total: u64 = response
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut reader = response.into_reader();
    let mut file = fs::File::create(path)
        .with_context(|| format!("creating model file {}", path.display()))?;
    let mut buf = [0u8; 65536];
    let mut downloaded: u64 = 0;
    let mut last_report: u64 = 0;

    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| "reading download stream")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .with_context(|| "writing model file")?;
        downloaded += n as u64;

        // Report progress every ~5 MiB.
        if downloaded - last_report >= 5_242_880 {
            if total > 0 {
                let pct = (downloaded * 100) / total;
                info!(pct, downloaded, total, "model download progress");
            } else {
                info!(downloaded, "model download progress (unknown total)");
            }
            last_report = downloaded;
        }
    }

    file.flush().with_context(|| "flushing model file")?;
    info!(downloaded, "model downloaded");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ms_to_samples_200ms() {
        assert_eq!(ms_to_samples(200, 16_000), 3_200);
    }

    #[test]
    fn truncate_prompt_keeps_tail() {
        let s = "a".repeat(50) + "TAIL";
        let t = truncate_prompt(&s, 4);
        assert_eq!(t, "TAIL");
    }

    #[test]
    fn collapse_whitespace() {
        assert_eq!(collapse_ws("  hello   world  "), "hello world");
    }
}
