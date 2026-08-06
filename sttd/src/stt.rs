//! Speech-to-text engine (transcribe-cpp over transcribe.cpp).
//!
//! # Architecture
//!
//! A **dedicated OS thread** owns the model session. Callers talk to it through
//! bounded `mpsc` channels:
//!
//! - **In**: bounded final/control [`SttCommand`] jobs plus a coalesced preview
//!   channel
//! - **Out**: [`TranscriptResult`] `{ text, is_final }` for live/preview path
//!
//! VAD-closed chunks are expected on a bounded channel (backpressure when the
//! worker is slower than speech). Edit mode uses a request/reply oneshot-style
//! channel so the full buffer can be transcribed in one shot.
//!
//! # Models
//!
//! transcribe.cpp GGUF models and legacy whisper.cpp GGML `.bin` models under
//! `~/.local/share/sttd/models/`. Residency: `preload` (default) or `lazy`.
//!
//! # Feature flag
//!
//! Real inference requires the `stt-engine` feature (links `transcribe-cpp`).
//! Without it the worker still starts and the API is complete, but
//! load/transcribe return clear errors. Build with CMake and a C++ toolchain.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{
    self, Receiver, RecvError, SendError, SyncSender, TryRecvError, TrySendError,
};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use tracing::{debug, error, info, warn};

use crate::config::{ModelConfig, ModelResidency};

// transcribe-cpp-sys 0.1.3 omits its Linux CBLAS dependency from the link manifest.
#[cfg(all(feature = "stt-engine", target_os = "linux"))]
#[link(name = "cblas")]
extern "C" {}

/// Speech engine + webrtc-vad capture rate (mono f32).
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
    /// Worker session that produced this result. Results from an older
    /// session must not be injected into a newly started recording.
    pub(crate) session_id: u64,
}

/// Jobs accepted by the STT worker thread.
#[derive(Debug)]
pub enum SttCommand {
    /// Live mode: VAD-closed speech chunk → committed text (`is_final = true`).
    LiveChunk(Vec<f32>),
    /// Edit mode: transcribe the entire buffer once; reply on `reply`.
    EditFull {
        samples: Vec<f32>,
        reply: mpsc::Sender<Result<TranscriptResult>>,
    },
    /// Load the model if not yet resident (lazy residency / explicit warm-up).
    EnsureLoaded { reply: mpsc::Sender<Result<()>> },
    /// Reply after every command queued before this barrier has completed.
    Barrier { reply: mpsc::Sender<()> },
    /// Clear Live overlap tail + prompt state (e.g. between Stop → Start).
    ResetSession { session_id: u64 },
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
    preview_tx: SyncSender<Vec<f32>>,
    result_rx: Receiver<TranscriptResult>,
    join: Option<JoinHandle<()>>,
    model_path: PathBuf,
    residency: ModelResidency,
    loaded: Arc<AtomicBool>,
    preview_cancel: Arc<AtomicBool>,
    session_id: AtomicU64,
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
        let preview_cancel = Arc::new(AtomicBool::new(false));

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
        // Preview work is lossy and must never occupy slots needed by final
        // Live chunks or control barriers. A one-entry channel also coalesces
        // previews naturally while the model is busy.
        let (preview_tx, preview_rx) = mpsc::sync_channel::<Vec<f32>>(1);
        // Results are small and the core loop drains them continuously. Keep
        // this channel unbounded so a stop barrier cannot deadlock behind a
        // full result queue while the core thread waits for final text.
        let (result_tx, result_rx) = mpsc::channel::<TranscriptResult>();

        let worker_cfg = WorkerConfig {
            model_path: model_path.clone(),
            residency,
            threads,
            language,
            overlap_samples,
            preload: residency == ModelResidency::Preload,
            loaded: Arc::clone(&loaded),
            preview_cancel: Arc::clone(&preview_cancel),
        };

        let join = thread::Builder::new()
            .name("sttd-transcribe".into())
            .spawn(move || worker_main(worker_cfg, cmd_rx, preview_rx, result_tx))
            .context("spawn stt worker thread")?;

        let engine = Self {
            cmd_tx,
            preview_tx,
            result_rx,
            join: Some(join),
            model_path,
            residency,
            loaded,
            preview_cancel,
            session_id: AtomicU64::new(0),
        };

        if residency == ModelResidency::Preload {
            engine
                .ensure_loaded()
                .context("preload transcription model at startup")?;
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
            .map_err(|_| anyhow!("timed out waiting for transcription model load"))?
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
        self.preview_tx
            .send(samples)
            .map_err(|_| anyhow!("stt worker thread is not running"))
    }

    pub fn try_push_preview(&self, samples: Vec<f32>) -> Result<(), TrySendError<Vec<f32>>> {
        self.preview_tx.try_send(samples)
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
    pub fn reset_session(&self) -> Result<u64> {
        let session_id = self
            .session_id
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        self.cmd_tx
            .send(SttCommand::ResetSession { session_id })
            .map_err(|_| anyhow!("stt worker thread is not running"))?;
        Ok(session_id)
    }

    pub fn session_id(&self) -> u64 {
        self.session_id.load(Ordering::Acquire)
    }

    /// Cancel queued/in-flight preview work at the next worker boundary.
    ///
    /// Preview jobs are deliberately lossy; final Live chunks must not wait
    /// behind them when the user stops recording.
    pub fn cancel_previews(&self) {
        self.preview_cancel.store(true, Ordering::Release);
    }

    /// Allow preview jobs for a new/continued Live session.
    pub fn resume_previews(&self) {
        self.preview_cancel.store(false, Ordering::Release);
    }

    /// Wait until all commands queued before this call have completed.
    pub fn wait_idle(&self, timeout: Duration) -> Result<()> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.cmd_tx
            .send(SttCommand::Barrier { reply: reply_tx })
            .map_err(|_| anyhow!("stt worker thread is not running"))?;
        reply_rx
            .recv_timeout(timeout)
            .map_err(|_| anyhow!("timed out waiting for stt worker to become idle"))
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

    /// Clone the lossy preview sender for the VAD/pipeline thread.
    pub fn preview_sender(&self) -> SyncSender<Vec<f32>> {
        self.preview_tx.clone()
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
    preview_cancel: Arc<AtomicBool>,
}

fn worker_main(
    cfg: WorkerConfig,
    cmd_rx: Receiver<SttCommand>,
    preview_rx: Receiver<Vec<f32>>,
    result_tx: mpsc::Sender<TranscriptResult>,
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

    loop {
        // Final chunks and control commands always win over previews.
        match cmd_rx.try_recv() {
            Ok(SttCommand::Barrier { reply }) => {
                if worker.cfg.preview_cancel.load(Ordering::Acquire) {
                    while preview_rx.try_recv().is_ok() {}
                }
                let _ = reply.send(());
                continue;
            }
            Ok(cmd) => {
                if !process_command(&mut worker, cmd, &result_tx) {
                    break;
                }
                continue;
            }
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {}
        }

        if !worker.cfg.preview_cancel.load(Ordering::Acquire) {
            match preview_rx.try_recv() {
                Ok(samples) => {
                    if !process_preview(&mut worker, samples, &result_tx) {
                        break;
                    }
                    continue;
                }
                Err(TryRecvError::Disconnected) | Err(TryRecvError::Empty) => {}
            }
        } else {
            while preview_rx.try_recv().is_ok() {}
        }

        match cmd_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(SttCommand::Barrier { reply }) => {
                if worker.cfg.preview_cancel.load(Ordering::Acquire) {
                    while preview_rx.try_recv().is_ok() {}
                }
                let _ = reply.send(());
            }
            Ok(cmd) => {
                if !process_command(&mut worker, cmd, &result_tx) {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }

    worker.unload();
    info!("stt worker: exited");
}

/// Process a final/control command. Returns false when the worker should exit.
fn process_command(
    worker: &mut SttWorker,
    cmd: SttCommand,
    result_tx: &mpsc::Sender<TranscriptResult>,
) -> bool {
    match cmd {
        SttCommand::Shutdown => {
            debug!("stt worker: shutdown");
            false
        }
        SttCommand::ResetSession { session_id } => {
            worker.reset_session(session_id);
            true
        }
        SttCommand::EnsureLoaded { reply } => {
            let res = worker.ensure_loaded();
            let _ = reply.send(res);
            true
        }
        SttCommand::Barrier { reply } => {
            let _ = reply.send(());
            true
        }
        SttCommand::LiveChunk(samples) => {
            match worker.transcribe_live(samples, true) {
                Ok(result) => {
                    if send_result(result_tx, result).is_err() {
                        warn!("stt worker: result consumer gone; exiting");
                        return false;
                    }
                }
                Err(err) => {
                    error!(error = %err, "stt worker: live chunk failed");
                }
            }
            true
        }
        SttCommand::EditFull { samples, reply } => {
            let res = worker.transcribe_edit(samples);
            let _ = reply.send(res);
            true
        }
    }
}

fn process_preview(
    worker: &mut SttWorker,
    samples: Vec<f32>,
    result_tx: &mpsc::Sender<TranscriptResult>,
) -> bool {
    if worker.cfg.preview_cancel.load(Ordering::Acquire) {
        return true;
    }

    match worker.transcribe_live(samples, false) {
        Ok(result) => {
            if send_result(result_tx, result).is_err() {
                warn!("stt worker: result consumer gone; exiting");
                false
            } else {
                true
            }
        }
        Err(err) => {
            error!(error = %err, "stt worker: live preview failed");
            true
        }
    }
}

fn send_result(
    tx: &mpsc::Sender<TranscriptResult>,
    result: TranscriptResult,
) -> Result<(), SendError<TranscriptResult>> {
    tx.send(result)
}

/// Worker-local state; lives only on the `sttd-transcribe` thread.
struct SttWorker {
    cfg: WorkerConfig,
    session_id: u64,
    /// Tail of the last Live chunk, prepended to the next for continuity.
    overlap_tail: Vec<f32>,
    /// Last committed Live text (Whisper initial prompt when supported).
    last_prompt: String,
    #[cfg(feature = "stt-engine")]
    session: Option<transcribe_cpp::Session>,
    #[cfg(feature = "stt-engine")]
    model_arch: Option<String>,
}

impl SttWorker {
    fn new(cfg: WorkerConfig) -> Self {
        Self {
            cfg,
            session_id: 0,
            overlap_tail: Vec::new(),
            last_prompt: String::new(),
            #[cfg(feature = "stt-engine")]
            session: None,
            #[cfg(feature = "stt-engine")]
            model_arch: None,
        }
    }

    fn clear_session(&mut self) {
        self.overlap_tail.clear();
        self.last_prompt.clear();
        debug!("stt worker: session reset");
    }

    fn reset_session(&mut self, session_id: u64) {
        self.session_id = session_id;
        self.clear_session();
    }

    fn unload(&mut self) {
        #[cfg(feature = "stt-engine")]
        {
            self.session = None;
            self.model_arch = None;
        }
        self.cfg.loaded.store(false, Ordering::Release);
        self.clear_session();
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
                        "transcription model not found at {} — install a supported GGUF or \
                         legacy whisper.cpp .bin model under ~/.local/share/sttd/models/",
                        path.display()
                    );
                }
                Err(e) => {
                    bail!(
                        "transcription model not found at {} and legacy Whisper \
                         auto-download failed: {e:#} — install a supported GGUF or \
                         whisper.cpp .bin model under ~/.local/share/sttd/models/",
                        path.display()
                    );
                }
            }
        }

        #[cfg(feature = "stt-engine")]
        {
            info!(path = %path.display(), "stt worker: loading transcription model");

            disable_engine_logging_once();

            let model = transcribe_cpp::Model::load(path).map_err(|e| {
                anyhow!(
                    "failed to load transcription model at {}: {e}",
                    path.display()
                )
            })?;
            let arch = model.arch();
            let variant = model.variant();
            let backend = model.backend();
            let capabilities = model.capabilities();
            if capabilities.native_sample_rate != SAMPLE_RATE_HZ as i32 {
                bail!(
                    "model at {} requires {} Hz audio; sttd currently supplies {} Hz",
                    path.display(),
                    capabilities.native_sample_rate,
                    SAMPLE_RATE_HZ
                );
            }

            let session = model
                .session_with(&transcribe_cpp::SessionOptions {
                    n_threads: self.cfg.threads,
                    ..Default::default()
                })
                .map_err(|e| anyhow!("failed to create transcription session: {e}"))?;

            info!(
                path = %path.display(),
                arch = %arch,
                variant = %variant,
                backend = %backend,
                supports_streaming = capabilities.supports_streaming,
                "stt worker: model ready"
            );

            self.model_arch = Some(arch);
            self.session = Some(session);
            self.cfg.loaded.store(true, Ordering::Release);
            Ok(())
        }

        #[cfg(not(feature = "stt-engine"))]
        {
            let _ = path;
            bail!(
                "sttd built without `stt-engine` feature (transcribe-cpp not linked); \
                 rebuild with default features and a CMake/C++ toolchain"
            );
        }
    }

    /// Live path with optional short overlap from the previous chunk.
    fn transcribe_live(&mut self, samples: Vec<f32>, is_final: bool) -> Result<TranscriptResult> {
        if samples.is_empty() {
            return Ok(TranscriptResult {
                text: String::new(),
                is_final,
                session_id: self.session_id,
            });
        }

        self.ensure_loaded()?;

        let mut pcm = Vec::with_capacity(self.overlap_tail.len() + samples.len());
        pcm.extend_from_slice(&self.overlap_tail);
        pcm.extend_from_slice(&samples);

        let use_prompt = is_final && !self.last_prompt.is_empty();
        let text = self.run_transcription(&pcm, use_prompt)?;

        if is_final {
            // Keep only the tail of the committed chunk. Preview jobs are
            // snapshots of the current utterance; updating the overlap from
            // them would duplicate the preview tail in the eventual final
            // chunk and make repeated previews affect decoder continuity.
            let n = self.cfg.overlap_samples.min(samples.len());
            if n > 0 {
                self.overlap_tail = samples[samples.len() - n..].to_vec();
            } else {
                self.overlap_tail.clear();
            }

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

        Ok(TranscriptResult {
            text,
            is_final,
            session_id: self.session_id,
        })
    }

    /// Edit path: single full-buffer pass, no live overlap.
    fn transcribe_edit(&mut self, samples: Vec<f32>) -> Result<TranscriptResult> {
        if samples.is_empty() {
            return Ok(TranscriptResult {
                text: String::new(),
                is_final: true,
                session_id: self.session_id,
            });
        }

        self.ensure_loaded()?;
        let text = self.run_transcription(&samples, false)?;
        info!(
            n_samples = samples.len(),
            chars = text.len(),
            "stt worker: edit full buffer"
        );
        Ok(TranscriptResult {
            text,
            is_final: true,
            session_id: self.session_id,
        })
    }

    #[cfg(feature = "stt-engine")]
    fn run_transcription(&mut self, pcm: &[f32], use_prompt: bool) -> Result<String> {
        let language = self.cfg.language.trim();
        let mut options = transcribe_cpp::RunOptions {
            timestamps: transcribe_cpp::TimestampKind::None,
            language: (!language.is_empty() && language != "auto").then(|| language.to_string()),
            ..Default::default()
        };

        // Prompt continuity and no-speech thresholds are Whisper-specific.
        // Other model families use their own shipped decoding defaults.
        if self.model_arch.as_deref() == Some("whisper") {
            options.family = Some(transcribe_cpp::RunExtension::Whisper(
                transcribe_cpp::WhisperRunOptions {
                    initial_prompt: use_prompt.then(|| self.last_prompt.clone()),
                    condition_on_prev_tokens: Some(use_prompt),
                    temperature: Some(0.0),
                    temperature_inc: Some(0.0),
                    no_speech_thold: Some(MAX_NO_SPEECH_PROBABILITY),
                    ..Default::default()
                },
            ));
        }

        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("transcription session missing after load"))?;
        let limits = session
            .limits()
            .map_err(|e| anyhow!("reading transcription session limits failed: {e}"))?;
        if limits.effective_max_audio_ms > 0 {
            let max_samples = (limits.effective_max_audio_ms as u64)
                .saturating_mul(u64::from(SAMPLE_RATE_HZ))
                / 1000;
            if pcm.len() as u64 > max_samples {
                bail!(
                    "audio buffer is too long for this model ({} ms > {} ms)",
                    pcm.len() as u64 * 1000 / u64::from(SAMPLE_RATE_HZ),
                    limits.effective_max_audio_ms
                );
            }
        }
        let transcript = session
            .run(pcm, &options)
            .map_err(|e| anyhow!("transcription run failed: {e}"))?;

        Ok(collapse_ws(&transcript.text))
    }

    #[cfg(not(feature = "stt-engine"))]
    fn run_transcription(&mut self, _pcm: &[f32], _use_prompt: bool) -> Result<String> {
        bail!(
            "sttd built without `stt-engine` feature (transcribe-cpp not linked); \
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

/// Whisper no-speech threshold applied by the family-specific decoder.
const MAX_NO_SPEECH_PROBABILITY: f32 = 0.6;

#[cfg(feature = "stt-engine")]
fn disable_engine_logging_once() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(transcribe_cpp::disable_logging);
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
    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

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
            if let Some(pct) = (downloaded * 100).checked_div(total) {
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
