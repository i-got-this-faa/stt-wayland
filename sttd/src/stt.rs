//! Speech-to-text engine (audio-cpp over audio.cpp).
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
//! audio.cpp GGUF ASR models (Qwen3 ASR, Parakeet TDT, Nemotron ASR, Citrinet)
//! under `~/.local/share/sttd/models/`, managed by `stt-settings` via the model
//! catalog. Residency: `preload` (default) or `lazy`.
//!
//! Final transcripts may additionally pass through the post-processing phase
//! ([`crate::postprocess`], e.g. S1-mini text normalization) inside this
//! worker, which keeps commit ordering and backpressure unchanged.
//!
//! # Feature flag
//!
//! Real inference requires the `stt-engine` feature (links `audio-cpp`).
//! Without it the worker still starts and the API is complete, but
//! load/transcribe return clear errors. Build with CMake and a C++ toolchain.

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

use crate::config::{ModelConfig, ModelResidency, PostProcessConfig};
use crate::postprocess::PostProcessor;

/// Speech engine + webrtc-vad capture rate (mono f32).
pub const SAMPLE_RATE_HZ: u32 = crate::config::PIPELINE_SAMPLE_RATE_HZ;

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
    /// Clear Live prompt state (e.g. between Stop → Start).
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
    /// Spawn the worker with default channel capacity.
    ///
    /// With `ModelResidency::Preload`, blocks until the model is loaded (or
    /// returns a clear error if the file is missing / unloadable).
    pub fn new(cfg: &ModelConfig, post_process: &PostProcessConfig) -> Result<Self> {
        Self::spawn(cfg, post_process, DEFAULT_CHANNEL_CAPACITY)
    }

    /// Spawn the worker thread.
    ///
    /// * `channel_capacity` — bound for command + result channels (backpressure).
    /// * `post_process` — optional text-normalization phase applied to final
    ///   transcripts inside the worker (keeps commit ordering intact).
    pub fn spawn(
        cfg: &ModelConfig,
        post_process: &PostProcessConfig,
        channel_capacity: usize,
    ) -> Result<Self> {
        let model_path = cfg.model_path();
        let capacity = channel_capacity.max(1);
        let threads = resolve_threads(cfg.threads);
        let language = cfg.language.clone();
        let residency = cfg.residency;
        let loaded = Arc::new(AtomicBool::new(false));
        let preview_cancel = Arc::new(AtomicBool::new(false));

        info!(
            path = %model_path.display(),
            residency = ?residency,
            capacity,
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
            preload: residency == ModelResidency::Preload,
            loaded: Arc::clone(&loaded),
            preview_cancel: Arc::clone(&preview_cancel),
            post: PostProcessor::new(post_process.clone()),
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

    /// Clear Live prompt continuity (call on Stop / mode switch).
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
    preload: bool,
    loaded: Arc<AtomicBool>,
    preview_cancel: Arc<AtomicBool>,
    /// Post-processing phase applied to final transcripts (none-op when off).
    post: PostProcessor,
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
    /// Engine handles. Dropping order matters: the session borrows the model in
    /// the C ABI, and the model borrows the registry. Fields drop in
    /// declaration order, so `session` is declared first.
    #[cfg(feature = "stt-engine")]
    session: Option<audio_cpp::Session>,
    #[cfg(feature = "stt-engine")]
    model: Option<audio_cpp::Model>,
    #[cfg(feature = "stt-engine")]
    registry: Option<audio_cpp::Registry>,
}

impl SttWorker {
    fn new(cfg: WorkerConfig) -> Self {
        Self {
            cfg,
            session_id: 0,
            #[cfg(feature = "stt-engine")]
            session: None,
            #[cfg(feature = "stt-engine")]
            model: None,
            #[cfg(feature = "stt-engine")]
            registry: None,
        }
    }

    fn clear_session(&mut self) {
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
            self.model = None;
            self.registry = None;
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
        let path = self.cfg.model_path.clone();

        if !path.exists() {
            bail!(
                "transcription model not found at {} — install a supported \
                 audio.cpp GGUF under ~/.local/share/sttd/models/ via stt-settings",
                path.display()
            );
        }

        #[cfg(feature = "stt-engine")]
        {
            info!(path = %path.display(), "stt worker: loading transcription model");

            let registry = audio_cpp::Registry::new()
                .map_err(|e| anyhow!("failed to create audio.cpp registry: {e}"))?;

            // GGUF weights cannot be auto-detected by audio.cpp; the family
            // must be derived from the managed filename.
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let family = model_family_for_filename(filename).ok_or_else(|| {
                anyhow!(
                    "cannot determine audio.cpp model family for {} — unsupported \
                     filename; supported prefixes: qwen3-asr, parakeet, nemotron, \
                     citrinet, hviske, fun-asr-nano",
                    path.display()
                )
            })?;

            let family_name = family.as_str().to_owned();
            let model = registry
                .load(&path.to_string_lossy(), Some(family), None)
                .map_err(|e| {
                    anyhow!(
                        "failed to load transcription model at {}: {e}",
                        path.display()
                    )
                })?;

            let session = model
                .create_task_session(
                    audio_cpp::TaskKind::Asr,
                    audio_cpp::RunMode::Offline,
                    audio_cpp::Backend::Cpu,
                    0, // device
                    self.cfg.threads,
                    None,
                )
                .map_err(|e| anyhow!("failed to create transcription session: {e}"))?;

            info!(
                path = %path.display(),
                family = %family_name,
                threads = self.cfg.threads,
                "stt worker: model ready"
            );

            self.registry = Some(registry);
            self.model = Some(model);
            self.session = Some(session);
            self.cfg.loaded.store(true, Ordering::Release);
            Ok(())
        }

        #[cfg(not(feature = "stt-engine"))]
        {
            let _ = path;
            bail!(
                "sttd built without `stt-engine` feature (audio-cpp not linked); \
                 rebuild with default features and a CMake/C++ toolchain"
            );
        }
    }

    /// Live path: transcribe each VAD-closed utterance independently.
    ///
    /// No audio is carried over from the previous chunk. VAD closes at a
    /// silence gap, so overlap is unnecessary; prepending a tail here would
    /// make the model emit those words twice because chunk transcripts are
    /// committed verbatim rather than text-stitched.
    fn transcribe_live(&mut self, samples: Vec<f32>, is_final: bool) -> Result<TranscriptResult> {
        if samples.is_empty() {
            return Ok(TranscriptResult {
                text: String::new(),
                is_final,
                session_id: self.session_id,
            });
        }

        self.ensure_loaded()?;

        let text = self.run_transcription(&samples)?;

        // Post-processing phase: normalize committed text only. Previews stay
        // raw so the overlay keeps its low latency.
        let text = if is_final {
            self.cfg.post.process(&text)
        } else {
            text
        };

        debug!(
            n_samples = samples.len(),
            is_final,
            chars = text.len(),
            "stt worker: live transcription"
        );

        Ok(TranscriptResult {
            text,
            is_final,
            session_id: self.session_id,
        })
    }

    /// Edit path: single full-buffer pass.
    fn transcribe_edit(&mut self, samples: Vec<f32>) -> Result<TranscriptResult> {
        if samples.is_empty() {
            return Ok(TranscriptResult {
                text: String::new(),
                is_final: true,
                session_id: self.session_id,
            });
        }

        self.ensure_loaded()?;
        let text = self.run_transcription(&samples)?;
        // Post-processing phase: the Edit buffer is committed in one pass.
        let text = self.cfg.post.process(&text);
        Ok(TranscriptResult {
            text,
            is_final: true,
            session_id: self.session_id,
        })
    }

    #[cfg(feature = "stt-engine")]
    fn run_transcription(&mut self, pcm: &[f32]) -> Result<String> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("transcription session missing after load"))?;

        // Inline mono 16 kHz PCM buffer — no temp WAV file round-trip.
        let buffer = audio_cpp::WavAudio {
            sample_rate: SAMPLE_RATE_HZ as i32,
            channels: 1,
            samples: pcm.to_vec(),
        };
        let mut request = audio_cpp::Request::asr(audio_cpp::AudioInput::Buffer(buffer));
        let language = self.cfg.language.trim();
        if !language.is_empty() && language != "auto" {
            request = request.option("language", language);
        }

        let result = session
            .run_offline(request)
            .map_err(|e| anyhow!("transcription run failed: {e}"))?;
        let text = result.text_output.map(|t| t.text).unwrap_or_default();
        Ok(collapse_ws(&text))
    }

    #[cfg(not(feature = "stt-engine"))]
    fn run_transcription(&mut self, _pcm: &[f32]) -> Result<String> {
        bail!(
            "sttd built without `stt-engine` feature (audio-cpp not linked); \
             rebuild with default features"
        );
    }
}

// --- helpers ------------------------------------------------------------------

fn resolve_threads(configured: u32) -> i32 {
    if configured > 0 {
        return configured as i32;
    }
    thread::available_parallelism()
        .map(|n| n.get().min(8) as i32)
        .unwrap_or(4)
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Map a managed model filename to the audio.cpp model family that can load
/// it. GGUF weights cannot be auto-detected by the engine, so the family hint
/// is required and derived from the filename prefix assigned by the catalog.
#[cfg(feature = "stt-engine")]
fn model_family_for_filename(filename: &str) -> Option<audio_cpp::ModelFamily> {
    use audio_cpp::ModelFamily;
    let lower = filename.to_ascii_lowercase();
    if lower.starts_with("qwen3-asr") {
        Some(ModelFamily::Qwen3Asr)
    } else if lower.starts_with("parakeet") {
        Some(ModelFamily::ParakeetTdt)
    } else if lower.starts_with("nemotron") {
        Some(ModelFamily::NemotronAsr)
    } else if lower.starts_with("citrinet") {
        Some(ModelFamily::CitrinetAsr)
    } else if lower.starts_with("hviske") {
        Some(ModelFamily::HviskeAsr)
    } else if lower.starts_with("fun-asr-nano") {
        Some(ModelFamily::FunAsrNano)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_whitespace() {
        assert_eq!(collapse_ws("  hello   world  "), "hello world");
    }

    #[cfg(feature = "stt-engine")]
    #[test]
    fn family_mapping_covers_catalog_prefixes() {
        use audio_cpp::ModelFamily;
        assert_eq!(
            model_family_for_filename("qwen3-asr-0.6b-q8_0.gguf"),
            Some(ModelFamily::Qwen3Asr)
        );
        assert_eq!(
            model_family_for_filename("parakeet-tdt-0.6b-v3-q8_0.gguf"),
            Some(ModelFamily::ParakeetTdt)
        );
        assert_eq!(
            model_family_for_filename("nemotron-3.5-asr-q8_0.gguf"),
            Some(ModelFamily::NemotronAsr)
        );
        assert_eq!(
            model_family_for_filename("citrinet-asr-q8_0.gguf"),
            Some(ModelFamily::CitrinetAsr)
        );
        assert_eq!(model_family_for_filename("ggml-base.en.bin"), None);
    }
}
