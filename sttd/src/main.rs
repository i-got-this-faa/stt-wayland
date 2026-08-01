//! `sttd` — Wayland speech-to-text daemon.
//!
//! Stack: PipeWire capture → webrtc-vad → whisper-rs → gtk4 overlay +
//! virtual-keyboard injection. Control plane: Unix socket NDJSON (`proto`).
//!
//! # Runtime layout
//!
//! | Thread            | Role                                              |
//! |-------------------|---------------------------------------------------|
//! | main (tokio)      | Core event loop: IPC, state machine, STT results  |
//! | sttd-pipewire     | PipeWire stream (open on Start, close on Stop)    |
//! | sttd-pipeline     | VAD chunking → STT command enqueue                |
//! | sttd-whisper      | whisper-rs worker                                 |
//! | sttd-overlay      | GTK4 + layer-shell main loop (warm when idle)     |
//!
//! Overlay and IPC share one state-transition path via [`CoreEvent`].

// Subsystem modules expose broader public APIs than the daemon currently calls.
#![allow(dead_code)]

mod audio;
mod config;
mod inject;
mod overlay;
mod state;
mod stt;
mod vad;

use std::os::unix::fs::FileTypeExt;
use std::os::unix::net::UnixStream as StdUnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use proto::{DaemonState, Mode, Request, Response};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

use crate::audio::AudioCapture;
use crate::config::{Config, ModelResidency};
use crate::inject::Injector;
use crate::overlay::{ChannelControlSink, ControlEvent, OverlayHandle};
use crate::state::{AppState, SetModeResult, StartResult, StopResult};
use crate::stt::{SttCommand, SttEngine, TranscriptResult};
use crate::vad::{Chunker, SpeechPending, DEFAULT_FRAME_MS, SAMPLE_RATE_HZ};

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(
    name = "sttd",
    about = "Wayland speech-to-text daemon (PipeWire + whisper.cpp)",
    version
)]
struct Cli {
    /// Run as a long-lived daemon (default when no subcommand flags given).
    #[arg(long, default_value_t = true)]
    daemon: bool,

    /// Path to config.toml (default: ~/.config/sttd/config.toml).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Override Unix socket path (default: $XDG_RUNTIME_DIR/sttd.sock).
    #[arg(long)]
    socket: Option<PathBuf>,

    /// Print resolved config and exit.
    #[arg(long)]
    print_config: bool,
}

// ─── Core control plane ─────────────────────────────────────────────────────

/// Events funneled into the single state-transition path.
enum CoreEvent {
    /// IPC request; reply is sent when the transition completes.
    Ipc {
        req: Request,
        reply: oneshot::Sender<Response>,
    },
    /// Overlay UI callbacks (pause / stop / copy / discard).
    Overlay(ControlEvent),
    /// Pipeline → core notifications.
    Pipe(PipeEvent),
    /// Graceful shutdown (signal).
    Shutdown,
}

/// Pipeline → core.
enum PipeEvent {
    SpeechPending(bool),
    /// Edit session hit max duration; daemon should stop into EditReview.
    EditCapReached,
    Error(String),
}

/// Core → VAD/STT pipeline thread.
enum PipeCmd {
    /// Begin feeding VAD for a new session.
    Arm {
        mode: Mode,
    },
    /// Disarm + flush open utterance; for Edit, return full session buffer.
    Flush {
        reply: Sender<Option<Vec<f32>>>,
    },
    SetMode(Mode),
    Reset,
    Shutdown,
}

/// PipeWire emits a short connection transient when the capture stream is
/// first attached. Do not let that transient reach VAD/Whisper as speech.
const CAPTURE_STARTUP_IGNORE_MS: u32 = 750;

// ─── entry ──────────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("sttd: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    let cfg = Config::load(cli.config.as_deref()).context("load config")?;

    if cli.print_config {
        println!("{}", toml::to_string_pretty(&cfg)?);
        return Ok(());
    }

    let socket = cli.socket.unwrap_or_else(proto::default_socket_path);

    if std::env::var_os("XDG_RUNTIME_DIR").is_none() {
        warn!("XDG_RUNTIME_DIR is unset; socket path may be wrong for a session daemon");
    }

    if !cli.daemon {
        info!("--daemon=false; nothing to do");
        return Ok(());
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("sttd-tokio")
        .build()
        .context("build tokio runtime")?;

    rt.block_on(async_main(cfg, socket))
}

async fn async_main(cfg: Config, socket: PathBuf) -> Result<()> {
    info!(
        socket = %socket.display(),
        model = %cfg.model.name,
        residency = ?cfg.model.residency,
        "starting sttd"
    );

    let mut daemon = Daemon::bootstrap(cfg).context("bootstrap daemon")?;

    prepare_socket_path(&socket)?;

    let listener = UnixListener::bind(&socket)
        .with_context(|| format!("bind Unix socket {}", socket.display()))?;
    info!(path = %socket.display(), "listening");

    let (core_tx, core_rx) = mpsc::channel::<CoreEvent>();

    // Overlay control → same core path as IPC.
    let (overlay_sink, overlay_rx) = ChannelControlSink::pair();
    daemon.overlay.set_control_sink(Arc::new(overlay_sink));

    {
        let tx = core_tx.clone();
        thread::Builder::new()
            .name("sttd-overlay-fwd".into())
            .spawn(move || {
                while let Ok(ev) = overlay_rx.recv() {
                    if tx.send(CoreEvent::Overlay(ev)).is_err() {
                        break;
                    }
                }
            })
            .context("spawn overlay event forwarder")?;
    }

    // Pipeline events already land on daemon.pipe_ev_rx; bridge into core_tx.
    {
        let tx = core_tx.clone();
        let pipe_rx = daemon
            .pipe_ev_rx
            .take()
            .ok_or_else(|| anyhow!("pipeline event receiver missing"))?;
        thread::Builder::new()
            .name("sttd-pipe-fwd".into())
            .spawn(move || {
                while let Ok(ev) = pipe_rx.recv() {
                    if tx.send(CoreEvent::Pipe(ev)).is_err() {
                        break;
                    }
                }
            })
            .context("spawn pipeline event forwarder")?;
    }

    {
        let tx = core_tx.clone();
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        let tx = tx.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_client(stream, tx).await {
                                debug!(error = %e, "client session error");
                            }
                        });
                    }
                    Err(e) => {
                        error!(error = %e, "accept failed");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                }
            }
        });
    }

    {
        let tx = core_tx.clone();
        tokio::spawn(async move {
            shutdown_signal().await;
            info!("shutdown signal received");
            let _ = tx.send(CoreEvent::Shutdown);
        });
    }

    let socket_for_cleanup = socket.clone();
    let result = tokio::task::spawn_blocking(move || daemon.run_core_loop(core_rx)).await;

    let _ = std::fs::remove_file(&socket_for_cleanup);

    match result {
        Ok(Ok(())) => {
            info!("sttd stopped");
            Ok(())
        }
        Ok(Err(e)) => Err(e),
        Err(e) => Err(anyhow!("core task join: {e}")),
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
        tokio::select! {
            _ = sigterm.recv() => {}
            _ = sigint.recv() => {}
            _ = tokio::signal::ctrl_c() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Prepare a Unix socket path without stealing an active daemon's endpoint.
///
/// A stale socket is safe to remove after a timed connect refuses the
/// connection. Regular files and active sockets are never overwritten.
fn prepare_socket_path(socket: &Path) -> Result<()> {
    if let Ok(metadata) = std::fs::symlink_metadata(socket) {
        if !metadata.file_type().is_socket() {
            bail!(
                "socket path {} exists and is not a Unix socket",
                socket.display()
            );
        }

        match StdUnixStream::connect(socket) {
            Ok(_) => bail!(
                "another sttd instance is already listening at {}",
                socket.display()
            ),
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::NotFound
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::ConnectionAborted
                ) =>
            {
                std::fs::remove_file(socket)
                    .with_context(|| format!("remove stale socket {}", socket.display()))?;
                debug!(path = %socket.display(), "removed stale socket");
            }
            Err(err) => {
                bail!("cannot inspect existing socket {}: {err}", socket.display());
            }
        }
    }

    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create socket dir {}", parent.display()))?;
    }
    Ok(())
}

async fn handle_client(stream: UnixStream, core_tx: Sender<CoreEvent>) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await.context("read request line")? {
        if line.trim().is_empty() {
            continue;
        }
        let req = match proto::parse_request_line(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::error(format!("invalid request: {e}"));
                writer
                    .write_all(resp.to_line()?.as_bytes())
                    .await
                    .context("write parse error")?;
                continue;
            }
        };

        let (reply_tx, reply_rx) = oneshot::channel();
        if core_tx
            .send(CoreEvent::Ipc {
                req,
                reply: reply_tx,
            })
            .is_err()
        {
            let resp = Response::error("daemon core is shutting down");
            let _ = writer.write_all(resp.to_line()?.as_bytes()).await;
            break;
        }

        let resp = match reply_rx.await {
            Ok(r) => r,
            Err(_) => Response::error("daemon core dropped reply"),
        };
        writer
            .write_all(resp.to_line()?.as_bytes())
            .await
            .context("write response")?;
    }
    Ok(())
}

// ─── Daemon ─────────────────────────────────────────────────────────────────

struct Daemon {
    cfg: Config,
    state: AppState,
    audio: AudioCapture,
    stt: SttEngine,
    inject: Injector,
    overlay: OverlayHandle,

    pipe_cmd_tx: Sender<PipeCmd>,
    /// Taken once at startup to bridge into CoreEvent.
    pipe_ev_rx: Option<Receiver<PipeEvent>>,
    _pipe_join: JoinHandle<()>,

    last_speech_pending: bool,
    /// STT result generation accepted by the active recording session.
    stt_session_id: u64,
}

impl Daemon {
    fn bootstrap(cfg: Config) -> Result<Self> {
        let inject = Injector::new(&cfg.injection).context("init injector")?;
        let mut state = AppState::new(cfg.model.clone());
        state.inject_available = inject.is_available();
        if cfg.injection.enabled && !inject.is_available() {
            let msg = inject
                .last_error()
                .unwrap_or(Injector::unavailable_message())
                .to_string();
            state.set_error(msg);
            warn!("virtual-keyboard unavailable — injection disabled; capture still works");
        }

        let stt = match SttEngine::spawn(
            &cfg.model,
            cfg.audio.stt_channel_capacity,
            stt::DEFAULT_OVERLAP_MS,
        ) {
            Ok(e) => e,
            Err(e) if cfg.model.residency == ModelResidency::Preload => {
                warn!(error = %e, "model preload failed; continuing with lazy residency");
                state.set_error(format!("model preload failed: {e:#}"));
                let mut lazy = cfg.model.clone();
                lazy.residency = ModelResidency::Lazy;
                SttEngine::spawn(
                    &lazy,
                    cfg.audio.stt_channel_capacity,
                    stt::DEFAULT_OVERLAP_MS,
                )
                .context("spawn STT worker (lazy fallback)")?
            }
            Err(e) => return Err(e).context("spawn STT worker"),
        };

        if stt.is_loaded() {
            info!(path = %stt.model_path().display(), "whisper model loaded");
            if state
                .last_error
                .as_deref()
                .is_some_and(|m| m.starts_with("model preload"))
            {
                state.clear_error();
            }
        } else {
            info!(
                path = %stt.model_path().display(),
                "whisper model deferred (lazy residency)"
            );
        }

        let mut audio = AudioCapture::new(&cfg.audio).context("init audio capture")?;
        let sample_rx = audio
            .take_sample_receiver()
            .ok_or_else(|| anyhow!("audio sample receiver missing"))?;

        // webrtc-vad's `Vad` is !Send — construct the Chunker on the pipeline thread.
        let silence_gap_ms = cfg.audio.silence_gap_ms;
        let max_edit_samples =
            (cfg.audio.max_edit_seconds as usize).saturating_mul(SAMPLE_RATE_HZ as usize);

        let (pipe_cmd_tx, pipe_cmd_rx) = mpsc::channel::<PipeCmd>();
        let (pipe_ev_tx, pipe_ev_rx) = mpsc::channel::<PipeEvent>();
        let stt_cmd = stt.command_sender();
        let stt_preview = stt.preview_sender();

        let pipe_join = thread::Builder::new()
            .name("sttd-pipeline".into())
            .spawn(move || {
                pipeline_main(
                    sample_rx,
                    silence_gap_ms,
                    stt_cmd,
                    stt_preview,
                    pipe_cmd_rx,
                    pipe_ev_tx,
                    max_edit_samples,
                );
            })
            .context("spawn pipeline thread")?;

        let overlay = OverlayHandle::new(&cfg.overlay).context("init overlay")?;
        let stt_session_id = stt.session_id();

        Ok(Self {
            cfg,
            state,
            audio,
            stt,
            inject,
            overlay,
            pipe_cmd_tx,
            pipe_ev_rx: Some(pipe_ev_rx),
            _pipe_join: pipe_join,
            last_speech_pending: false,
            stt_session_id,
        })
    }

    /// Idle: no PW stream, no mapped overlay. Blocks until Shutdown.
    fn run_core_loop(&mut self, core_rx: Receiver<CoreEvent>) -> Result<()> {
        info!("core loop running (idle)");
        let poll = Duration::from_millis(10);

        loop {
            // Drain STT transcripts (Live commit / preview).
            self.drain_stt_results();

            match core_rx.recv_timeout(poll) {
                Ok(CoreEvent::Shutdown) => {
                    self.shutdown_subsystems();
                    return Ok(());
                }
                Ok(CoreEvent::Ipc { req, reply }) => {
                    let resp = self.handle_request(req);
                    let _ = reply.send(resp);
                }
                Ok(CoreEvent::Overlay(ev)) => {
                    if let Err(e) = self.handle_overlay(ev) {
                        warn!(error = %e, "overlay control failed");
                    }
                }
                Ok(CoreEvent::Pipe(ev)) => {
                    self.handle_pipe_event(ev);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    self.shutdown_subsystems();
                    return Ok(());
                }
            }
        }
    }

    fn drain_stt_results(&mut self) {
        loop {
            match self.stt.try_recv_result() {
                Ok(Some(result)) => self.on_transcript(result),
                Ok(None) => break,
                Err(_) => {
                    warn!("STT result channel disconnected");
                    break;
                }
            }
        }
    }

    fn on_transcript(&mut self, result: TranscriptResult) {
        if result.session_id != self.stt_session_id {
            debug!(
                result_session = result.session_id,
                active_session = self.stt_session_id,
                "ignoring stale STT result"
            );
            return;
        }

        let text = result.text.trim();
        if text.is_empty() {
            return;
        }

        // Only apply Live path while capturing in Live mode.
        if self.state.mode != Mode::Live {
            return;
        }
        if self.state.state != DaemonState::Recording {
            return;
        }

        if result.is_final {
            debug!(n = text.len(), "live commit");
            self.overlay.append_committed(text);
            self.overlay.set_preview("");
            if self.inject.is_enabled() {
                // Trailing space between chunks for natural injection.
                let mut batch = text.to_string();
                if !batch.ends_with(char::is_whitespace) {
                    batch.push(' ');
                }
                if let Err(e) = self.inject.inject_text(&batch) {
                    warn!(error = %e, "inject failed");
                    self.state.set_error(format!("inject: {e:#}"));
                }
            }
        } else {
            // In-progress preview — overlay only (~200 ms crossfade in UI).
            self.overlay.set_preview(text);
        }
    }

    fn handle_pipe_event(&mut self, ev: PipeEvent) {
        match ev {
            PipeEvent::SpeechPending(pending) => {
                if pending != self.last_speech_pending {
                    self.last_speech_pending = pending;
                    self.overlay.set_speech_pending(pending);
                }
            }
            PipeEvent::EditCapReached => {
                info!("edit buffer cap reached — stopping into EditReview");
                if self.state.state == DaemonState::Recording
                    || self.state.state == DaemonState::Paused
                {
                    if let Err(e) = self.do_stop() {
                        warn!(error = %e, "auto-stop on edit cap failed");
                    }
                }
            }
            PipeEvent::Error(msg) => {
                warn!(error = %msg, "pipeline error");
                self.state.set_error(msg);
            }
        }
    }

    fn handle_overlay(&mut self, ev: ControlEvent) -> Result<()> {
        match ev {
            ControlEvent::Pause => {
                let _ = self.do_pause();
            }
            ControlEvent::Resume => {
                let _ = self.do_resume();
            }
            ControlEvent::Stop => {
                self.do_stop()?;
            }
            ControlEvent::CommitEdit(text) => {
                self.do_copy_edit(Some(text))?;
            }
            ControlEvent::DiscardEdit => {
                self.do_discard_edit();
            }
        }
        Ok(())
    }

    fn handle_request(&mut self, req: Request) -> Response {
        match req {
            Request::Status => self.state.status_snapshot(),
            Request::Start { mode } => match self.do_start(mode) {
                Ok(()) => Response::ok(),
                Err(e) => Response::error(format!("{e:#}")),
            },
            Request::Stop => match self.do_stop() {
                Ok(()) => Response::ok(),
                Err(e) => Response::error(format!("{e:#}")),
            },
            Request::Toggle { mode } => match self.do_toggle(mode) {
                Ok(()) => Response::ok(),
                Err(e) => Response::error(format!("{e:#}")),
            },
            Request::SetMode { mode } => {
                self.do_set_mode(mode);
                Response::ok()
            }
            Request::Pause => {
                if self.do_pause() {
                    Response::ok()
                } else {
                    Response::ok_msg("not recording")
                }
            }
            Request::Resume => {
                if self.do_resume() {
                    Response::ok()
                } else {
                    Response::ok_msg("not paused")
                }
            }
        }
    }

    // ── transitions (single path) ─────────────────────────────────────────

    fn do_start(&mut self, mode: Mode) -> Result<()> {
        let previous_state = self.state.state;
        let previous_mode = self.state.mode;
        let previous_edit_buffer = self.state.edit_buffer.clone();

        match self.state.start(mode) {
            StartResult::AlreadyRecording => {
                debug!("start while recording — no-op Ok");
                return Ok(());
            }
            StartResult::Resumed => {
                // Paused → Recording: re-enable feeding only.
                if previous_mode != self.state.mode {
                    // A paused session can be explicitly retargeted with
                    // `start --mode`; discard partial VAD/STT continuity so
                    // the new mode starts at a clean boundary.
                    self.stt.cancel_previews();
                    if let Err(e) = self.stt.wait_idle(Duration::from_secs(30)) {
                        self.state.state = previous_state;
                        self.state.mode = previous_mode;
                        self.state.edit_buffer = previous_edit_buffer;
                        self.state
                            .set_error(format!("stt worker unavailable: {e:#}"));
                        return Err(e).context("retarget paused session");
                    }
                    if let Ok(session_id) = self.stt.reset_session() {
                        self.stt_session_id = session_id;
                    }
                    self.pipe_cmd(PipeCmd::Reset);
                    self.pipe_cmd(PipeCmd::SetMode(mode));
                }
                self.stt.resume_previews();
                self.audio.resume_feed();
                self.overlay.set_paused(false);
                info!(?mode, "resumed");
                return Ok(());
            }
            StartResult::Started => {}
        }

        // Preview work is lossy. Cancel it before model/session commands so a
        // new session cannot inherit stale preview jobs from the old one.
        self.stt.cancel_previews();

        // Ensure model for lazy residency / prior preload failure.
        if let Err(e) = self.stt.ensure_loaded() {
            // Keep an EditReview usable if a new start cannot load the model.
            self.state.state = previous_state;
            self.state.mode = previous_mode;
            self.state.edit_buffer = previous_edit_buffer;
            self.state.set_error(format!("model load failed: {e:#}"));
            return Err(e).context("ensure whisper model loaded");
        }

        if let Ok(session_id) = self.stt.reset_session() {
            self.stt_session_id = session_id;
        }
        if let Err(e) = self.stt.wait_idle(Duration::from_secs(30)) {
            self.state.state = previous_state;
            self.state.mode = previous_mode;
            self.state.edit_buffer = previous_edit_buffer;
            self.state
                .set_error(format!("stt worker unavailable: {e:#}"));
            return Err(e).context("prepare stt worker");
        }
        self.stt.resume_previews();

        // Clear model errors once load succeeds.
        if self
            .state
            .last_error
            .as_deref()
            .is_some_and(|m| m.contains("model"))
        {
            self.state.clear_error();
        }

        // Leave any prior EditReview surface.
        if self.overlay.is_enabled() {
            // show_live will replace; destroy first for cleanliness.
        }

        self.pipe_cmd(PipeCmd::Reset);
        self.pipe_cmd(PipeCmd::Arm { mode });

        if let Err(e) = self.audio.start() {
            // Do not leave the state machine claiming to record when the
            // PipeWire stream failed to open.
            self.pipe_cmd(PipeCmd::Reset);
            self.state.state = previous_state;
            self.state.mode = previous_mode;
            self.state.edit_buffer = previous_edit_buffer;
            self.state.set_error(format!("audio start failed: {e:#}"));
            return Err(e).context("open PipeWire capture stream");
        }
        self.audio.resume_feed();

        self.last_speech_pending = false;
        self.overlay.set_speech_pending(false);
        self.overlay.set_paused(false);
        self.overlay.set_committed("");
        self.overlay.set_preview("");
        self.overlay.show_live();

        info!(?mode, "recording started");
        Ok(())
    }

    fn do_stop(&mut self) -> Result<()> {
        if !self.state.is_capturing() {
            debug!("stop while idle/review — no-op");
            return Ok(());
        }

        let mode = self.state.mode;
        let was_paused = self.state.state == DaemonState::Paused;
        // Stop accepting lossy previews before flushing the VAD pipeline.
        // Final chunks must reach the worker ahead of any stale previews.
        self.stt.cancel_previews();

        // Flush pipeline (and take full Edit buffer when accumulating).
        let flushed = self.pipe_flush();

        // Close audio stream.
        if let Err(e) = self.audio.stop() {
            warn!(error = %e, "audio stop error");
        }

        self.last_speech_pending = false;

        match mode {
            Mode::Live => {
                if was_paused {
                    // A paused Live session has no more audio to commit. Do
                    // not wait behind an in-flight preview; it can finish in
                    // the background and its session tag will be discarded.
                    let stop = self.state.stop();
                    if stop != StopResult::Stopped {
                        warn!(?stop, "paused live stop produced unexpected state transition");
                    }
                    if let Ok(session_id) = self.stt.reset_session() {
                        self.stt_session_id = session_id;
                    }
                    self.overlay.destroy_surface();
                    info!("paused live session stopped → idle");
                    return Ok(());
                }

                // Live: force-close any open utterance so the last words are transcribed.
                if let Some(tail) = flushed {
                    if !tail.is_empty() {
                        if let Err(e) = self.stt.push_chunk(tail) {
                            warn!(error = %e, "failed to enqueue final live chunk");
                        }
                    }
                }

                // Keep the state in Recording/Paused while final results are
                // drained so `on_transcript` still performs injection. The
                // barrier also covers final chunks that closed just before
                // Stop and were already queued on the worker.
                if let Err(e) = self.stt.wait_idle(Duration::from_secs(30)) {
                    self.state
                        .set_error(format!("stt stop flush failed: {e:#}"));
                    warn!(error = %e, "timed out waiting for final live transcript");
                }
                self.drain_stt_results();

                let stop = self.state.stop();
                if stop != StopResult::Stopped {
                    warn!(?stop, "live stop produced unexpected state transition");
                }
                if let Ok(session_id) = self.stt.reset_session() {
                    self.stt_session_id = session_id;
                }
                self.overlay.destroy_surface();
                info!("stopped → idle");
                Ok(())
            }
            Mode::Edit => {
                let stop = self.state.stop();
                if stop != StopResult::EnteredEditReview {
                    warn!(?stop, "edit stop produced unexpected state transition");
                    return Ok(());
                }
                let samples = flushed.unwrap_or_default();
                info!(n_samples = samples.len(), "edit stop — full-buffer STT");

                let text = if samples.is_empty() {
                    String::new()
                } else {
                    match self.stt.transcribe_full(samples) {
                        Ok(r) => r.text,
                        Err(e) => {
                            self.state.set_error(format!("edit STT failed: {e:#}"));
                            // Stay in EditReview with empty buffer so user can discard.
                            String::new()
                        }
                    }
                };

                if let Ok(session_id) = self.stt.reset_session() {
                    self.stt_session_id = session_id;
                }
                self.state.set_edit_buffer(text.clone());
                self.overlay.show_edit_review(&text);
                info!("entered EditReview");
                Ok(())
            }
        }
    }

    fn do_toggle(&mut self, mode: Mode) -> Result<()> {
        // Peek current state before mutating so we know which branch to run.
        match self.state.state {
            DaemonState::Recording => self.do_stop(),
            DaemonState::Idle | DaemonState::Paused | DaemonState::EditReview => {
                self.do_start(mode)
            }
        }
    }

    fn do_set_mode(&mut self, mode: Mode) {
        match self.state.set_mode(mode) {
            SetModeResult::Stored { mode } => {
                debug!(?mode, "mode stored (no active session retarget)");
            }
            SetModeResult::SessionModeChanged { from, to } => {
                info!(?from, ?to, "retargeting pipeline mode mid-session");
                if let Ok(session_id) = self.stt.reset_session() {
                    self.stt_session_id = session_id;
                }
                self.pipe_cmd(PipeCmd::Reset);
                self.pipe_cmd(PipeCmd::SetMode(to));
            }
        }
    }

    fn do_pause(&mut self) -> bool {
        if !self.state.pause() {
            return false;
        }
        self.stt.cancel_previews();
        self.audio.pause_feed();
        self.overlay.set_paused(true);
        info!("paused");
        true
    }

    fn do_resume(&mut self) -> bool {
        if !self.state.resume() {
            return false;
        }
        self.stt.resume_previews();
        self.audio.resume_feed();
        self.overlay.set_paused(false);
        info!("resumed");
        true
    }

    fn do_copy_edit(&mut self, text_override: Option<String>) -> Result<()> {
        if self.state.state != DaemonState::EditReview {
            bail!("not in edit review");
        }
        let text = text_override.unwrap_or_else(|| self.state.edit_buffer.clone());
        if !text.is_empty() {
            if self.inject.is_enabled() {
                if let Err(e) = self.inject.inject_text(&text) {
                    self.state.set_error(format!("inject: {e:#}"));
                    return Err(e).context("inject edit buffer");
                }
            } else if self.cfg.injection.enabled {
                let msg = self
                    .inject
                    .last_error()
                    .unwrap_or(Injector::unavailable_message())
                    .to_string();
                self.state.set_error(msg.clone());
                bail!("{msg}");
            }
        }
        self.state.finish_edit_review();
        self.overlay.destroy_surface();
        info!("edit copy → idle");
        Ok(())
    }

    fn do_discard_edit(&mut self) {
        if self.state.state != DaemonState::EditReview {
            return;
        }
        self.state.finish_edit_review();
        self.overlay.destroy_surface();
        info!("edit discard → idle");
    }

    fn pipe_cmd(&self, cmd: PipeCmd) {
        if let Err(e) = self.pipe_cmd_tx.send(cmd) {
            warn!("pipeline command channel closed: {e}");
        }
    }

    fn pipe_flush(&self) -> Option<Vec<f32>> {
        let (tx, rx) = mpsc::channel();
        if self.pipe_cmd_tx.send(PipeCmd::Flush { reply: tx }).is_err() {
            warn!("pipeline flush: command channel closed");
            return None;
        }
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(buf) => buf,
            Err(_) => {
                warn!("pipeline flush timed out");
                None
            }
        }
    }

    fn shutdown_subsystems(&mut self) {
        info!("shutting down subsystems");
        // Best-effort stop capture.
        if self.state.is_capturing() {
            let _ = self.audio.stop();
        }
        self.pipe_cmd(PipeCmd::Shutdown);
        self.overlay.destroy_surface();
        let _ = self.stt.reset_session();
        // SttEngine + OverlayHandle drop joins their threads.
    }
}

// ─── VAD pipeline thread ────────────────────────────────────────────────────

fn pipeline_main(
    sample_rx: Receiver<Vec<f32>>,
    silence_gap_ms: u32,
    stt_cmd: SyncSender<SttCommand>,
    stt_preview: SyncSender<Vec<f32>>,
    cmd_rx: Receiver<PipeCmd>,
    ev_tx: Sender<PipeEvent>,
    max_edit_samples: usize,
) {
    info!("pipeline thread started");

    // Construct VAD here — `webrtc_vad::Vad` is not `Send`.
    let mut chunker = match Chunker::new(
        SAMPLE_RATE_HZ,
        silence_gap_ms,
        DEFAULT_FRAME_MS,
        None,
        SpeechPending::new(),
    ) {
        Ok(c) => c,
        Err(e) => {
            let _ = ev_tx.send(PipeEvent::Error(format!("VAD init failed: {e:#}")));
            error!(error = %e, "pipeline: chunker init failed");
            return;
        }
    };

    let mut armed = false;
    let mut mode = Mode::Live;
    let mut last_pending = false;
    let mut last_preview = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .unwrap_or_else(Instant::now);
    let preview_interval = Duration::from_millis(400);
    let mut startup_ignore_samples: usize = 0;
    // Wall-clock edit capture budget (~max_edit_seconds of mono PCM).
    let mut edit_samples: usize = 0;
    let mut cap_notified = false;

    loop {
        // Drain commands first.
        loop {
            match cmd_rx.try_recv() {
                Ok(PipeCmd::Shutdown) => {
                    info!("pipeline shutdown");
                    return;
                }
                Ok(PipeCmd::Arm { mode: m }) => {
                    mode = m;
                    armed = true;
                    startup_ignore_samples = (SAMPLE_RATE_HZ as usize)
                        .saturating_mul(CAPTURE_STARTUP_IGNORE_MS as usize)
                        / 1000;
                    cap_notified = false;
                    edit_samples = 0;
                    chunker.reset();
                    chunker.set_accumulate_session(m == Mode::Edit);
                    last_pending = false;
                    debug!(?m, ignore_ms = CAPTURE_STARTUP_IGNORE_MS, "pipeline armed");
                }
                Ok(PipeCmd::Flush { reply }) => {
                    armed = false;
                    startup_ignore_samples = 0;
                    let buf = chunker.flush_all();
                    chunker.set_accumulate_session(false);
                    edit_samples = 0;
                    let _ = reply.send(buf);
                    last_pending = false;
                    let _ = ev_tx.send(PipeEvent::SpeechPending(false));
                    debug!("pipeline flushed / disarmed");
                }
                Ok(PipeCmd::SetMode(m)) => {
                    mode = m;
                    chunker.set_accumulate_session(m == Mode::Edit);
                    if m != Mode::Edit {
                        edit_samples = 0;
                        cap_notified = false;
                    }
                    debug!(?m, "pipeline mode set");
                }
                Ok(PipeCmd::Reset) => {
                    chunker.reset();
                    startup_ignore_samples = 0;
                    last_pending = false;
                    cap_notified = false;
                    edit_samples = 0;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    info!("pipeline command channel closed");
                    return;
                }
            }
        }

        // Pull PCM (audio already gates feeding via set_feeding).
        match sample_rx.recv_timeout(Duration::from_millis(15)) {
            Ok(samples) => {
                if !armed || samples.is_empty() {
                    continue;
                }

                if startup_ignore_samples > 0 {
                    startup_ignore_samples = startup_ignore_samples.saturating_sub(samples.len());
                    continue;
                }

                // Edit cap: stop accepting once capture exceeds max_edit_seconds.
                if mode == Mode::Edit && edit_samples >= max_edit_samples {
                    if !cap_notified {
                        cap_notified = true;
                        let _ = ev_tx.send(PipeEvent::EditCapReached);
                    }
                    continue;
                }

                if mode == Mode::Edit {
                    edit_samples = edit_samples.saturating_add(samples.len());
                }

                // Live: closed chunks → STT. Edit: session accumulate inside chunker.
                if let Some(chunk) = chunker.push_samples(&samples) {
                    if mode == Mode::Live && !chunk.is_empty() {
                        if let Err(e) = stt_cmd.try_send(SttCommand::LiveChunk(chunk)) {
                            match e {
                                std::sync::mpsc::TrySendError::Full(_) => {
                                    warn!("STT command queue full — dropping live chunk");
                                }
                                std::sync::mpsc::TrySendError::Disconnected(_) => {
                                    let _ = ev_tx
                                        .send(PipeEvent::Error("STT worker disconnected".into()));
                                    return;
                                }
                            }
                        }
                    }
                }

                let pending = chunker.speech_pending().get();
                if pending != last_pending {
                    last_pending = pending;
                    let _ = ev_tx.send(PipeEvent::SpeechPending(pending));
                }

                // Live in-progress preview (throttled).
                if mode == Mode::Live && pending && last_preview.elapsed() >= preview_interval {
                    let cur = chunker.peek_current();
                    // Need a bit of audio before whisper is useful (~0.5 s).
                    if cur.len() >= SAMPLE_RATE_HZ as usize / 2 {
                        let preview = cur.to_vec();
                        if stt_preview.try_send(preview).is_ok() {
                            last_preview = Instant::now();
                        }
                    }
                }

                if mode == Mode::Edit && !cap_notified && edit_samples >= max_edit_samples {
                    cap_notified = true;
                    let _ = ev_tx.send(PipeEvent::EditCapReached);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                info!("pipeline sample channel closed");
                return;
            }
        }
    }
}
