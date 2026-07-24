//! PipeWire audio capture for `sttd`.
//!
//! Mono **16 kHz f32** (config-driven rate/channels). The stream is opened on
//! [`AudioCapture::start`] and closed on [`AudioCapture::stop`].
//! [`AudioCapture::set_feeding`] (Pause/Resume) stops pushing samples without
//! tearing the PipeWire stream down.
//!
//! Samples are delivered on a bounded `sync_channel` of `Vec<f32>`. The PipeWire
//! process callback only ever uses `try_send` so it never blocks the mainloop.
//!
//! # System packages
//!
//! Building with the default `audio` feature needs libpipewire development files:
//!
//! | Distro | Packages |
//! |--------|----------|
//! | Arch   | `pipewire` `libpipewire` (and `pkgconf`) |
//! | Debian/Ubuntu | `libpipewire-0.3-dev` `libspa-0.2-dev` `pkg-config` |
//! | Fedora | `pipewire-devel` `pkgconf-pkg-config` |
//!
//! Without system libs: `cargo check -p sttd --no-default-features` (null backend).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::config::AudioConfig;

#[cfg(feature = "audio")]
use std::sync::mpsc::{RecvTimeoutError, TrySendError};
#[cfg(feature = "audio")]
use std::time::Duration;
#[cfg(feature = "audio")]
use anyhow::anyhow;
#[cfg(feature = "audio")]
use tracing::trace;

/// Default bound on the PCM sample queue (number of `Vec<f32>` frames).
///
/// Each element is one PipeWire buffer's worth of mono f32. Sized for modest
/// backpressure without multi-second latency (~0.5–1 s depending on quanta).
pub const DEFAULT_PCM_QUEUE_CAPACITY: usize = 48;

/// Errors specific to audio capture.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("pipewire: {0}")]
    PipeWire(String),

    #[error("capture thread panicked or exited unexpectedly")]
    ThreadDied,

    #[error("timed out waiting for capture thread to become ready")]
    ReadyTimeout,

    #[error("capture thread failed to start: {0}")]
    StartFailed(String),
}

/// Commands sent into the PipeWire mainloop thread.
enum CaptureCmd {
    Stop,
}

/// Shared control flags between the host API and the capture thread.
struct CaptureShared {
    /// When false, process still dequeues PW buffers but does not enqueue PCM.
    feeding: AtomicBool,
    /// True while the PipeWire mainloop thread is alive and streaming.
    running: AtomicBool,
}

impl CaptureShared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            feeding: AtomicBool::new(true),
            running: AtomicBool::new(false),
        })
    }
}

/// Command sender into the capture thread (type depends on backend).
#[cfg(feature = "audio")]
type CmdSender = pipewire::channel::Sender<CaptureCmd>;
#[cfg(not(feature = "audio"))]
type CmdSender = mpsc::Sender<CaptureCmd>;

/// PipeWire (or null) audio capture handle.
///
/// Typical use:
/// ```ignore
/// let mut cap = AudioCapture::new(&cfg.audio)?;
/// let rx = cap.take_sample_receiver().expect("receiver once");
/// cap.start()?;
/// // ... read rx in VAD loop ...
/// cap.set_feeding(false); // Pause
/// cap.set_feeding(true);  // Resume
/// cap.stop()?;
/// ```
pub struct AudioCapture {
    sample_rate: u32,
    channels: u16,
    node_name: Option<String>,
    #[allow(dead_code)]
    pcm_capacity: usize,
    /// Producer side kept so restarts still push into the same queue.
    sample_tx: SyncSender<Vec<f32>>,
    sample_rx: Option<Receiver<Vec<f32>>>,
    shared: Arc<CaptureShared>,
    /// Channel into the capture thread (present only while running).
    cmd_tx: Option<CmdSender>,
    join: Option<JoinHandle<Result<()>>>,
}

impl AudioCapture {
    /// Create a capture handle. Does **not** open a PipeWire stream yet.
    pub fn new(cfg: &AudioConfig) -> Result<Self> {
        Self::with_capacity(cfg, DEFAULT_PCM_QUEUE_CAPACITY)
    }

    /// Like [`new`](Self::new) but with an explicit PCM queue capacity.
    pub fn with_capacity(cfg: &AudioConfig, pcm_capacity: usize) -> Result<Self> {
        let pcm_capacity = pcm_capacity.max(1);
        let (sample_tx, sample_rx) = mpsc::sync_channel(pcm_capacity);

        debug!(
            sample_rate = cfg.sample_rate,
            channels = cfg.channels,
            node = ?cfg.node_name,
            pcm_capacity,
            "audio: init (stream deferred to start)"
        );

        Ok(Self {
            sample_rate: cfg.sample_rate,
            channels: cfg.channels,
            node_name: cfg.node_name.clone(),
            pcm_capacity,
            sample_tx,
            sample_rx: Some(sample_rx),
            shared: CaptureShared::new(),
            cmd_tx: None,
            join: None,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Take the PCM receiver (mono `f32` frames). Call once after `new`.
    ///
    /// Further starts keep using the same `SyncSender`; if the receiver was
    /// dropped, samples are discarded via `try_send` errors.
    pub fn take_sample_receiver(&mut self) -> Option<Receiver<Vec<f32>>> {
        self.sample_rx.take()
    }

    /// Whether a capture stream is currently running.
    pub fn is_running(&self) -> bool {
        self.shared.running.load(Ordering::Acquire)
    }

    /// Whether process callbacks push samples into the queue.
    pub fn is_feeding(&self) -> bool {
        self.shared.feeding.load(Ordering::Acquire)
    }

    /// Pause/Resume without tearing the PipeWire stream.
    ///
    /// `false` → still connected, still dequeues buffers, does not feed VAD.
    pub fn set_feeding(&self, feeding: bool) {
        self.shared.feeding.store(feeding, Ordering::Release);
        debug!(feeding, "audio: set_feeding");
    }

    /// Pause feeding VAD; keep stream alive.
    pub fn pause_feed(&mut self) {
        self.set_feeding(false);
    }

    /// Resume feeding VAD after pause.
    pub fn resume_feed(&mut self) {
        self.set_feeding(true);
    }

    /// Open the PipeWire stream (called on Start).
    ///
    /// No-op if already running.
    pub fn start(&mut self) -> Result<()> {
        if self.is_running() {
            debug!("audio: start ignored (already running)");
            return Ok(());
        }

        // Drain any stale PCM left from a previous session.
        if let Some(rx) = self.sample_rx.as_ref() {
            while rx.try_recv().is_ok() {}
        }

        self.shared.feeding.store(true, Ordering::Release);

        #[cfg(feature = "audio")]
        self.start_pipewire()?;

        #[cfg(not(feature = "audio"))]
        self.start_null()?;

        Ok(())
    }

    /// Close the PipeWire stream (called on Stop).
    ///
    /// No-op if not running.
    pub fn stop(&mut self) -> Result<()> {
        if !self.is_running() && self.join.is_none() {
            debug!("audio: stop ignored (not running)");
            return Ok(());
        }

        debug!("audio: stop requested");

        if let Some(tx) = self.cmd_tx.take() {
            // Best-effort; thread may already be exiting.
            let _ = tx.send(CaptureCmd::Stop);
        }

        if let Some(handle) = self.join.take() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    self.shared.running.store(false, Ordering::Release);
                    return Err(e).context("capture thread returned error on stop");
                }
                Err(_) => {
                    self.shared.running.store(false, Ordering::Release);
                    return Err(AudioError::ThreadDied.into());
                }
            }
        }

        self.shared.running.store(false, Ordering::Release);
        info!("audio: stream stopped");
        Ok(())
    }

    // ── PipeWire backend ──────────────────────────────────────────────────

    #[cfg(feature = "audio")]
    fn start_pipewire(&mut self) -> Result<()> {
        let sample_rate = self.sample_rate;
        let channels = self.channels;
        let node_name = self.node_name.clone();
        let sample_tx = self.sample_tx.clone();
        let shared = Arc::clone(&self.shared);

        let (cmd_tx, cmd_rx) = pipewire::channel::channel::<CaptureCmd>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

        let handle = thread::Builder::new()
            .name("sttd-pipewire".into())
            .spawn(move || {
                pipewire_thread(
                    sample_rate,
                    channels,
                    node_name,
                    sample_tx,
                    shared,
                    cmd_rx,
                    ready_tx,
                )
            })
            .context("spawn pipewire capture thread")?;

        // Wait for setup (or failure). Timeout avoids hanging the daemon forever.
        const READY_TIMEOUT: Duration = Duration::from_secs(5);
        match ready_rx.recv_timeout(READY_TIMEOUT) {
            Ok(Ok(())) => {
                self.cmd_tx = Some(cmd_tx);
                self.join = Some(handle);
                info!(
                    sample_rate = self.sample_rate,
                    channels = self.channels,
                    "audio: pipewire stream started"
                );
                Ok(())
            }
            Ok(Err(msg)) => {
                let _ = handle.join();
                self.shared.running.store(false, Ordering::Release);
                Err(AudioError::StartFailed(msg).into())
            }
            Err(RecvTimeoutError::Timeout) => {
                let _ = cmd_tx.send(CaptureCmd::Stop);
                let _ = handle.join();
                self.shared.running.store(false, Ordering::Release);
                Err(AudioError::ReadyTimeout.into())
            }
            Err(RecvTimeoutError::Disconnected) => match handle.join() {
                Ok(Ok(())) => Err(AudioError::StartFailed(
                    "capture thread exited before ready".into(),
                )
                .into()),
                Ok(Err(e)) => Err(e).context("capture thread failed during start"),
                Err(_) => Err(AudioError::ThreadDied.into()),
            },
        }
    }

    // ── Null backend (no libpipewire / --no-default-features) ─────────────

    #[cfg(not(feature = "audio"))]
    fn start_null(&mut self) -> Result<()> {
        let shared = Arc::clone(&self.shared);
        let (cmd_tx, cmd_rx) = mpsc::channel::<CaptureCmd>();

        shared.running.store(true, Ordering::Release);

        let handle = thread::Builder::new()
            .name("sttd-audio-null".into())
            .spawn(move || {
                debug!("audio: null backend running (no libpipewire)");
                // Block until Stop. No samples are produced.
                while let Ok(cmd) = cmd_rx.recv() {
                    match cmd {
                        CaptureCmd::Stop => break,
                    }
                }
                shared.running.store(false, Ordering::Release);
                Ok(())
            })
            .context("spawn null audio thread")?;

        self.cmd_tx = Some(cmd_tx);
        self.join = Some(handle);
        info!("audio: null backend started (feature `audio` disabled)");
        Ok(())
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        if self.is_running() || self.join.is_some() {
            if let Err(e) = self.stop() {
                warn!("audio: error while stopping on drop: {e:#}");
            }
        }
    }
}

// ── PipeWire thread implementation ─────────────────────────────────────────

#[cfg(feature = "audio")]
fn pipewire_thread(
    sample_rate: u32,
    channels: u16,
    node_name: Option<String>,
    sample_tx: SyncSender<Vec<f32>>,
    shared: Arc<CaptureShared>,
    cmd_rx: pipewire::channel::Receiver<CaptureCmd>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<()> {
    use pipewire as pw;
    use pw::spa::param::format::{MediaSubtype, MediaType};
    use pw::spa::param::format_utils;
    use pw::spa::pod::Pod;
    use pw::{properties::properties, spa};

    // Convert early setup failures into a ready-channel error so start() unblocks.
    let report_err = |msg: String| {
        let _ = ready_tx.send(Err(msg.clone()));
        Err(anyhow!(msg))
    };

    pw::init();

    let mainloop = match pw::main_loop::MainLoopRc::new(None) {
        Ok(m) => m,
        Err(e) => return report_err(format!("MainLoop: {e}")),
    };
    let context = match pw::context::ContextRc::new(&mainloop, None) {
        Ok(c) => c,
        Err(e) => return report_err(format!("Context: {e}")),
    };
    let core = match context.connect_rc(None) {
        Ok(c) => c,
        Err(e) => return report_err(format!("connect to pipewire: {e}")),
    };

    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::NODE_NAME => "sttd-capture",
        *pw::keys::NODE_DESCRIPTION => "sttd speech capture",
    };
    if let Some(ref target) = node_name {
        // "target.object" is PW_KEY_TARGET_OBJECT (pipewire ≥0.3.44).
        // Use the string key so we don't require a versioned crate feature.
        props.insert("target.object", target.as_str());
    }

    let stream = match pw::stream::StreamRc::new(core, "sttd-capture", props) {
        Ok(s) => s,
        Err(e) => return report_err(format!("Stream::new: {e}")),
    };

    // Negotiated format filled in param_changed (defaults to requested).
    struct UserData {
        rate: u32,
        channels: u32,
        sample_tx: SyncSender<Vec<f32>>,
        shared: Arc<CaptureShared>,
        dropped_frames: u64,
    }

    let user_data = UserData {
        rate: sample_rate,
        channels: u32::from(channels).max(1),
        sample_tx,
        shared: Arc::clone(&shared),
        dropped_frames: 0,
    };

    let listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed(|_stream, _ud, old, new| {
            debug!(?old, ?new, "audio: pipewire stream state");
        })
        .param_changed(|_stream, user_data, id, param| {
            let Some(param) = param else {
                return;
            };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let (media_type, media_subtype) = match format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }
            let mut info = spa::param::audio::AudioInfoRaw::new();
            if info.parse(param).is_err() {
                return;
            }
            user_data.rate = info.rate();
            user_data.channels = info.channels().max(1);
            info!(
                rate = user_data.rate,
                channels = user_data.channels,
                "audio: capturing"
            );
        })
        .process(|stream, user_data| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }

            // Always dequeue; only push when feeding.
            if !user_data.shared.feeding.load(Ordering::Relaxed) {
                return;
            }

            let data = &mut datas[0];
            let chunk = data.chunk();
            let size = chunk.size() as usize;
            let offset = chunk.offset() as usize;
            if size == 0 {
                return;
            }

            let Some(bytes) = data.data() else {
                return;
            };
            if offset.saturating_add(size) > bytes.len() {
                return;
            }
            let raw = &bytes[offset..offset + size];
            let n_channels = user_data.channels.max(1) as usize;
            let sample_bytes = std::mem::size_of::<f32>();
            let n_interleaved = size / sample_bytes;
            if n_interleaved == 0 {
                return;
            }
            let n_frames = n_interleaved / n_channels;
            if n_frames == 0 {
                return;
            }

            let mut mono = Vec::with_capacity(n_frames);
            if n_channels == 1 {
                for i in 0..n_frames {
                    let start = i * sample_bytes;
                    let frame = f32::from_le_bytes([
                        raw[start],
                        raw[start + 1],
                        raw[start + 2],
                        raw[start + 3],
                    ]);
                    mono.push(frame);
                }
            } else {
                for f in 0..n_frames {
                    let mut acc = 0.0f32;
                    for ch in 0..n_channels {
                        let idx = (f * n_channels + ch) * sample_bytes;
                        let s = f32::from_le_bytes([
                            raw[idx],
                            raw[idx + 1],
                            raw[idx + 2],
                            raw[idx + 3],
                        ]);
                        acc += s;
                    }
                    mono.push(acc / n_channels as f32);
                }
            }

            match user_data.sample_tx.try_send(mono) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    user_data.dropped_frames += 1;
                    if user_data.dropped_frames == 1 || user_data.dropped_frames % 100 == 0 {
                        warn!(
                            dropped = user_data.dropped_frames,
                            "audio: PCM queue full; dropping frames (consumer too slow)"
                        );
                    }
                }
                Err(TrySendError::Disconnected(_)) => {
                    // VAD side dropped the receiver; keep streaming silently.
                    trace!("audio: sample receiver disconnected");
                }
            }
        })
        .register();

    let _listener = match listener {
        Ok(l) => l,
        Err(e) => return report_err(format!("stream listener: {e}")),
    };

    // Request mono f32 at the configured rate; PipeWire converts if needed.
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    audio_info.set_rate(sample_rate);
    audio_info.set_channels(u32::from(channels).max(1));

    let obj = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = match spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    ) {
        Ok((cursor, _)) => cursor.into_inner(),
        Err(e) => return report_err(format!("serialize format pod: {e}")),
    };

    let pod = match Pod::from_bytes(&values) {
        Some(p) => p,
        None => return report_err("invalid format pod".into()),
    };
    let mut params = [pod];

    // MAP_BUFFERS + AUTOCONNECT; avoid RT_PROCESS so process may allocate safely.
    if let Err(e) = stream.connect(
        spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
        &mut params,
    ) {
        return report_err(format!("stream.connect: {e}"));
    }

    shared.running.store(true, Ordering::Release);
    let _ = ready_tx.send(Ok(()));

    let mainloop_quit = mainloop.clone();
    let stream_stop = stream.clone();
    let shared_stop = Arc::clone(&shared);
    let _cmd = cmd_rx.attach(mainloop.loop_(), move |cmd| match cmd {
        CaptureCmd::Stop => {
            debug!("audio: pipewire thread received Stop");
            let _ = stream_stop.disconnect();
            shared_stop.running.store(false, Ordering::Release);
            mainloop_quit.quit();
        }
    });

    info!(
        sample_rate,
        channels,
        node = ?node_name,
        "audio: pipewire mainloop running"
    );
    mainloop.run();

    shared.running.store(false, Ordering::Release);
    debug!("audio: pipewire mainloop exited");
    Ok(())
}
