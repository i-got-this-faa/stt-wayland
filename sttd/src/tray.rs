//! StatusNotifierItem tray integration for `sttd`.
//!
//! The tray task is deliberately isolated from the daemon core. Its callbacks
//! only enqueue [`TrayEvent`]s; all state transitions continue to happen on the
//! core loop thread.

use std::sync::mpsc::{self, Receiver, Sender};

use anyhow::Result;
use proto::{DaemonState, Mode};
use tracing::debug;

use crate::config::TrayConfig;

/// User actions emitted by the tray task and handled by the daemon core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayEvent {
    Toggle,
    Quit,
}

/// The small state snapshot exposed to the tray host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraySnapshot {
    pub state: DaemonState,
    pub mode: Mode,
    pub error: Option<String>,
    pub model: String,
}

impl TraySnapshot {
    /// Human-readable tooltip and menu information text.
    pub fn description(&self) -> String {
        if let Some(error) = &self.error {
            return format!("error: {error}");
        }

        let model = if self.model.is_empty() {
            "unknown model"
        } else {
            self.model.as_str()
        };

        match self.state {
            DaemonState::Idle => format!("idle · {model}"),
            DaemonState::Paused => format!("paused · {model}"),
            // EditReview intentionally keeps the recording icon so the tray
            // remains associated with the active edit session.
            DaemonState::Recording | DaemonState::EditReview => {
                format!("recording ({}) · {model}", self.mode)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IconKind {
    Idle,
    Recording,
    Paused,
    Error,
}

impl IconKind {
    #[cfg(feature = "tray")]
    fn index(self) -> usize {
        match self {
            Self::Idle => 0,
            Self::Recording => 1,
            Self::Paused => 2,
            Self::Error => 3,
        }
    }
}

fn icon_for(snapshot: &TraySnapshot) -> IconKind {
    if snapshot.error.is_some() {
        return IconKind::Error;
    }

    match snapshot.state {
        DaemonState::Idle => IconKind::Idle,
        DaemonState::Recording | DaemonState::EditReview => IconKind::Recording,
        DaemonState::Paused => IconKind::Paused,
    }
}

fn toggle_label(state: DaemonState) -> &'static str {
    match state {
        DaemonState::Recording => "Stop",
        DaemonState::Idle | DaemonState::Paused | DaemonState::EditReview => "Start",
    }
}

/// Demultiply premultiplied RGBA pixels into straight RGBA.
fn demultiply_alpha(data: &mut [u8]) {
    for pixel in data.chunks_exact_mut(4) {
        let alpha = pixel[3];
        if alpha == 0 {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        } else if alpha != 255 {
            let alpha_f = alpha as u16;
            pixel[0] = ((pixel[0] as u16 * 255 + alpha_f / 2) / alpha_f).min(255) as u8;
            pixel[1] = ((pixel[1] as u16 * 255 + alpha_f / 2) / alpha_f).min(255) as u8;
            pixel[2] = ((pixel[2] as u16 * 255 + alpha_f / 2) / alpha_f).min(255) as u8;
        }
    }
}

/// Convert straight RGBA pixels to the network-byte-order ARGB32 format
/// required by StatusNotifierItem.
fn rgba_to_argb(data: &mut [u8]) {
    for pixel in data.chunks_exact_mut(4) {
        pixel.rotate_right(1);
    }
}

/// Handle owned by the daemon core for pushing snapshots and stopping the
/// tray service. The receiver is taken by the core-event bridge at startup.
pub struct TrayHandle {
    #[cfg(feature = "tray")]
    handle: std::sync::Mutex<Option<ksni::blocking::Handle<Tray>>>,
    #[cfg(feature = "tray")]
    enabled: bool,
    // Keep the sender alive for the lifetime of the daemon even when the
    // session bus is unavailable and no ksni service could be spawned.
    ev_tx: Sender<TrayEvent>,
}

impl TrayHandle {
    /// Create the tray handle and return its event receiver.
    ///
    /// Missing session D-Bus infrastructure is intentionally a soft failure:
    /// speech-to-text remains usable without a tray host.
    pub fn pair(cfg: &TrayConfig) -> Result<(Self, Receiver<TrayEvent>)> {
        let (ev_tx, ev_rx) = mpsc::channel();

        #[cfg(not(feature = "tray"))]
        {
            let _ = cfg;
            debug!("tray feature is not compiled; continuing without tray icon");
        }

        Ok((
            Self {
                #[cfg(feature = "tray")]
                handle: std::sync::Mutex::new(None),
                #[cfg(feature = "tray")]
                enabled: cfg.enabled,
                ev_tx,
            },
            ev_rx,
        ))
    }

    /// Start ksni on the non-Tokio core-loop thread.
    pub fn start(&self) {
        #[cfg(feature = "tray")]
        if self.enabled {
            let icons = match rasterize_icon_set() {
                Ok(icons) => Some(icons),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "tray icon rasterization failed; using themed icon names"
                    );
                    None
                }
            };

            let tray = Tray {
                snapshot: TraySnapshot {
                    state: DaemonState::Idle,
                    mode: Mode::Live,
                    error: None,
                    model: String::new(),
                },
                icons,
                ev_tx: self.ev_tx.clone(),
            };

            use ksni::blocking::TrayMethods;
            match tray.assume_sni_available(true).spawn() {
                Ok(handle) => {
                    if let Ok(mut guard) = self.handle.lock() {
                        *guard = Some(handle);
                    } else {
                        debug!("tray handle lock poisoned; stopping tray service");
                    }
                }
                Err(error) => {
                    debug!(error = %error, "tray unavailable; continuing without tray icon");
                }
            }
        } else {
            debug!("tray disabled by configuration");
        }
    }

    /// Push a new snapshot only when the caller has detected a state change.
    pub fn sync(&self, snapshot: TraySnapshot) {
        #[cfg(feature = "tray")]
        {
            let Ok(mut guard) = self.handle.lock() else {
                debug!("tray handle lock poisoned; disabling tray updates");
                return;
            };

            let service_gone = guard
                .as_ref()
                .is_some_and(|handle| handle.update(|tray| tray.apply(snapshot.clone())).is_none());

            if service_gone {
                *guard = None;
                debug!("tray service stopped; disabling tray updates");
            }
        }

        #[cfg(not(feature = "tray"))]
        let _ = snapshot;
    }

    /// Stop the ksni service before the daemon tears down its other threads.
    pub fn shutdown(&self) {
        #[cfg(feature = "tray")]
        {
            let handle = self.handle.lock().ok().and_then(|mut guard| guard.take());
            if let Some(handle) = handle {
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    handle.shutdown().wait();
                    let _ = tx.send(());
                });
                match rx.recv_timeout(std::time::Duration::from_secs(3)) {
                    Ok(()) => debug!("tray shutdown completed"),
                    Err(_) => debug!("tray shutdown timed out; continuing anyway"),
                }
            }
        }
    }
}

#[cfg(feature = "tray")]
struct IconSet {
    by_kind: [Vec<ksni::Icon>; 4],
}

#[cfg(feature = "tray")]
impl IconSet {
    fn get(&self, kind: IconKind) -> &[ksni::Icon] {
        &self.by_kind[kind.index()]
    }
}

#[cfg(feature = "tray")]
struct Tray {
    snapshot: TraySnapshot,
    icons: Option<IconSet>,
    ev_tx: Sender<TrayEvent>,
}

#[cfg(feature = "tray")]
impl Tray {
    fn apply(&mut self, snapshot: TraySnapshot) {
        self.snapshot = snapshot;
    }

    fn send(&self, event: TrayEvent) {
        let _ = self.ev_tx.send(event);
    }
}

#[cfg(feature = "tray")]
impl ksni::Tray for Tray {
    fn id(&self) -> String {
        "sttd".into()
    }

    fn title(&self) -> String {
        "sttd".into()
    }

    fn status(&self) -> ksni::Status {
        if self.snapshot.error.is_some() {
            ksni::Status::NeedsAttention
        } else {
            ksni::Status::Active
        }
    }

    fn icon_name(&self) -> String {
        match icon_for(&self.snapshot) {
            IconKind::Idle | IconKind::Recording => "audio-input-microphone",
            IconKind::Paused => "media-playback-paused-symbolic",
            IconKind::Error => "dialog-error-symbolic",
        }
        .into()
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        self.icons
            .as_ref()
            .map(|icons| icons.get(icon_for(&self.snapshot)).to_vec())
            .unwrap_or_default()
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            icon_name: self.icon_name(),
            icon_pixmap: self.icon_pixmap(),
            title: self.title(),
            description: self.snapshot.description(),
        }
    }

    fn activate(&mut self, _x: i32, _y: i32) {
        self.send(TrayEvent::Toggle);
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::StandardItem;

        let toggle_icon = match self.snapshot.state {
            DaemonState::Recording => "media-playback-stop-symbolic",
            DaemonState::Idle | DaemonState::Paused | DaemonState::EditReview => {
                "media-playback-start-symbolic"
            }
        };

        let toggle_tx = self.ev_tx.clone();
        let quit_tx = self.ev_tx.clone();

        let info = StandardItem {
            label: format!("sttd — {}", self.snapshot.description()),
            enabled: false,
            ..Default::default()
        };
        let toggle = StandardItem {
            label: toggle_label(self.snapshot.state).into(),
            icon_name: toggle_icon.into(),
            activate: Box::new(move |_tray| {
                let _ = toggle_tx.send(TrayEvent::Toggle);
            }),
            ..Default::default()
        };
        let quit = StandardItem {
            label: "Quit".into(),
            icon_name: "application-exit-symbolic".into(),
            activate: Box::new(move |_tray| {
                let _ = quit_tx.send(TrayEvent::Quit);
            }),
            ..Default::default()
        };

        vec![
            info.into(),
            toggle.into(),
            ksni::MenuItem::Separator,
            quit.into(),
        ]
    }

    fn watcher_offline(&self, reason: ksni::OfflineReason) -> bool {
        debug!(?reason, "tray watcher offline; waiting for it to return");
        true
    }
}

#[cfg(feature = "tray")]
fn rasterize_icon_set() -> Result<IconSet> {
    let mut by_kind: [Vec<ksni::Icon>; 4] = std::array::from_fn(|_| Vec::with_capacity(2));

    for kind in [
        IconKind::Idle,
        IconKind::Recording,
        IconKind::Paused,
        IconKind::Error,
    ] {
        for size in [16_u32, 24_u32] {
            by_kind[kind.index()].push(rasterize_svg(svg_for(kind), size)?);
        }
    }

    Ok(IconSet { by_kind })
}

#[cfg(feature = "tray")]
fn svg_for(kind: IconKind) -> &'static [u8] {
    match kind {
        IconKind::Idle => include_bytes!("../../assets/tray/idle.svg"),
        IconKind::Recording => include_bytes!("../../assets/tray/recording.svg"),
        IconKind::Paused => include_bytes!("../../assets/tray/paused.svg"),
        IconKind::Error => include_bytes!("../../assets/tray/error.svg"),
    }
}

#[cfg(feature = "tray")]
fn rasterize_svg(svg: &[u8], size: u32) -> Result<ksni::Icon> {
    let tree = resvg::usvg::Tree::from_data(svg, &resvg::usvg::Options::default())
        .map_err(|error| anyhow::anyhow!("parse tray SVG: {error}"))?;
    let svg_size = tree.size();
    if svg_size.width() <= 0.0 || svg_size.height() <= 0.0 {
        return Err(anyhow::anyhow!("tray SVG has an empty viewport"));
    }

    let mut pixmap = resvg::tiny_skia::Pixmap::new(size, size)
        .ok_or_else(|| anyhow::anyhow!("allocate {size}x{size} tray pixmap"))?;
    let transform = resvg::tiny_skia::Transform::from_scale(
        size as f32 / svg_size.width(),
        size as f32 / svg_size.height(),
    );
    let mut pixmap_mut = pixmap.as_mut();
    resvg::render(&tree, transform, &mut pixmap_mut);

    let mut data = pixmap.data().to_vec();
    demultiply_alpha(&mut data);
    rgba_to_argb(&mut data);
    Ok(ksni::Icon {
        width: size as i32,
        height: size as i32,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(state: DaemonState, mode: Mode, error: Option<&str>) -> TraySnapshot {
        TraySnapshot {
            state,
            mode,
            error: error.map(str::to_owned),
            model: "ggml-base.en.bin".into(),
        }
    }

    #[test]
    fn state_maps_to_expected_icon_and_error_wins() {
        assert_eq!(
            icon_for(&snapshot(DaemonState::Idle, Mode::Live, None)),
            IconKind::Idle
        );
        assert_eq!(
            icon_for(&snapshot(DaemonState::Recording, Mode::Live, None)),
            IconKind::Recording
        );
        assert_eq!(
            icon_for(&snapshot(DaemonState::EditReview, Mode::Edit, None)),
            IconKind::Recording
        );
        assert_eq!(
            icon_for(&snapshot(DaemonState::Paused, Mode::Live, None)),
            IconKind::Paused
        );
        assert_eq!(
            icon_for(&snapshot(DaemonState::Recording, Mode::Live, Some("boom"))),
            IconKind::Error
        );
    }

    #[test]
    fn menu_toggle_labels_follow_state() {
        assert_eq!(toggle_label(DaemonState::Idle), "Start");
        assert_eq!(toggle_label(DaemonState::Paused), "Start");
        assert_eq!(toggle_label(DaemonState::EditReview), "Start");
        assert_eq!(toggle_label(DaemonState::Recording), "Stop");

        let info = snapshot(DaemonState::Paused, Mode::Live, None).description();
        assert!(info.contains("paused"));
        assert!(info.contains("ggml-base.en.bin"));
    }

    #[test]
    fn snapshot_equality_is_suitable_for_diffing() {
        let original = snapshot(DaemonState::Idle, Mode::Live, None);
        assert_eq!(original, original.clone());

        let mut changed = original.clone();
        changed.mode = Mode::Edit;
        assert_ne!(original, changed);
    }

    #[test]
    fn rgba_conversion_is_argb32_big_endian() {
        let mut pixel = [0x11, 0x22, 0x33, 0x44];
        rgba_to_argb(&mut pixel);
        assert_eq!(pixel, [0x44, 0x11, 0x22, 0x33]);
    }

    #[test]
    fn error_tooltip_contains_error_text() {
        let tooltip = snapshot(DaemonState::Recording, Mode::Live, Some("audio failed"));
        assert_eq!(tooltip.description(), "error: audio failed");
    }

    #[cfg(feature = "tray")]
    #[test]
    fn bundled_icons_rasterize_at_both_sizes() {
        let icons = rasterize_icon_set().expect("bundled tray SVGs should render");
        for kind in [
            IconKind::Idle,
            IconKind::Recording,
            IconKind::Paused,
            IconKind::Error,
        ] {
            let pixmaps = icons.get(kind);
            assert_eq!(pixmaps.len(), 2);
            assert_eq!(pixmaps[0].width, 16);
            assert_eq!(pixmaps[0].height, 16);
            assert_eq!(pixmaps[1].width, 24);
            assert_eq!(pixmaps[1].height, 24);
            assert!(pixmaps.iter().all(|icon| !icon.data.is_empty()));
        }
    }
}
