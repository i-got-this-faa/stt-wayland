//! GTK4 + gtk4-layer-shell overlay.
//!
//! - [`Layer::Overlay`](gtk4_layer_shell::Layer::Overlay)
//! - keyboard interactivity **None** while recording / live / paused
//! - EditReview: temporarily **OnDemand** so `GtkTextView` is editable;
//!   on Copy/Discard → None and surface destroyed (Idle)
//! - `gtk::init` once; runtime stays warm with **zero** surfaces when idle
//! - Destroy (not hide) the layer surface on stop / leave review

mod edit_pane;
mod live_card;

pub use edit_pane::EditPane;
pub use live_card::LiveCard;

use std::sync::{mpsc, Arc, Mutex};

#[cfg(feature = "overlay")]
use std::thread;
#[cfg(feature = "overlay")]
use std::time::Duration;

use anyhow::Result;
use tracing::debug;

#[cfg(feature = "overlay")]
use anyhow::Context;
#[cfg(feature = "overlay")]
use tracing::{info, warn};

use crate::config::OverlayConfig;

// ── Control plane (UI → daemon) ─────────────────────────────────────────────

/// UI actions. Overlay never talks to the control socket; the daemon wires this.
pub trait ControlSink: Send + Sync {
    fn pause(&self);
    fn resume(&self);
    fn stop(&self);
    fn commit_edit(&self, text: String);
    fn discard_edit(&self);
}

/// Control events emitted by the overlay (for channel-based sinks).
#[derive(Debug, Clone)]
pub enum ControlEvent {
    Pause,
    Resume,
    Stop,
    CommitEdit(String),
    DiscardEdit,
}

/// [`ControlSink`] that forwards into an `mpsc` channel (daemon owns the receiver).
#[derive(Clone)]
pub struct ChannelControlSink {
    tx: Arc<Mutex<mpsc::Sender<ControlEvent>>>,
}

impl ChannelControlSink {
    pub fn pair() -> (Self, mpsc::Receiver<ControlEvent>) {
        let (tx, rx) = mpsc::channel();
        (
            Self {
                tx: Arc::new(Mutex::new(tx)),
            },
            rx,
        )
    }

    pub fn from_sender(tx: mpsc::Sender<ControlEvent>) -> Self {
        Self {
            tx: Arc::new(Mutex::new(tx)),
        }
    }
}

impl ControlSink for ChannelControlSink {
    fn pause(&self) {
        let _ = self.tx.lock().ok().and_then(|tx| tx.send(ControlEvent::Pause).ok());
    }
    fn resume(&self) {
        let _ = self
            .tx
            .lock()
            .ok()
            .and_then(|tx| tx.send(ControlEvent::Resume).ok());
    }
    fn stop(&self) {
        let _ = self.tx.lock().ok().and_then(|tx| tx.send(ControlEvent::Stop).ok());
    }
    fn commit_edit(&self, text: String) {
        let _ = self
            .tx
            .lock()
            .ok()
            .and_then(|tx| tx.send(ControlEvent::CommitEdit(text)).ok());
    }
    fn discard_edit(&self) {
        let _ = self
            .tx
            .lock()
            .ok()
            .and_then(|tx| tx.send(ControlEvent::DiscardEdit).ok());
    }
}

// ── Layout ──────────────────────────────────────────────────────────────────

/// Layer-shell anchor + margin configuration.
#[derive(Debug, Clone)]
pub struct OverlayLayout {
    pub anchor_left: bool,
    pub anchor_right: bool,
    pub anchor_top: bool,
    pub anchor_bottom: bool,
    pub margin_left: i32,
    pub margin_right: i32,
    pub margin_top: i32,
    pub margin_bottom: i32,
}

impl Default for OverlayLayout {
    fn default() -> Self {
        // Bottom-centered floating card: stretch horizontally, pin bottom.
        Self {
            anchor_left: true,
            anchor_right: true,
            anchor_top: false,
            anchor_bottom: true,
            margin_left: 0,
            margin_right: 0,
            margin_top: 0,
            margin_bottom: 48,
        }
    }
}

// ── Commands (daemon → UI thread) ───────────────────────────────────────────

enum UiCmd {
    ShowLive,
    ShowEditReview { text: String },
    /// Destroy any mapped surface (stop / leave review / hide).
    DestroySurface,
    SetPreview(String),
    SetCommitted(String),
    AppendCommitted(String),
    SetSpeechPending(bool),
    SetPaused(bool),
    SetControl(Arc<dyn ControlSink>),
    SetLayout(OverlayLayout),
    Shutdown,
}

// ── Overlay handle ──────────────────────────────────────────────────────────

/// Overlay controller. GTK runs on a dedicated thread; methods are thread-safe
/// (command queue). When the `overlay` feature is off, this is a pure state stub.
pub struct OverlayHandle {
    enabled: bool,
    preview_crossfade_ms: u32,
    edit_buffer_cap_secs: u32,
    layout: OverlayLayout,
    live: LiveCard,
    edit: EditPane,
    #[cfg(feature = "overlay")]
    cmd_tx: Option<mpsc::Sender<UiCmd>>,
    #[cfg(feature = "overlay")]
    _join: Option<thread::JoinHandle<()>>,
    control: Option<Arc<dyn ControlSink>>,
}

impl OverlayHandle {
    pub fn new(cfg: &OverlayConfig) -> Result<Self> {
        Self::with_layout(cfg, OverlayLayout::default())
    }

    pub fn with_layout(cfg: &OverlayConfig, layout: OverlayLayout) -> Result<Self> {
        debug!(
            enabled = cfg.enabled,
            crossfade_ms = cfg.preview_crossfade_ms,
            "overlay: init"
        );

        #[cfg(feature = "overlay")]
        let (cmd_tx, join) = if cfg.enabled {
            let (tx, rx) = mpsc::channel::<UiCmd>();
            let crossfade = cfg.preview_crossfade_ms;
            let layout_for_thread = layout.clone();
            let join = thread::Builder::new()
                .name("sttd-overlay".into())
                .spawn(move || {
                    if let Err(e) = gtk_thread_main(rx, layout_for_thread, crossfade) {
                        warn!(error = %e, "overlay GTK thread exited with error");
                    }
                })
                .context("spawn overlay GTK thread")?;
            info!("overlay: GTK runtime warm (no surfaces)");
            (Some(tx), Some(join))
        } else {
            (None, None)
        };

        #[cfg(not(feature = "overlay"))]
        {
            let _ = &layout;
            debug!("overlay: compiled without `overlay` feature (stub)");
        }

        Ok(Self {
            enabled: cfg.enabled,
            preview_crossfade_ms: cfg.preview_crossfade_ms,
            edit_buffer_cap_secs: cfg.edit_buffer_cap_secs,
            layout,
            live: LiveCard::new(cfg.preview_crossfade_ms),
            edit: EditPane::new(cfg.edit_buffer_cap_secs),
            #[cfg(feature = "overlay")]
            cmd_tx,
            #[cfg(feature = "overlay")]
            _join: join,
            control: None,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn preview_crossfade_ms(&self) -> u32 {
        self.preview_crossfade_ms
    }

    pub fn edit_buffer_cap_secs(&self) -> u32 {
        self.edit_buffer_cap_secs
    }

    pub fn layout(&self) -> &OverlayLayout {
        &self.layout
    }

    /// Replace layer-shell anchors/margins (applied on next surface create).
    pub fn set_layout(&mut self, layout: OverlayLayout) {
        self.layout = layout.clone();
        self.send(UiCmd::SetLayout(layout));
    }

    /// Wire UI callbacks. Required before mic/stop/copy/discard do anything useful.
    pub fn set_control_sink(&mut self, sink: Arc<dyn ControlSink>) {
        self.control = Some(Arc::clone(&sink));
        self.send(UiCmd::SetControl(sink));
    }

    pub fn live_card(&mut self) -> &mut LiveCard {
        &mut self.live
    }

    pub fn edit_pane(&mut self) -> &mut EditPane {
        &mut self.edit
    }

    /// Show live preview (keyboard interactivity None).
    pub fn show_live(&mut self) {
        if !self.enabled {
            return;
        }
        debug!("overlay: show live card");
        self.edit.hide();
        self.live.show();
        self.send(UiCmd::ShowLive);
    }

    /// Destroy all overlay surfaces (do not merely hide). Keeps GTK runtime warm.
    pub fn hide(&mut self) {
        debug!("overlay: destroy surface(s)");
        self.live.hide();
        self.edit.hide();
        self.send(UiCmd::DestroySurface);
    }

    /// Alias for stop path: tear down surface completely.
    pub fn destroy_surface(&mut self) {
        self.hide();
    }

    /// Enter EditReview with OnDemand keyboard interactivity.
    pub fn show_edit_review(&mut self, text: &str) {
        if !self.enabled {
            return;
        }
        debug!(n_chars = text.chars().count(), "overlay: edit review");
        self.live.hide();
        self.edit.show(text);
        self.send(UiCmd::ShowEditReview {
            text: text.to_string(),
        });
    }

    pub fn set_preview(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.live.set_preview(text.clone());
        self.send(UiCmd::SetPreview(text));
    }

    pub fn set_committed(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.live.set_committed(text.clone());
        self.send(UiCmd::SetCommitted(text));
    }

    pub fn append_committed(&mut self, text: &str) {
        self.live.append_committed(text);
        self.send(UiCmd::AppendCommitted(text.to_string()));
    }

    pub fn set_speech_pending(&mut self, pending: bool) {
        self.live.set_speech_pending(pending);
        self.send(UiCmd::SetSpeechPending(pending));
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.live.set_paused(paused);
        self.send(UiCmd::SetPaused(paused));
    }

    #[cfg(feature = "overlay")]
    fn send(&self, cmd: UiCmd) {
        if let Some(tx) = &self.cmd_tx {
            if let Err(e) = tx.send(cmd) {
                warn!(error = %e, "overlay: command channel closed");
            }
        }
    }

    #[cfg(not(feature = "overlay"))]
    fn send(&self, _cmd: UiCmd) {}
}

impl Drop for OverlayHandle {
    fn drop(&mut self) {
        self.send(UiCmd::Shutdown);
    }
}

// ── GTK thread ──────────────────────────────────────────────────────────────

#[cfg(feature = "overlay")]
struct UiState {
    layout: OverlayLayout,
    crossfade_ms: u32,
    control: Option<Arc<dyn ControlSink>>,
    surface: Option<Surface>,
    shutdown: bool,
}

#[cfg(feature = "overlay")]
struct Surface {
    window: gtk4::Window,
    stack: gtk4::Stack,
    live: live_card::LiveCardWidgets,
    edit: edit_pane::EditPaneWidgets,
    mode: SurfaceMode,
}

#[cfg(feature = "overlay")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SurfaceMode {
    Live,
    Edit,
}

#[cfg(feature = "overlay")]
struct NoopSink;

#[cfg(feature = "overlay")]
impl ControlSink for NoopSink {
    fn pause(&self) {}
    fn resume(&self) {}
    fn stop(&self) {}
    fn commit_edit(&self, _text: String) {}
    fn discard_edit(&self) {}
}

#[cfg(feature = "overlay")]
fn gtk_thread_main(
    rx: mpsc::Receiver<UiCmd>,
    layout: OverlayLayout,
    crossfade_ms: u32,
) -> Result<()> {
    use std::cell::RefCell;
    use std::rc::Rc;

    gtk4::init().context("gtk::init")?;

    if !gtk4_layer_shell::is_supported() {
        warn!("gtk4-layer-shell protocol not supported on this compositor");
    }

    install_css(crossfade_ms);

    let state = Rc::new(RefCell::new(UiState {
        layout,
        crossfade_ms,
        control: None,
        surface: None,
        shutdown: false,
    }));

    let main_loop = glib::MainLoop::new(None, false);

    let state_poll = Rc::clone(&state);
    let loop_quit = main_loop.clone();
    glib::timeout_add_local(Duration::from_millis(8), move || {
        loop {
            match rx.try_recv() {
                Ok(cmd) => {
                    let shutdown = matches!(cmd, UiCmd::Shutdown);
                    handle_cmd(&state_poll, cmd);
                    if shutdown || state_poll.borrow().shutdown {
                        destroy_surface(&state_poll);
                        state_poll.borrow_mut().shutdown = true;
                        loop_quit.quit();
                        return glib::ControlFlow::Break;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    destroy_surface(&state_poll);
                    state_poll.borrow_mut().shutdown = true;
                    loop_quit.quit();
                    return glib::ControlFlow::Break;
                }
            }
        }
        glib::ControlFlow::Continue
    });

    debug!("overlay: entering GTK main loop (warm, zero surfaces)");
    main_loop.run();
    debug!("overlay: GTK main loop exited");
    Ok(())
}

#[cfg(feature = "overlay")]
fn install_css(crossfade_ms: u32) {
    let base = include_str!("style.css");
    let dynamic = format!(
        "\nlabel.sttd-preview {{ transition: opacity {crossfade_ms}ms ease-in-out; }}\n"
    );
    let css = format!("{base}{dynamic}");

    let provider = gtk4::CssProvider::new();
    // `load_from_string` needs gtk feature v4_12; `load_from_data` is always available.
    provider.load_from_data(&css);

    if let Some(display) = gtk4::gdk::Display::default() {
        gtk4::style_context_add_provider_for_display(
            &display,
            &provider,
            gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    } else {
        warn!("overlay: no default GdkDisplay; CSS not installed");
    }
}

#[cfg(feature = "overlay")]
fn handle_cmd(state: &std::rc::Rc<std::cell::RefCell<UiState>>, cmd: UiCmd) {
    match cmd {
        UiCmd::SetControl(sink) => {
            // If a surface was built with NoopSink, rebuild so buttons bind the real sink.
            let had_surface = state.borrow().surface.is_some();
            let mode = state.borrow().surface.as_ref().map(|s| s.mode);
            let edit_text = state
                .borrow()
                .surface
                .as_ref()
                .map(|s| s.edit.text());
            state.borrow_mut().control = Some(sink);
            if had_surface {
                destroy_surface(state);
                match mode {
                    Some(SurfaceMode::Edit) => {
                        ensure_edit(state, &edit_text.unwrap_or_default());
                    }
                    Some(SurfaceMode::Live) | None => ensure_live(state),
                }
            }
        }
        UiCmd::SetLayout(layout) => {
            state.borrow_mut().layout = layout;
            let st = state.borrow();
            if let Some(surf) = st.surface.as_ref() {
                apply_layout(&surf.window, &st.layout);
            }
        }
        UiCmd::ShowLive => ensure_live(state),
        UiCmd::ShowEditReview { text } => ensure_edit(state, &text),
        UiCmd::DestroySurface => destroy_surface(state),
        UiCmd::SetPreview(text) => {
            if let Some(surf) = state.borrow().surface.as_ref() {
                if surf.mode == SurfaceMode::Live {
                    surf.live.set_preview_crossfade(&text);
                }
            }
        }
        UiCmd::SetCommitted(text) => {
            if let Some(surf) = state.borrow().surface.as_ref() {
                surf.live.set_committed(&text);
            }
        }
        UiCmd::AppendCommitted(text) => {
            if let Some(surf) = state.borrow().surface.as_ref() {
                let cur = surf.live.committed.text();
                let mut s = cur.to_string();
                if !s.is_empty()
                    && !s.ends_with(char::is_whitespace)
                    && !text.starts_with(char::is_whitespace)
                {
                    s.push(' ');
                }
                s.push_str(&text);
                surf.live.set_committed(&s);
            }
        }
        UiCmd::SetSpeechPending(pending) => {
            if let Some(surf) = state.borrow().surface.as_ref() {
                surf.live.set_speech_pending(pending);
            }
        }
        UiCmd::SetPaused(paused) => {
            if let Some(surf) = state.borrow().surface.as_ref() {
                surf.live.set_paused(paused);
            }
        }
        UiCmd::Shutdown => {
            destroy_surface(state);
            state.borrow_mut().shutdown = true;
        }
    }
}

#[cfg(feature = "overlay")]
fn apply_layout(window: &gtk4::Window, layout: &OverlayLayout) {
    use gtk4_layer_shell::{Edge, LayerShell};

    window.set_anchor(Edge::Left, layout.anchor_left);
    window.set_anchor(Edge::Right, layout.anchor_right);
    window.set_anchor(Edge::Top, layout.anchor_top);
    window.set_anchor(Edge::Bottom, layout.anchor_bottom);
    window.set_margin(Edge::Left, layout.margin_left);
    window.set_margin(Edge::Right, layout.margin_right);
    window.set_margin(Edge::Top, layout.margin_top);
    window.set_margin(Edge::Bottom, layout.margin_bottom);
}

#[cfg(feature = "overlay")]
fn create_surface(state: &std::rc::Rc<std::cell::RefCell<UiState>>) {
    use gtk4::prelude::*;
    use gtk4_layer_shell::{KeyboardMode, Layer, LayerShell};

    let mut st = state.borrow_mut();
    if st.surface.is_some() {
        return;
    }

    let sink = match st.control.clone() {
        Some(s) => s,
        None => {
            warn!("overlay: creating surface without ControlSink (no-op callbacks)");
            Arc::new(NoopSink) as Arc<dyn ControlSink>
        }
    };

    let crossfade = st.crossfade_ms;
    let layout = st.layout.clone();

    let window = gtk4::Window::new();
    window.add_css_class("sttd-overlay");
    window.set_decorated(false);
    window.set_resizable(false);
    window.set_title(Some("sttd"));
    // Keep process alive after last window closes — runtime stays warm.
    window.set_hide_on_close(true);

    window.init_layer_shell();
    window.set_layer(Layer::Overlay);
    window.set_namespace("sttd");
    window.set_keyboard_mode(KeyboardMode::None);
    window.set_exclusive_zone(0);
    apply_layout(&window, &layout);

    let stack = gtk4::Stack::new();
    stack.set_transition_type(gtk4::StackTransitionType::Crossfade);
    stack.set_transition_duration(120);

    let live = live_card::LiveCardWidgets::build(crossfade, Arc::clone(&sink));
    let edit = edit_pane::EditPaneWidgets::build(sink);

    stack.add_named(&live.root, Some("live"));
    stack.add_named(&edit.root, Some("edit"));
    stack.set_visible_child_name("live");

    window.set_child(Some(&stack));

    st.surface = Some(Surface {
        window,
        stack,
        live,
        edit,
        mode: SurfaceMode::Live,
    });

    debug!("overlay: surface created");
}

#[cfg(feature = "overlay")]
fn ensure_live(state: &std::rc::Rc<std::cell::RefCell<UiState>>) {
    use gtk4::prelude::*;
    use gtk4_layer_shell::{KeyboardMode, LayerShell};

    create_surface(state);
    let mut st = state.borrow_mut();
    let Some(surf) = st.surface.as_mut() else {
        return;
    };
    surf.mode = SurfaceMode::Live;
    surf.stack.set_visible_child_name("live");
    surf.window.set_keyboard_mode(KeyboardMode::None);
    surf.window.present();
    debug!("overlay: live presented (keyboard None)");
}

#[cfg(feature = "overlay")]
fn ensure_edit(state: &std::rc::Rc<std::cell::RefCell<UiState>>, text: &str) {
    use gtk4::prelude::*;
    use gtk4_layer_shell::{KeyboardMode, LayerShell};

    create_surface(state);
    let mut st = state.borrow_mut();
    let Some(surf) = st.surface.as_mut() else {
        return;
    };
    surf.mode = SurfaceMode::Edit;
    surf.edit.set_text(text);
    surf.stack.set_visible_child_name("edit");
    // OnDemand so TextView can receive keys when focused.
    surf.window.set_keyboard_mode(KeyboardMode::OnDemand);
    surf.window.present();
    surf.edit.grab_focus();
    debug!("overlay: edit review presented (keyboard OnDemand)");
}

#[cfg(feature = "overlay")]
fn destroy_surface(state: &std::rc::Rc<std::cell::RefCell<UiState>>) {
    use gtk4::prelude::*;
    use gtk4_layer_shell::{KeyboardMode, LayerShell};

    let mut st = state.borrow_mut();
    if let Some(surf) = st.surface.take() {
        surf.window.set_keyboard_mode(KeyboardMode::None);
        surf.window.set_child(None::<&gtk4::Widget>);
        surf.window.destroy();
        debug!("overlay: surface destroyed");
    }
}
