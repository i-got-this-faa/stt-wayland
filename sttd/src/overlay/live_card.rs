//! Live-mode preview card (overlay-only partial text).
//!
//! VAD-closed chunks are committed+injected; in-progress preview is shown here
//! with ~200 ms crossfade. Mic toggles Pause/Resume; stop ends the session.

use tracing::trace;

#[cfg(feature = "overlay")]
use std::sync::Arc;

#[cfg(feature = "overlay")]
use super::ControlSink;

/// Logical live-card state (mirrored on the UI thread when the feature is on).
#[derive(Debug, Clone, Default)]
pub struct LiveCardState {
    pub preview: String,
    pub committed: String,
    pub speech_pending: bool,
    pub paused: bool,
    pub visible: bool,
}

/// Scaffold / state handle for the live preview card.
pub struct LiveCard {
    crossfade_ms: u32,
    state: LiveCardState,
}

impl LiveCard {
    pub fn new(crossfade_ms: u32) -> Self {
        Self {
            crossfade_ms,
            state: LiveCardState::default(),
        }
    }

    pub fn show(&mut self) {
        self.state.visible = true;
    }

    pub fn hide(&mut self) {
        self.state.visible = false;
        self.state.preview.clear();
        self.state.committed.clear();
        self.state.speech_pending = false;
        self.state.paused = false;
    }

    pub fn is_visible(&self) -> bool {
        self.state.visible
    }

    pub fn crossfade_ms(&self) -> u32 {
        self.crossfade_ms
    }

    pub fn state(&self) -> &LiveCardState {
        &self.state
    }

    /// Update in-progress (non-committed) preview text.
    pub fn set_preview(&mut self, text: impl Into<String>) {
        self.state.preview = text.into();
        trace!(preview = %self.state.preview, "live_card: preview");
    }

    pub fn preview(&self) -> &str {
        &self.state.preview
    }

    /// Replace committed labels text (already injected / finalized chunks).
    pub fn set_committed(&mut self, text: impl Into<String>) {
        self.state.committed = text.into();
        trace!(committed = %self.state.committed, "live_card: committed");
    }

    /// Append a committed fragment (typically after VAD-closed STT).
    pub fn append_committed(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if !self.state.committed.is_empty()
            && !self.state.committed.ends_with(char::is_whitespace)
            && !text.starts_with(char::is_whitespace)
        {
            self.state.committed.push(' ');
        }
        self.state.committed.push_str(text);
    }

    pub fn committed(&self) -> &str {
        &self.state.committed
    }

    pub fn set_speech_pending(&mut self, pending: bool) {
        self.state.speech_pending = pending;
    }

    pub fn speech_pending(&self) -> bool {
        self.state.speech_pending
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.state.paused = paused;
    }

    pub fn is_paused(&self) -> bool {
        self.state.paused
    }
}

/// GTK widgets for the live card (feature-gated).
#[cfg(feature = "overlay")]
pub struct LiveCardWidgets {
    pub root: gtk4::Box,
    pub mic_btn: gtk4::Button,
    pub stop_btn: gtk4::Button,
    pub committed: gtk4::Label,
    pub preview: gtk4::Label,
    pub dots: gtk4::Box,
    pub crossfade_ms: u32,
    /// Generation counter so overlapping crossfade timeouts don't stomp each other.
    pub preview_gen: std::rc::Rc<std::cell::Cell<u64>>,
}

#[cfg(feature = "overlay")]
impl LiveCardWidgets {
    pub fn build(crossfade_ms: u32, sink: Arc<dyn ControlSink>) -> Self {
        use gtk4::prelude::*;

        let root = gtk4::Box::new(gtk4::Orientation::Horizontal, 12);
        root.add_css_class("sttd-card");
        root.add_css_class("sttd-card-inner");
        root.set_valign(gtk4::Align::Center);
        root.set_halign(gtk4::Align::Center);

        let mic_btn = gtk4::Button::with_label("●");
        mic_btn.add_css_class("sttd-mic");
        mic_btn.set_tooltip_text(Some("Pause / Resume"));
        mic_btn.set_focusable(false);

        let stop_btn = gtk4::Button::with_label("■");
        stop_btn.add_css_class("sttd-stop");
        stop_btn.set_tooltip_text(Some("Stop"));
        stop_btn.set_focusable(false);

        let text_col = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        text_col.add_css_class("sttd-text-col");
        text_col.set_size_request(300, -1);
        text_col.set_valign(gtk4::Align::Center);

        let committed = gtk4::Label::new(None);
        committed.add_css_class("sttd-committed");
        committed.set_halign(gtk4::Align::Start);
        committed.set_xalign(0.0);
        committed.set_single_line_mode(true);
        committed.set_ellipsize(gtk4::pango::EllipsizeMode::Start);
        committed.set_width_chars(36);
        committed.set_max_width_chars(36);
        committed.set_selectable(false);

        let preview = gtk4::Label::new(None);
        preview.add_css_class("sttd-preview");
        preview.set_halign(gtk4::Align::Start);
        preview.set_xalign(0.0);
        preview.set_single_line_mode(true);
        preview.set_ellipsize(gtk4::pango::EllipsizeMode::Start);
        preview.set_width_chars(36);
        preview.set_max_width_chars(36);
        preview.set_selectable(false);
        preview.set_opacity(1.0);

        text_col.append(&committed);
        text_col.append(&preview);

        let dots = gtk4::Box::new(gtk4::Orientation::Horizontal, 2);
        dots.add_css_class("sttd-dots");
        dots.set_valign(gtk4::Align::End);
        for _ in 0..3 {
            let d = gtk4::Label::new(Some("•"));
            d.add_css_class("sttd-dot");
            dots.append(&d);
        }
        // Keep the indicator allocation stable; toggling visibility would
        // change the card width whenever speech starts or stops.
        dots.set_visible(true);

        // Mic: Pause when recording, Resume when paused (tracked via CSS class).
        {
            let sink = Arc::clone(&sink);
            let mic = mic_btn.clone();
            mic_btn.connect_clicked(move |_| {
                let paused = mic.has_css_class("paused");
                if paused {
                    sink.resume();
                } else {
                    sink.pause();
                }
            });
        }

        {
            let sink = Arc::clone(&sink);
            stop_btn.connect_clicked(move |_| {
                sink.stop();
            });
        }

        root.append(&mic_btn);
        root.append(&stop_btn);
        root.append(&text_col);
        root.append(&dots);

        Self {
            root,
            mic_btn,
            stop_btn,
            committed,
            preview,
            dots,
            crossfade_ms,
            preview_gen: std::rc::Rc::new(std::cell::Cell::new(0)),
        }
    }

    pub fn set_paused(&self, paused: bool) {
        use gtk4::prelude::*;
        if paused {
            self.mic_btn.add_css_class("paused");
            self.mic_btn.set_label("❚❚");
            self.mic_btn.set_tooltip_text(Some("Resume"));
        } else {
            self.mic_btn.remove_css_class("paused");
            self.mic_btn.set_label("●");
            self.mic_btn.set_tooltip_text(Some("Pause"));
        }
    }

    pub fn set_committed(&self, text: &str) {
        self.committed.set_text(text);
    }

    pub fn set_speech_pending(&self, pending: bool) {
        use gtk4::prelude::*;
        if pending {
            self.dots.add_css_class("pending");
        } else {
            self.dots.remove_css_class("pending");
        }
    }

    /// Crossfade preview text over `crossfade_ms` (fade out → swap → fade in).
    pub fn set_preview_crossfade(&self, text: &str) {
        use gtk4::prelude::*;
        use std::time::Duration;

        let new_text = text.to_string();
        let preview = self.preview.clone();
        let ms = self.crossfade_ms.max(1);
        let half = Duration::from_millis((ms as u64) / 2);

        let gen = self.preview_gen.get().wrapping_add(1);
        self.preview_gen.set(gen);

        // Immediate if empty→empty or first paint with empty current.
        let current = preview.text();
        if current.is_empty() && new_text.is_empty() {
            preview.remove_css_class("fading");
            preview.set_opacity(1.0);
            return;
        }
        if current.is_empty() {
            preview.set_text(&new_text);
            preview.remove_css_class("fading");
            preview.set_opacity(1.0);
            return;
        }

        preview.add_css_class("fading");
        preview.set_opacity(0.0);

        let preview_mid = preview.clone();
        let gen_holder = std::rc::Rc::clone(&self.preview_gen);
        glib::timeout_add_local(half, move || {
            if gen_holder.get() != gen {
                return glib::ControlFlow::Break;
            }
            preview_mid.set_text(&new_text);
            preview_mid.remove_css_class("fading");
            preview_mid.set_opacity(1.0);
            glib::ControlFlow::Break
        });
    }
}
