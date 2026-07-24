//! Edit-mode review pane (`GtkTextView`).
//!
//! Keyboard interactivity is OnDemand only while this pane is shown.
//! Copy / Discard returns interactivity to None and state to Idle (via ControlSink).
//! Buffer cap ≈ 5 minutes of speech (wall-clock / config) — enforced by caller.

use tracing::trace;

#[cfg(feature = "overlay")]
use std::sync::Arc;

#[cfg(feature = "overlay")]
use super::ControlSink;

/// Logical edit-pane state.
#[derive(Debug, Clone, Default)]
pub struct EditPaneState {
    pub buffer: String,
    pub visible: bool,
}

/// Scaffold / state handle for the edit review pane.
pub struct EditPane {
    cap_secs: u32,
    state: EditPaneState,
}

impl EditPane {
    pub fn new(cap_secs: u32) -> Self {
        Self {
            cap_secs,
            state: EditPaneState::default(),
        }
    }

    pub fn show(&mut self, text: &str) {
        self.state.buffer = text.to_string();
        self.state.visible = true;
    }

    pub fn hide(&mut self) {
        self.state.visible = false;
        self.state.buffer.clear();
    }

    pub fn is_visible(&self) -> bool {
        self.state.visible
    }

    pub fn cap_secs(&self) -> u32 {
        self.cap_secs
    }

    pub fn buffer(&self) -> &str {
        &self.state.buffer
    }

    pub fn set_buffer(&mut self, text: impl Into<String>) {
        self.state.buffer = text.into();
        trace!(n = self.state.buffer.len(), "edit_pane: set_buffer");
    }

    /// Append transcribed text (cap is caller's responsibility).
    pub fn append(&mut self, text: &str) {
        self.state.buffer.push_str(text);
    }

    /// User chose Copy — returns buffer contents; caller injects then finish_edit_review.
    pub fn take_for_copy(&mut self) -> String {
        self.state.visible = false;
        std::mem::take(&mut self.state.buffer)
    }

    /// User chose Discard.
    pub fn discard(&mut self) {
        self.state.buffer.clear();
        self.hide();
    }
}

/// GTK widgets for the edit review pane (feature-gated).
#[cfg(feature = "overlay")]
pub struct EditPaneWidgets {
    pub root: gtk4::Box,
    pub view: gtk4::TextView,
    pub buffer: gtk4::TextBuffer,
    pub copy_btn: gtk4::Button,
    pub discard_btn: gtk4::Button,
}

#[cfg(feature = "overlay")]
impl EditPaneWidgets {
    pub fn build(sink: Arc<dyn ControlSink>) -> Self {
        use gtk4::prelude::*;

        let root = gtk4::Box::new(gtk4::Orientation::Vertical, 8);
        root.add_css_class("sttd-edit-card");
        root.set_halign(gtk4::Align::Center);
        root.set_valign(gtk4::Align::Center);

        let title = gtk4::Label::new(Some("Edit review — fix text, then Copy or Discard"));
        title.add_css_class("sttd-edit-title");
        title.set_halign(gtk4::Align::Start);

        let buffer = gtk4::TextBuffer::new(None);
        let view = gtk4::TextView::with_buffer(&buffer);
        view.add_css_class("sttd-edit-view");
        view.set_wrap_mode(gtk4::WrapMode::WordChar);
        view.set_accepts_tab(false);
        view.set_hexpand(true);
        view.set_vexpand(true);
        view.set_editable(true);
        view.set_cursor_visible(true);
        view.set_top_margin(8);
        view.set_bottom_margin(8);
        view.set_left_margin(10);
        view.set_right_margin(10);

        let scroll = gtk4::ScrolledWindow::new();
        scroll.set_child(Some(&view));
        scroll.set_min_content_height(120);
        scroll.set_max_content_height(320);
        scroll.set_propagate_natural_height(true);
        scroll.set_policy(gtk4::PolicyType::Automatic, gtk4::PolicyType::Automatic);
        scroll.set_hexpand(true);
        scroll.set_vexpand(true);

        let actions = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        actions.add_css_class("sttd-edit-actions");
        actions.set_halign(gtk4::Align::End);

        let discard_btn = gtk4::Button::with_label("Discard");
        discard_btn.add_css_class("sttd-discard");

        let copy_btn = gtk4::Button::with_label("Copy");
        copy_btn.add_css_class("sttd-copy");

        {
            let sink = Arc::clone(&sink);
            let buffer = buffer.clone();
            copy_btn.connect_clicked(move |_| {
                let (start, end) = buffer.bounds();
                let text = buffer.text(&start, &end, false);
                sink.commit_edit(text.to_string());
            });
        }

        {
            let sink = Arc::clone(&sink);
            discard_btn.connect_clicked(move |_| {
                sink.discard_edit();
            });
        }

        actions.append(&discard_btn);
        actions.append(&copy_btn);

        root.append(&title);
        root.append(&scroll);
        root.append(&actions);

        Self {
            root,
            view,
            buffer,
            copy_btn,
            discard_btn,
        }
    }

    pub fn set_text(&self, text: &str) {
        use gtk4::prelude::*;
        self.buffer.set_text(text);
    }

    pub fn text(&self) -> String {
        use gtk4::prelude::*;
        let (start, end) = self.buffer.bounds();
        self.buffer.text(&start, &end, false).to_string()
    }

    pub fn grab_focus(&self) {
        use gtk4::prelude::*;
        self.view.grab_focus();
    }
}
