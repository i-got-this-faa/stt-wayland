//! GTK4 + libadwaita settings application for `sttd`.
//!
//! The app edits the daemon's complete TOML configuration through named
//! profiles. Saving updates only the selected profile; applying a profile also
//! mirrors it to `~/.config/sttd/config.toml`, which keeps the existing
//! systemd user unit compatible. Restart `sttd` after applying changes.

#[path = "config.rs"]
mod config;
#[path = "model_catalog.rs"]
mod model_catalog;

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use adw::prelude::*;
use anyhow::{Context, Result};
use glib::value::ToValue;
use gtk4::{gio, glib};

use config::{
    default_models_dir, Config, ModelResidency, OverlayAnchor, PostProcessConfig, ProfileStore,
    PIPELINE_SAMPLE_RATE_HZ,
};
use model_catalog::{
    delete_managed_model, delete_post_process_model, download_model, download_post_process_model,
    format_size, install_state, managed_model_path, model_for_filename, post_process_install_state,
    InstallState, ModelSpec, POST_PROCESS_CATALOG, MODEL_CATALOG,
};

const ANCHORS: [(OverlayAnchor, &str); 9] = [
    (OverlayAnchor::Top, "Top"),
    (OverlayAnchor::Bottom, "Bottom"),
    (OverlayAnchor::Left, "Left"),
    (OverlayAnchor::Right, "Right"),
    (OverlayAnchor::TopLeft, "Top Left"),
    (OverlayAnchor::TopRight, "Top Right"),
    (OverlayAnchor::BottomLeft, "Bottom Left"),
    (OverlayAnchor::BottomRight, "Bottom Right"),
    (OverlayAnchor::Center, "Center"),
];

const RESIDENCIES: [&str; 2] = ["Preload", "Lazy"];

fn main() -> gtk4::glib::ExitCode {
    let app = adw::Application::builder()
        .application_id("io.github.stt_wayland.Settings")
        .build();

    app.connect_activate(|app| {
        if let Err(error) = build_ui(app) {
            show_startup_error(app, &error);
        }
    });
    app.run()
}

#[derive(Debug)]
struct EditorState {
    store: ProfileStore,
    /// Profile currently shown in the editor.
    current: String,
    /// Profile mirrored to `config.toml`, if any.
    active: Option<String>,
    config: Config,
}

#[derive(Debug, Clone)]
enum ModelChoice {
    Catalog(&'static ModelSpec),
    Custom(PathBuf),
}

impl ModelChoice {
    fn path(&self) -> Option<PathBuf> {
        match self {
            Self::Catalog(spec) => managed_model_path(&default_models_dir(), spec),
            Self::Custom(path) => Some(path.clone()),
        }
    }
}

#[derive(Debug)]
enum DownloadEvent {
    Progress { downloaded: u64, total: u64 },
    Finished(std::result::Result<PathBuf, String>),
}

/// Every editable row, kept flat so `collect_config` / `populate_form` can
/// read and write them without walking the widget tree.
struct Form {
    /// Guards signal handlers while rows are rebuilt programmatically.
    updating: Cell<bool>,
    /// Catalog and custom paths backing the rows of `model_picker`.
    model_choices: RefCell<Vec<ModelChoice>>,
    model_busy: Cell<bool>,
    model_cancel: RefCell<Option<Arc<AtomicBool>>>,
    /// Post-processing catalog ids backing `pp_model_picker`.
    pp_model_choices: RefCell<Vec<&'static model_catalog::PostProcessModelSpec>>,
    pp_model_busy: Cell<bool>,
    pp_model_cancel: RefCell<Option<Arc<AtomicBool>>>,
    profiles: adw::ComboRow,
    model_picker: adw::ComboRow,
    model_action_button: gtk4::Button,
    model_progress: gtk4::ProgressBar,
    model_path: adw::EntryRow,
    residency: adw::ComboRow,
    threads: adw::SpinRow,
    language: adw::EntryRow,
    vad_silence_ms: adw::SpinRow,
    max_edit_seconds: adw::SpinRow,
    stt_channel_capacity: adw::SpinRow,
    node_name: adw::EntryRow,
    overlay_enabled: adw::SwitchRow,
    overlay_anchor: adw::ComboRow,
    overlay_margin: adw::SpinRow,
    crossfade_ms: adw::SpinRow,
    injection_enabled: adw::SwitchRow,
    batch_mode: adw::SwitchRow,
    char_delay_ms: adw::SpinRow,
    tray_enabled: adw::SwitchRow,
    pp_enabled: adw::SwitchRow,
    pp_model_picker: adw::ComboRow,
    pp_model_action_button: gtk4::Button,
    pp_model_progress: gtk4::ProgressBar,
    pp_styling: adw::ComboRow,
    pp_structure: adw::ComboRow,
    pp_context: adw::ComboRow,
}

impl Form {
    fn new() -> Self {
        let profiles = adw::ComboRow::new();
        profiles.set_title("Profile");

        let model_picker = adw::ComboRow::new();
        model_picker.set_title("Speech Model");
        model_picker.set_subtitle("Choose an installed model or one to download");

        let model_action_button = gtk4::Button::from_icon_name("folder-download-symbolic");
        model_action_button.set_tooltip_text(Some("Download selected model"));
        model_action_button.set_valign(gtk4::Align::Center);
        model_action_button.add_css_class("flat");

        let model_progress = gtk4::ProgressBar::new();
        model_progress.set_show_text(true);
        model_progress.set_size_request(110, -1);
        model_progress.set_valign(gtk4::Align::Center);
        model_progress.set_opacity(0.0);

        let model_controls = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
        model_controls.set_size_request(154, -1);
        model_controls.set_halign(gtk4::Align::End);
        model_controls.set_valign(gtk4::Align::Center);
        model_controls.append(&model_progress);
        model_controls.append(&model_action_button);
        model_picker.add_suffix(&model_controls);

        let model_path = entry_row(
            "Model File",
            "Path to a local audio.cpp GGUF file (advanced; managed models are selected above)",
        );

        let residency = string_combo(&RESIDENCIES);
        residency.set_title("Residency");
        residency.set_subtitle("Preload at startup, or load lazily on first use");

        let threads = spin_row("Threads", "Inference threads (0 = auto)", 0.0, 128.0, 1.0);
        let language = entry_row(
            "Language",
            "Language or locale hint such as \"en\" or \"en-US\" (auto = detect)",
        );

        let vad_silence_ms = spin_row(
            "VAD Silence Gap",
            "Silence in milliseconds that ends a speech chunk",
            50.0,
            10_000.0,
            50.0,
        );
        let max_edit_seconds = spin_row(
            "Edit Capture Limit",
            "Maximum seconds kept for edit review",
            1.0,
            3_600.0,
            1.0,
        );
        let stt_channel_capacity = spin_row(
            "STT Queue Capacity",
            "Audio chunks buffered for transcription",
            1.0,
            256.0,
            1.0,
        );
        let node_name = entry_row(
            "Input Device",
            "Optional input device name (empty = system default). On Linux, use `pw-cli ls Node` to list PipeWire nodes.",
        );

        let overlay_enabled = switch_row("Enable Overlay", "Show the live transcription card");
        let anchor_labels: Vec<&str> = ANCHORS.iter().map(|(_, label)| *label).collect();
        let overlay_anchor = string_combo(&anchor_labels);
        overlay_anchor.set_title("Anchor");
        overlay_anchor.set_subtitle("Screen edge the overlay attaches to");
        let overlay_margin = spin_row(
            "Margin",
            "Distance from the anchored edge in pixels",
            0.0,
            2_000.0,
            1.0,
        );
        let crossfade_ms = spin_row(
            "Preview Crossfade",
            "Fade duration for preview text in milliseconds",
            0.0,
            5_000.0,
            10.0,
        );

        let injection_enabled = switch_row(
            "Enable Text Injection",
            "Type transcriptions into the focused window",
        );
        let batch_mode = switch_row(
            "Instant Input",
            "Type committed text without a per-character delay",
        );
        let char_delay_ms = spin_row(
            "Character Delay",
            "Delay between injected characters when Instant Input is off",
            0.0,
            1_000.0,
            1.0,
        );

        let tray_enabled = switch_row("Enable Tray Icon", "Show a status icon in the system tray");

        let pp_model_picker = adw::ComboRow::new();
        pp_model_picker.set_title("Normalization Model");
        pp_model_picker.set_subtitle("Choose an installed model or one to download");

        let pp_model_action_button = gtk4::Button::from_icon_name("folder-download-symbolic");
        pp_model_action_button.set_tooltip_text(Some("Download selected model"));
        pp_model_action_button.set_valign(gtk4::Align::Center);
        pp_model_action_button.add_css_class("flat");

        let pp_model_progress = gtk4::ProgressBar::new();
        pp_model_progress.set_show_text(true);
        pp_model_progress.set_size_request(110, -1);
        pp_model_progress.set_valign(gtk4::Align::Center);
        pp_model_progress.set_opacity(0.0);

        let pp_model_controls = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
        pp_model_controls.set_size_request(154, -1);
        pp_model_controls.set_halign(gtk4::Align::End);
        pp_model_controls.set_valign(gtk4::Align::Center);
        pp_model_controls.append(&pp_model_progress);
        pp_model_controls.append(&pp_model_action_button);
        pp_model_picker.add_suffix(&pp_model_controls);

        let pp_enabled = switch_row(
            "Enable Post-Processing",
            "Rewrite final transcripts with a local normalization model",
        );

        let axis_combo = |axis: &[(&str, &str)], title: &str, subtitle: &str| {
            let labels: Vec<&str> = axis.iter().map(|(_, label)| *label).collect();
            let row = string_combo(&labels);
            row.set_title(title);
            row.set_subtitle(subtitle);
            row
        };
        let pp_styling = axis_combo(
            PostProcessConfig::STYLINGS,
            "Styling",
            "Register of the written text",
        );
        let pp_structure = axis_combo(
            PostProcessConfig::STRUCTURES,
            "Structure",
            "Allow breaking enumerations into lists",
        );
        let pp_context = axis_combo(
            PostProcessConfig::CONTEXTS,
            "Context",
            "Destination conventions for the text",
        );

        Self {
            updating: Cell::new(false),
            model_choices: RefCell::new(Vec::new()),
            model_busy: Cell::new(false),
            model_cancel: RefCell::new(None),
            pp_model_choices: RefCell::new(Vec::new()),
            pp_model_busy: Cell::new(false),
            pp_model_cancel: RefCell::new(None),
            profiles,
            model_picker,
            model_action_button,
            model_progress,
            model_path,
            residency,
            threads,
            language,
            vad_silence_ms,
            max_edit_seconds,
            stt_channel_capacity,
            node_name,
            overlay_enabled,
            overlay_anchor,
            overlay_margin,
            crossfade_ms,
            injection_enabled,
            batch_mode,
            char_delay_ms,
            tray_enabled,
            pp_enabled,
            pp_model_picker,
            pp_model_action_button,
            pp_model_progress,
            pp_styling,
            pp_structure,
            pp_context,
        }
    }
}

/// Restores the previous guard value on drop, so nested refreshes cannot
/// accidentally re-enable signal handlers for an outer update.
struct UpdatingGuard<'a> {
    cell: &'a Cell<bool>,
    previous: bool,
}

impl<'a> UpdatingGuard<'a> {
    fn new(form: &'a Form) -> Self {
        Self {
            cell: &form.updating,
            previous: form.updating.replace(true),
        }
    }
}

impl Drop for UpdatingGuard<'_> {
    fn drop(&mut self) {
        self.cell.set(self.previous);
    }
}

fn string_combo(items: &[&str]) -> adw::ComboRow {
    let row = adw::ComboRow::new();
    row.set_model(Some(&gtk4::StringList::new(items)));
    row
}

fn spin_row(title: &str, subtitle: &str, min: f64, max: f64, step: f64) -> adw::SpinRow {
    let row = adw::SpinRow::with_range(min, max, step);
    row.set_title(title);
    row.set_subtitle(subtitle);
    row
}

fn switch_row(title: &str, subtitle: &str) -> adw::SwitchRow {
    let row = adw::SwitchRow::new();
    row.set_title(title);
    row.set_subtitle(subtitle);
    row
}

fn entry_row(title: &str, tooltip: &str) -> adw::EntryRow {
    let row = adw::EntryRow::new();
    row.set_title(title);
    row.set_tooltip_text(Some(tooltip));
    row
}

struct SettingsSection {
    name: &'static str,
    title: &'static str,
    sidebar_subtitle: &'static str,
    description: &'static str,
    icon_name: &'static str,
}

const SETTINGS_SECTIONS: [SettingsSection; 4] = [
    SettingsSection {
        name: "speech",
        title: "Speech",
        sidebar_subtitle: "Model and recognition",
        description: "Choose the model and tune recognition behavior.",
        icon_name: "audio-input-microphone-symbolic",
    },
    SettingsSection {
        name: "polish",
        title: "Post-Processing",
        sidebar_subtitle: "Text normalization",
        description: "Clean up final transcripts with a local normalization model.",
        icon_name: "spell-check-symbolic",
    },
    SettingsSection {
        name: "audio",
        title: "Audio",
        sidebar_subtitle: "Capture and detection",
        description: "Choose the audio source and when a speech chunk ends.",
        icon_name: "audio-card-symbolic",
    },
    SettingsSection {
        name: "interface",
        title: "Interface",
        sidebar_subtitle: "Overlay and text input",
        description: "Control what sttd shows and how text enters the focused window.",
        icon_name: "video-display-symbolic",
    },
];

/// Builds a single-column settings page with a centered, readable width.
fn settings_page(
    section: &SettingsSection,
    groups: &[&adw::PreferencesGroup],
) -> gtk4::ScrolledWindow {
    let content = gtk4::Box::new(gtk4::Orientation::Vertical, 18);
    content.set_margin_top(28);
    content.set_margin_bottom(40);
    content.set_margin_start(24);
    content.set_margin_end(24);
    content.set_valign(gtk4::Align::Start);

    let description = gtk4::Label::new(Some(section.description));
    description.set_xalign(0.0);
    description.set_wrap(true);
    description.add_css_class("dim-label");
    content.append(&description);

    for group in groups {
        group.set_hexpand(true);
        content.append(*group);
    }

    let clamp = adw::Clamp::builder()
        .maximum_size(720)
        .tightening_threshold(560)
        .child(&content)
        .build();
    gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .child(&clamp)
        .build()
}

fn build_ui(app: &adw::Application) -> Result<()> {
    let store = ProfileStore::default();
    store
        .ensure_initialized()
        .context("initialize profile store")?;
    let active = store
        .active()?
        .ok_or_else(|| anyhow::anyhow!("profile store has no active profile"))?;
    let config = store.load(&active).context("load active profile")?;

    let state = Rc::new(RefCell::new(EditorState {
        store,
        current: active,
        active: None,
        config,
    }));
    let active_pointer = state.borrow().store.active()?;
    state.borrow_mut().active = active_pointer;

    let form = Rc::new(Form::new());
    populate_form(&form, &state.borrow().config);
    refresh_models(&form, &state.borrow().config);

    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("STT Settings")
        .default_width(1000)
        .default_height(720)
        .build();

    // ── Header bar: window title plus the primary actions ────────────
    let header = adw::HeaderBar::new();
    let window_title = adw::WindowTitle::new("Speech", "");
    header.set_title_widget(Some(&window_title));

    let sidebar_button = gtk4::Button::from_icon_name("view-list-symbolic");
    sidebar_button.set_tooltip_text(Some("Show settings categories"));
    sidebar_button.set_visible(false);
    header.pack_start(&sidebar_button);

    let save_button = gtk4::Button::with_label("Save");
    save_button.set_tooltip_text(Some("Save changes to the selected profile"));
    let apply_button = gtk4::Button::with_label("Apply");
    apply_button.set_tooltip_text(Some("Save and activate the selected profile"));
    apply_button.add_css_class("suggested-action");
    header.pack_end(&apply_button);
    header.pack_end(&save_button);

    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&header);

    // ── Restart reminder, revealed after a profile is applied ────────
    let banner = adw::Banner::new("Restart sttd to use the applied profile.");
    banner.set_button_label(Some("Restart"));
    banner.set_revealed(false);
    toolbar.add_top_bar(&banner);

    let profile_group = adw::PreferencesGroup::new();
    profile_group.set_title("Profile");
    let profile_menu = gio::Menu::new();
    profile_menu.append(Some("New Profile…"), Some("win.new-profile"));
    profile_menu.append(Some("Duplicate Profile"), Some("win.duplicate-profile"));
    profile_menu.append(Some("Delete Profile…"), Some("win.delete-profile"));
    let profile_menu_button = gtk4::MenuButton::builder()
        .icon_name("open-menu-symbolic")
        .menu_model(&profile_menu)
        .tooltip_text("Profile actions")
        .valign(gtk4::Align::Center)
        .build();
    profile_menu_button.add_css_class("flat");
    form.profiles.add_suffix(&profile_menu_button);
    profile_group.add(&form.profiles);

    let model_group = adw::PreferencesGroup::new();
    model_group.set_title("Speech Model");
    model_group.set_description(Some(
        "Models are downloaded into the app data directory and verified before installation.",
    ));
    model_group.add(&form.model_picker);
    model_group.add(&form.model_path);
    model_group.add(&form.residency);

    let pp_model_group = adw::PreferencesGroup::new();
    pp_model_group.set_title("Normalization Model");
    pp_model_group.set_description(Some(
        "Downloads are verified and installed under the app data directory.",
    ));
    pp_model_group.add(&form.pp_model_picker);

    let pp_rules_group = adw::PreferencesGroup::new();
    pp_rules_group.set_title("Normalization Rules");
    pp_rules_group.set_description(Some(
        "Control-line values steer the model; only the trained combinations are offered.",
    ));
    pp_rules_group.add(&form.pp_enabled);
    pp_rules_group.add(&form.pp_styling);
    pp_rules_group.add(&form.pp_structure);
    pp_rules_group.add(&form.pp_context);

    let recognition_group = adw::PreferencesGroup::new();
    recognition_group.set_title("Recognition");
    recognition_group.set_description(Some("Set the language hint and inference thread count."));
    recognition_group.add(&form.threads);
    recognition_group.add(&form.language);

    let capture_group = adw::PreferencesGroup::new();
    capture_group.set_title("Capture");
    capture_group.set_description(Some(
        "Choose the input source. Leave Input Device empty for the system default; capture is always converted to mono 16 kHz for the speech pipeline.",
    ));
    let format_row = adw::ActionRow::builder()
        .title("Capture Format")
        .subtitle("Mono, 16 kHz")
        .build();
    format_row.set_activatable(false);
    capture_group.add(&format_row);
    capture_group.add(&form.node_name);

    let detection_group = adw::PreferencesGroup::new();
    detection_group.set_title("Speech Detection");
    detection_group.set_description(Some(
        "Control when speech ends and how audio reaches transcription.",
    ));
    detection_group.add(&form.vad_silence_ms);
    detection_group.add(&form.max_edit_seconds);
    detection_group.add(&form.stt_channel_capacity);

    let overlay_group = adw::PreferencesGroup::new();
    overlay_group.set_title("Desktop Feedback");
    overlay_group.set_description(Some(
        "Configure the live transcription card and system tray.",
    ));
    overlay_group.add(&form.overlay_enabled);
    overlay_group.add(&form.overlay_anchor);
    overlay_group.add(&form.overlay_margin);
    overlay_group.add(&form.crossfade_ms);
    overlay_group.add(&form.tray_enabled);

    let injection_group = adw::PreferencesGroup::new();
    injection_group.set_title("Text Injection");
    injection_group.set_description(Some(
        "Control how transcribed text enters the focused window.",
    ));
    injection_group.add(&form.injection_enabled);
    injection_group.add(&form.batch_mode);
    injection_group.add(&form.char_delay_ms);

    let stack = gtk4::Stack::builder()
        .hexpand(true)
        .vexpand(true)
        .transition_type(gtk4::StackTransitionType::Crossfade)
        .transition_duration(160)
        .build();
    let page_groups = [
        [&model_group, &recognition_group],
        [&pp_model_group, &pp_rules_group],
        [&capture_group, &detection_group],
        [&overlay_group, &injection_group],
    ];
    for (section, groups) in SETTINGS_SECTIONS.iter().zip(page_groups) {
        stack.add_named(&settings_page(section, &groups), Some(section.name));
    }

    let navigation = gtk4::ListBox::new();
    navigation.set_selection_mode(gtk4::SelectionMode::Single);
    navigation.set_activate_on_single_click(true);
    navigation.set_margin_top(12);
    navigation.set_margin_bottom(12);
    navigation.set_margin_start(12);
    navigation.set_margin_end(12);
    navigation.add_css_class("navigation-sidebar");
    for section in &SETTINGS_SECTIONS {
        let row = adw::ActionRow::builder()
            .title(section.title)
            .subtitle(section.sidebar_subtitle)
            .activatable(true)
            .build();
        let icon = gtk4::Image::from_icon_name(section.icon_name);
        row.add_prefix(&icon);
        navigation.append(&row);
    }

    let brand = gtk4::Box::new(gtk4::Orientation::Horizontal, 12);
    brand.set_margin_top(20);
    brand.set_margin_bottom(12);
    brand.set_margin_start(20);
    brand.set_margin_end(20);
    let brand_icon = gtk4::Image::from_icon_name("audio-input-microphone-symbolic");
    brand_icon.set_pixel_size(28);
    let brand_copy = gtk4::Box::new(gtk4::Orientation::Vertical, 2);
    let brand_title = gtk4::Label::new(Some("sttd"));
    brand_title.set_xalign(0.0);
    brand_title.add_css_class("title-3");
    let brand_subtitle = gtk4::Label::new(Some("Speech to text"));
    brand_subtitle.set_xalign(0.0);
    brand_subtitle.add_css_class("dim-label");
    brand_copy.append(&brand_title);
    brand_copy.append(&brand_subtitle);
    brand.append(&brand_icon);
    brand.append(&brand_copy);

    let sidebar = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    sidebar.set_size_request(280, -1);
    sidebar.add_css_class("sidebar");
    sidebar.append(&brand);
    profile_group.set_margin_top(0);
    profile_group.set_margin_bottom(12);
    profile_group.set_margin_start(12);
    profile_group.set_margin_end(12);
    sidebar.append(&profile_group);
    sidebar.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    sidebar.append(&navigation);
    let sidebar_spacer = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    sidebar_spacer.set_vexpand(true);
    sidebar.append(&sidebar_spacer);

    let split_view = adw::OverlaySplitView::builder()
        .sidebar(&sidebar)
        .content(&stack)
        .min_sidebar_width(280.0)
        .max_sidebar_width(280.0)
        .pin_sidebar(true)
        .show_sidebar(true)
        .build();
    toolbar.set_content(Some(&split_view));

    // Below tablet width the sidebar becomes a dismissible overlay.
    let narrow = adw::Breakpoint::new(adw::BreakpointCondition::new_length(
        adw::BreakpointConditionLengthType::MaxWidth,
        760.0,
        adw::LengthUnit::Px,
    ));
    narrow.add_setter(&split_view, "collapsed", Some(&true.to_value()));
    narrow.add_setter(&split_view, "pin-sidebar", Some(&false.to_value()));
    narrow.add_setter(&split_view, "show-sidebar", Some(&false.to_value()));
    window.add_breakpoint(narrow);

    split_view
        .bind_property("collapsed", &sidebar_button, "visible")
        .sync_create()
        .build();
    {
        let split_view = split_view.clone();
        sidebar_button.connect_clicked(move |_| {
            split_view.set_show_sidebar(!split_view.shows_sidebar());
        });
    }
    {
        let split_view = split_view.clone();
        let window_title = window_title.clone();
        let stack = stack.clone();
        navigation.connect_row_selected(move |_, row| {
            let Some(section) = row
                .and_then(|row| usize::try_from(row.index()).ok())
                .and_then(|index| SETTINGS_SECTIONS.get(index))
            else {
                return;
            };
            stack.set_visible_child_name(section.name);
            window_title.set_title(section.title);
            if split_view.is_collapsed() {
                split_view.set_show_sidebar(false);
            }
        });
    }
    if let Some(first_row) = navigation.row_at_index(0) {
        navigation.select_row(Some(&first_row));
    }

    let toast_overlay = adw::ToastOverlay::new();
    toast_overlay.set_child(Some(&toolbar));
    window.set_content(Some(&toast_overlay));

    {
        let overlay = toast_overlay.clone();
        let banner = banner.clone();
        banner.connect_button_clicked(move |banner| {
            let output = std::process::Command::new("systemctl")
                .args(["--user", "restart", "sttd"])
                .stderr(std::process::Stdio::piped())
                .output();
            match output {
                Ok(out) if out.status.success() => {
                    toast(&overlay, "Restarted sttd.");
                    banner.set_revealed(false);
                }
                Ok(out) => {
                    let msg = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    toast_error(&overlay, &format!("Could not restart sttd: {msg}"));
                }
                Err(e) => {
                    toast_error(&overlay, &format!("Could not restart sttd: {e}"));
                }
            }
        });
    }

    connect_signals(
        &window,
        &toast_overlay,
        &banner,
        &form,
        &state,
        &save_button,
        &apply_button,
        &form.model_action_button,
        &form.pp_model_action_button,
    );

    refresh_profiles(&form, &state);
    window.present();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn connect_signals(
    window: &adw::ApplicationWindow,
    overlay: &adw::ToastOverlay,
    banner: &adw::Banner,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
    save_button: &gtk4::Button,
    apply_button: &gtk4::Button,
    model_action_button: &gtk4::Button,
    pp_model_action_button: &gtk4::Button,
) {
    // Selecting a profile loads it into the form.
    {
        let window = window.clone();
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let overlay = overlay.clone();
        let profiles = form.profiles.clone();
        profiles.connect_selected_notify(move |row| {
            if form.updating.get() {
                return;
            }
            let Some(name) = row
                .selected_item()
                .and_downcast::<gtk4::StringObject>()
                .map(|item| item.string().to_string())
            else {
                return;
            };
            request_profile_switch(&window, &form, &state, &overlay, &name);
        });
    }

    // Picking an installed model selects it; an uninstalled catalog model stays
    // selected so its download action remains available.
    {
        let form = Rc::clone(form);
        let picker = form.model_picker.clone();
        picker.connect_selected_notify(move |row| {
            if form.updating.get() {
                return;
            }
            let index = row.selected() as usize;
            let choice = form.model_choices.borrow().get(index).cloned();
            match choice {
                Some(ModelChoice::Catalog(spec)) => {
                    if let Some(path) = managed_model_path(&default_models_dir(), spec) {
                        let text = path.to_string_lossy().into_owned();
                        if form.model_path.text().as_str() != text {
                            form.model_path.set_text(&text);
                        }
                    }
                    // Fill the language hint only when the user has not set
                    // one; changing models must not silently rewrite it.
                    if form.language.text().trim().is_empty() {
                        form.language.set_text(spec.default_language);
                    }
                }
                Some(ModelChoice::Custom(path)) => {
                    let text = path.to_string_lossy().into_owned();
                    if form.model_path.text().as_str() != text {
                        form.model_path.set_text(&text);
                    }
                }
                None => {}
            }
            update_model_manager(&form);
        });
    }

    // Typing a path that matches a catalog or discovered custom model selects it.
    {
        let form = Rc::clone(form);
        let entry = form.model_path.clone();
        entry.connect_changed(move |row| {
            if form.updating.get() {
                return;
            }
            let text = row.text();
            let choices = form.model_choices.borrow();
            let index = choices.iter().position(|choice| {
                choice
                    .path()
                    .is_some_and(|path| path.to_string_lossy() == text.as_str())
            });
            let selected = index.map_or(gtk4::INVALID_LIST_POSITION, |index| index as u32);
            if form.model_picker.selected() != selected {
                form.model_picker.set_selected(selected);
            }
            update_model_manager(&form);
        });
    }

    {
        let window = window.clone();
        let overlay = overlay.clone();
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let model_action_button = model_action_button.clone();
        model_action_button.connect_clicked(move |_| {
            manage_selected_model(&window, &overlay, &form, &state);
        });
    }

    // Post-processing model management (download / cancel / delete).
    {
        let window = window.clone();
        let overlay = overlay.clone();
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let pp_model_action_button = pp_model_action_button.clone();
        pp_model_action_button.connect_clicked(move |_| {
            manage_selected_pp_model(&window, &overlay, &form, &state);
        });
    }

    // Keep the install-state subtitle fresh while picking or toggling.
    {
        let form = Rc::clone(form);
        let picker = form.pp_model_picker.clone();
        picker.connect_selected_notify(move |_| {
            if form.updating.get() {
                return;
            }
            update_pp_model_manager(&form);
        });
    }
    {
        let form = Rc::clone(form);
        let toggle = form.pp_enabled.clone();
        toggle.connect_active_notify(move |_| {
            if form.updating.get() {
                return;
            }
            sync_pp_controls(&form);
            update_pp_model_manager(&form);
        });
    }
    {
        let form = Rc::clone(form);
        let toggle = form.injection_enabled.clone();
        toggle.connect_active_notify(move |_| sync_injection_controls(&form));
    }
    {
        let form = Rc::clone(form);
        let toggle = form.batch_mode.clone();
        toggle.connect_active_notify(move |_| sync_injection_controls(&form));
    }
    {
        let form = Rc::clone(form);
        let toggle = form.overlay_enabled.clone();
        toggle.connect_active_notify(move |_| sync_overlay_controls(&form));
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let overlay = overlay.clone();
        save_button.connect_clicked(move |_| {
            if !selected_model_is_ready(&form, &overlay) {
                return;
            }
            let name = state.borrow().current.clone();
            let cfg = collect_config(&state, &form);
            let store = state.borrow().store.clone();
            match store.save(&name, &cfg) {
                Ok(()) => {
                    state.borrow_mut().config = cfg;
                    toast(&overlay, &format!("Saved '{name}'."));
                }
                Err(error) => {
                    toast_error(&overlay, &format!("Could not save profile: {error:#}"));
                }
            }
        });
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let overlay = overlay.clone();
        let banner = banner.clone();
        apply_button.connect_clicked(move |_| {
            if !selected_model_is_ready(&form, &overlay) {
                return;
            }
            let name = state.borrow().current.clone();
            let cfg = collect_config(&state, &form);
            let store = state.borrow().store.clone();
            let result = store.save(&name, &cfg).and_then(|()| store.activate(&name));
            match result {
                Ok(()) => {
                    {
                        let mut editor = state.borrow_mut();
                        editor.config = cfg;
                        editor.active = Some(name.clone());
                    }
                    refresh_profiles(&form, &state);
                    banner.set_title(&format!("Applied '{name}'. Restart sttd to use it."));
                    banner.set_revealed(true);
                }
                Err(error) => {
                    toast_error(&overlay, &format!("Could not apply profile: {error:#}"));
                }
            }
        });
    }

    // Profile actions, registered on the window so the menu model reaches them.
    let actions = gio::SimpleActionGroup::new();
    window.insert_action_group("win", Some(&actions));

    {
        let window = window.clone();
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let overlay = overlay.clone();
        let action = gio::SimpleAction::new("new-profile", None);
        action.connect_activate(move |_, _| {
            prompt_new_profile(&window, &overlay, &form, &state);
        });
        actions.add_action(&action);
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let overlay = overlay.clone();
        let action = gio::SimpleAction::new("duplicate-profile", None);
        action.connect_activate(move |_, _| {
            if !selected_model_is_ready(&form, &overlay) {
                return;
            }
            let store = state.borrow().store.clone();
            let base = format!("{} Copy", state.borrow().current);
            let name = unique_profile_name(&store, &base);
            let cfg = collect_config(&state, &form);
            match store.save(&name, &cfg) {
                Ok(()) => {
                    {
                        let mut editor = state.borrow_mut();
                        editor.current = name.clone();
                        editor.config = cfg;
                    }
                    refresh_profiles(&form, &state);
                    toast(&overlay, &format!("Duplicated profile as '{name}'."));
                }
                Err(error) => {
                    toast_error(&overlay, &format!("Could not duplicate profile: {error:#}"))
                }
            }
        });
        actions.add_action(&action);
    }

    {
        let window = window.clone();
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let overlay = overlay.clone();
        let action = gio::SimpleAction::new("delete-profile", None);
        action.connect_activate(move |_, _| {
            confirm_delete_profile(&window, &overlay, &form, &state);
        });
        actions.add_action(&action);
    }
}

fn prompt_new_profile(
    window: &adw::ApplicationWindow,
    overlay: &adw::ToastOverlay,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
) {
    if !selected_model_is_ready(form, overlay) {
        return;
    }
    let dialog = adw::AlertDialog::new(
        Some("New Profile"),
        Some("Save the current settings as a new profile."),
    );
    let entry = gtk4::Entry::builder()
        .placeholder_text("Profile name")
        .activates_default(true)
        .build();
    dialog.set_extra_child(Some(&entry));
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("create", "Create");
    dialog.set_response_appearance("create", adw::ResponseAppearance::Suggested);
    dialog.set_default_response(Some("create"));
    dialog.set_close_response("cancel");

    let form = Rc::clone(form);
    let state = Rc::clone(state);
    let overlay = overlay.clone();
    dialog.connect_response(None, move |_, response| {
        if response != "create" {
            return;
        }
        let name = entry.text();
        create_profile(&overlay, &form, &state, name.trim());
    });
    dialog.present(Some(window));
}

fn create_profile(
    overlay: &adw::ToastOverlay,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
    name: &str,
) {
    if name.is_empty() {
        toast_error(overlay, "Profile name cannot be empty.");
        return;
    }
    let store = state.borrow().store.clone();
    if store
        .list()
        .map(|names| names.iter().any(|existing| existing == name))
        .unwrap_or(false)
    {
        toast_error(
            overlay,
            &format!("A profile named '{name}' already exists."),
        );
        return;
    }
    let cfg = collect_config(state, form);
    match store.save(name, &cfg) {
        Ok(()) => {
            {
                let mut editor = state.borrow_mut();
                editor.current = name.to_string();
                editor.config = cfg;
            }
            refresh_profiles(form, state);
            toast(overlay, &format!("Created profile '{name}'."));
        }
        Err(error) => toast_error(overlay, &format!("Could not create profile: {error:#}")),
    }
}

fn confirm_delete_profile(
    window: &adw::ApplicationWindow,
    overlay: &adw::ToastOverlay,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
) {
    let name = state.borrow().current.clone();
    let heading = format!("Delete Profile '{name}'?");
    let dialog = adw::AlertDialog::new(
        Some(heading.as_str()),
        Some("This permanently removes the profile from disk. Unsaved edits to it are lost."),
    );
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("delete", "Delete");
    dialog.set_response_appearance("delete", adw::ResponseAppearance::Destructive);
    dialog.set_default_response(Some("cancel"));
    dialog.set_close_response("cancel");

    let form = Rc::clone(form);
    let state = Rc::clone(state);
    let overlay = overlay.clone();
    dialog.connect_response(None, move |_, response| {
        if response != "delete" {
            return;
        }
        let store = state.borrow().store.clone();
        match store.delete(&name) {
            Ok(()) => {
                let next = store
                    .active()
                    .ok()
                    .flatten()
                    .or_else(|| store.list().ok().and_then(|names| names.into_iter().next()));
                if let Some(next) = next {
                    if let Ok(cfg) = store.load(&next) {
                        {
                            let mut editor = state.borrow_mut();
                            editor.current = next;
                            editor.config = cfg.clone();
                            editor.active = store.active().ok().flatten();
                        }
                        populate_form(&form, &cfg);
                        refresh_models(&form, &cfg);
                        refresh_profiles(&form, &state);
                    }
                }
                toast(&overlay, "Profile deleted.");
            }
            Err(error) => {
                toast_error(&overlay, &format!("Could not delete profile: {error:#}"));
            }
        }
    });
    dialog.present(Some(window));
}

fn has_unsaved_changes(state: &Rc<RefCell<EditorState>>, form: &Form) -> bool {
    let edited = collect_config(state, form);
    let saved = state.borrow().config.clone();
    match (toml::to_string(&edited), toml::to_string(&saved)) {
        (Ok(edited), Ok(saved)) => edited != saved,
        _ => true,
    }
}

fn revert_profile_selection(form: &Form, state: &Rc<RefCell<EditorState>>) {
    let current = state.borrow().current.clone();
    let store = state.borrow().store.clone();
    let Ok(names) = store.list() else {
        return;
    };
    let Some(index) = names.iter().position(|name| name == &current) else {
        return;
    };
    let _guard = UpdatingGuard::new(form);
    form.profiles.set_selected(index as u32);
}

fn save_current_config(form: &Rc<Form>, state: &Rc<RefCell<EditorState>>) -> Result<()> {
    let name = state.borrow().current.clone();
    let cfg = collect_config(state, form);
    let store = state.borrow().store.clone();
    store.save(&name, &cfg)?;
    state.borrow_mut().config = cfg;
    Ok(())
}

fn request_profile_switch(
    window: &adw::ApplicationWindow,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
    overlay: &adw::ToastOverlay,
    name: &str,
) {
    let current = state.borrow().current.clone();
    if name == current {
        return;
    }

    if !has_unsaved_changes(state, form) {
        load_profile(form, state, overlay, name);
        return;
    }

    let dialog = adw::AlertDialog::new(
        Some("Unsaved Changes"),
        Some(&format!(
            "Save the changes to '{current}' before switching to '{name}'?"
        )),
    );
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("discard", "Discard");
    dialog.add_response("save", "Save");
    dialog.set_response_appearance("discard", adw::ResponseAppearance::Destructive);
    dialog.set_response_appearance("save", adw::ResponseAppearance::Suggested);
    dialog.set_default_response(Some("save"));
    dialog.set_close_response("cancel");

    let form = Rc::clone(form);
    let state = Rc::clone(state);
    let overlay = overlay.clone();
    let name = name.to_string();
    dialog.connect_response(None, move |_, response| match response {
        "save" => {
            if !selected_model_is_ready(&form, &overlay) {
                revert_profile_selection(&form, &state);
                return;
            }
            match save_current_config(&form, &state) {
                Ok(()) => load_profile(&form, &state, &overlay, &name),
                Err(error) => {
                    toast_error(&overlay, &format!("Could not save profile: {error:#}"));
                    revert_profile_selection(&form, &state);
                }
            }
        }
        "discard" => load_profile(&form, &state, &overlay, &name),
        _ => revert_profile_selection(&form, &state),
    });
    dialog.present(Some(window));
}

fn load_profile(
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
    overlay: &adw::ToastOverlay,
    name: &str,
) {
    let store = state.borrow().store.clone();
    match store.load(name) {
        Ok(cfg) => {
            {
                let mut editor = state.borrow_mut();
                editor.current = name.to_string();
                editor.config = cfg.clone();
            }
            populate_form(form, &cfg);
            refresh_models(form, &cfg);
            toast(overlay, &format!("Loaded '{name}'."));
        }
        Err(error) => toast_error(overlay, &format!("Could not load profile: {error:#}")),
    }
}

fn refresh_profiles(form: &Rc<Form>, state: &Rc<RefCell<EditorState>>) {
    let (store, current, active) = {
        let editor = state.borrow();
        (
            editor.store.clone(),
            editor.current.clone(),
            editor.active.clone(),
        )
    };
    let names = match store.list() {
        Ok(names) => names,
        Err(error) => {
            form.profiles
                .set_subtitle(&format!("Could not list profiles: {error:#}"));
            return;
        }
    };

    let _guard = UpdatingGuard::new(form);
    let model = gtk4::StringList::new(&[]);
    for name in &names {
        model.append(name);
    }
    form.profiles.set_model(Some(&model));
    let index = names.iter().position(|name| name == &current);
    if let Some(index) = index {
        form.profiles.set_selected(index as u32);
    } else {
        form.profiles.set_selected(0);
        if let Some(fallback) = names.first() {
            state.borrow_mut().current = fallback.clone();
        }
    }

    form.profiles.set_subtitle(
        &active
            .map(|name| format!("Active profile: {name}"))
            .unwrap_or_else(|| "No active profile".into()),
    );
}

fn refresh_models(form: &Form, cfg: &Config) {
    let current = cfg.model.path.clone();
    let models_dir = default_models_dir();
    let mut custom_paths = BTreeSet::new();
    let mut dirs = vec![default_models_dir()];
    if let Some(parent) = Path::new(&current).parent() {
        dirs.push(parent.to_path_buf());
    }

    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && is_model_file(&path) && !is_catalog_path(&models_dir, &path) {
                custom_paths.insert(path);
            }
        }
    }
    let current_path = PathBuf::from(&current);
    if !current.is_empty() && !is_catalog_path(&models_dir, &current_path) {
        custom_paths.insert(current_path.clone());
    }

    let _guard = UpdatingGuard::new(form);
    form.model_path.set_text(&current);
    let model = gtk4::StringList::new(&[]);
    let mut choices = Vec::with_capacity(MODEL_CATALOG.len() + custom_paths.len());
    for spec in MODEL_CATALOG {
        model.append(&catalog_label(spec, install_state(&models_dir, spec)));
        choices.push(ModelChoice::Catalog(spec));
    }
    for path in custom_paths {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        let status = if path.is_file() {
            "Custom"
        } else {
            "Custom, missing"
        };
        model.append(&format!("{name} ({status})"));
        choices.push(ModelChoice::Custom(path));
    }
    form.model_picker.set_model(Some(&model));
    *form.model_choices.borrow_mut() = choices;

    let selected = form
        .model_choices
        .borrow()
        .iter()
        .position(|choice| {
            choice
                .path()
                .is_some_and(|path| path.to_string_lossy().as_ref() == current)
        })
        .map_or(gtk4::INVALID_LIST_POSITION, |index| index as u32);
    form.model_picker.set_selected(selected);
    drop(_guard);
    update_model_manager(form);
}

fn is_model_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("bin") || extension.eq_ignore_ascii_case("gguf")
        })
}

fn is_catalog_path(models_dir: &Path, path: &Path) -> bool {
    path.parent() == Some(models_dir)
        && path
            .file_name()
            .and_then(|filename| filename.to_str())
            .and_then(model_for_filename)
            .is_some()
}

fn catalog_label(spec: &ModelSpec, install: InstallState) -> String {
    let recommendation = if spec.recommended {
        "Recommended, "
    } else {
        ""
    };
    let status = match install {
        InstallState::Installed => "Installed",
        InstallState::Missing => "Download",
        InstallState::Invalid => "Invalid file",
    };
    format!(
        "{} ({}{} - {})",
        spec.name,
        recommendation,
        status,
        format_size(spec.size_bytes)
    )
}

fn selected_model_choice(form: &Form) -> Option<ModelChoice> {
    let selected = form.model_picker.selected();
    if selected == gtk4::INVALID_LIST_POSITION {
        return None;
    }
    form.model_choices.borrow().get(selected as usize).cloned()
}

fn selected_model_is_ready(form: &Form, overlay: &adw::ToastOverlay) -> bool {
    if form.model_busy.get() {
        toast_error(
            overlay,
            "Wait for the model download to finish or cancel it first.",
        );
        return false;
    }
    if form.pp_model_busy.get() {
        toast_error(
            overlay,
            "Wait for the post-processing model download to finish or cancel it first.",
        );
        return false;
    }
    if form.pp_enabled.is_active() {
        let ready = selected_pp_spec(form).is_some_and(|spec| {
            post_process_install_state(&default_models_dir(), spec) == InstallState::Installed
        });
        if !ready {
            toast_error(
                overlay,
                "Download the selected post-processing model (or disable post-processing) before saving this profile.",
            );
            return false;
        }
    }
    match selected_model_choice(form) {
        Some(ModelChoice::Catalog(spec)) => match install_state(&default_models_dir(), spec) {
            InstallState::Installed => true,
            InstallState::Missing => {
                toast_error(
                    overlay,
                    "Download the selected model before saving this profile.",
                );
                false
            }
            InstallState::Invalid => {
                toast_error(
                    overlay,
                    "Delete the invalid model file, then download it again before saving.",
                );
                false
            }
        },
        Some(ModelChoice::Custom(path)) if path.is_file() => true,
        Some(ModelChoice::Custom(_)) => {
            toast_error(overlay, "The selected custom model file no longer exists.");
            false
        }
        None => {
            let typed = form.model_path.text();
            let path = Path::new(typed.trim());
            if !typed.trim().is_empty() && path.is_file() {
                true
            } else if typed.trim().is_empty() {
                toast_error(
                    overlay,
                    "Select an installed model or enter a model file path before saving this profile.",
                );
                false
            } else {
                toast_error(
                    overlay,
                    "The model file path does not point to an existing file.",
                );
                false
            }
        }
    }
}

fn update_model_manager(form: &Form) {
    if form.model_busy.get() {
        return;
    }

    form.model_progress.set_opacity(0.0);
    form.model_progress.set_text(None);
    match selected_model_choice(form) {
        Some(ModelChoice::Catalog(spec)) => {
            let install = install_state(&default_models_dir(), spec);
            let status = match install {
                InstallState::Installed => "Installed",
                InstallState::Missing => "Download required",
                InstallState::Invalid => "Invalid file - delete before downloading",
            };
            form.model_picker.set_subtitle(&format!(
                "{} | {} | {} | {}",
                spec.family,
                spec.languages,
                format_size(spec.size_bytes),
                status
            ));
            form.model_picker.set_tooltip_text(Some(&format!(
                "{} {} License: {}.",
                spec.description,
                if spec.streaming {
                    "Streaming-capable architecture."
                } else {
                    "Utterance transcription."
                },
                spec.license
            )));
            form.model_action_button.set_visible(true);
            form.model_action_button.set_sensitive(true);
            if install != InstallState::Missing {
                form.model_action_button
                    .set_icon_name("user-trash-symbolic");
                form.model_action_button
                    .set_tooltip_text(Some("Delete selected model"));
                form.model_action_button.add_css_class("destructive-action");
            } else {
                form.model_action_button
                    .set_icon_name("folder-download-symbolic");
                form.model_action_button
                    .set_tooltip_text(Some("Download selected model"));
                form.model_action_button
                    .remove_css_class("destructive-action");
            }
        }
        Some(ModelChoice::Custom(path)) => {
            let status = if path.is_file() {
                "Installed"
            } else {
                "Missing"
            };
            form.model_picker
                .set_subtitle(&format!("Custom local model | {status}"));
            form.model_picker
                .set_tooltip_text(Some(path.display().to_string().as_str()));
            form.model_action_button.set_visible(false);
        }
        None => {
            let typed = form.model_path.text();
            let path = Path::new(typed.trim());
            if typed.trim().is_empty() {
                form.model_picker
                    .set_subtitle("Select an installed model or one to download");
                form.model_picker.set_tooltip_text(None);
            } else {
                let status = if path.is_file() {
                    "Installed"
                } else {
                    "Missing"
                };
                form.model_picker
                    .set_subtitle(&format!("Custom local model | {status}"));
                form.model_picker.set_tooltip_text(Some(typed.as_str()));
            }
            form.model_action_button.set_visible(false);
        }
    }
}

fn manage_selected_model(
    window: &adw::ApplicationWindow,
    overlay: &adw::ToastOverlay,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
) {
    let Some(ModelChoice::Catalog(spec)) = selected_model_choice(form) else {
        return;
    };
    if form.model_busy.get() {
        if let Some(cancel) = form.model_cancel.borrow().as_ref() {
            cancel.store(true, Ordering::Release);
            form.model_action_button.set_sensitive(false);
            form.model_progress.set_text(Some("Cancelling"));
        }
        return;
    }
    match install_state(&default_models_dir(), spec) {
        InstallState::Installed | InstallState::Invalid => {
            confirm_delete_model(window, overlay, form, state, spec)
        }
        InstallState::Missing => start_model_download(window, overlay, form, state, spec),
    }
}

fn start_model_download(
    window: &adw::ApplicationWindow,
    overlay: &adw::ToastOverlay,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
    spec: &'static ModelSpec,
) {
    if form.model_busy.replace(true) {
        return;
    }

    form.model_picker.set_activatable(false);
    form.profiles.set_sensitive(false);
    window.set_deletable(false);
    form.model_picker
        .set_subtitle(&format!("Downloading {}", spec.name));
    form.model_action_button
        .set_icon_name("process-stop-symbolic");
    form.model_action_button
        .set_tooltip_text(Some("Cancel model download"));
    form.model_action_button
        .remove_css_class("destructive-action");
    form.model_action_button.set_sensitive(true);
    form.model_action_button.set_visible(true);
    form.model_progress.set_fraction(0.0);
    form.model_progress.set_text(Some("Starting"));
    form.model_progress.set_opacity(1.0);

    let (sender, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    *form.model_cancel.borrow_mut() = Some(Arc::clone(&cancel));
    let models_dir = default_models_dir();
    let worker = thread::Builder::new()
        .name(format!("sttd-model-{}", spec.id))
        .spawn(move || {
            let progress_sender = sender.clone();
            let result = download_model(&models_dir, spec, &cancel, move |downloaded, total| {
                let _ = progress_sender.send(DownloadEvent::Progress { downloaded, total });
            })
            .map_err(|error| format!("{error:#}"));
            let _ = sender.send(DownloadEvent::Finished(result));
        });

    if let Err(error) = worker {
        form.model_busy.set(false);
        form.model_cancel.borrow_mut().take();
        form.model_picker.set_activatable(true);
        form.profiles.set_sensitive(true);
        window.set_deletable(true);
        update_model_manager(form);
        toast_error(overlay, &format!("Could not start model download: {error}"));
        return;
    }

    let overlay = overlay.clone();
    let window = window.clone();
    let form = Rc::clone(form);
    let state = Rc::clone(state);
    glib::timeout_add_local(Duration::from_millis(100), move || loop {
        match receiver.try_recv() {
            Ok(DownloadEvent::Progress { downloaded, total }) => {
                let fraction = if total == 0 {
                    0.0
                } else {
                    downloaded.min(total) as f64 / total as f64
                };
                form.model_progress.set_fraction(fraction);
                form.model_progress
                    .set_text(Some(&format!("{:.0}%", fraction * 100.0)));
            }
            Ok(DownloadEvent::Finished(result)) => {
                form.model_busy.set(false);
                let was_cancelled = form
                    .model_cancel
                    .borrow_mut()
                    .take()
                    .is_some_and(|cancel| cancel.load(Ordering::Acquire));
                form.model_picker.set_activatable(true);
                form.profiles.set_sensitive(true);
                window.set_deletable(true);
                match result {
                    Ok(path) => {
                        form.model_path.set_text(path.to_string_lossy().as_ref());
                        let mut cfg = state.borrow().config.clone();
                        cfg.model.path = path.to_string_lossy().into_owned();
                        refresh_models(&form, &cfg);
                        toast(&overlay, &format!("Installed {}.", spec.name));
                    }
                    Err(error) => {
                        update_model_manager(&form);
                        if was_cancelled {
                            toast(&overlay, "Model download cancelled.");
                        } else {
                            toast_error(&overlay, &format!("Model download failed: {error}"));
                        }
                    }
                }
                return glib::ControlFlow::Break;
            }
            Err(mpsc::TryRecvError::Empty) => return glib::ControlFlow::Continue,
            Err(mpsc::TryRecvError::Disconnected) => {
                form.model_busy.set(false);
                form.model_cancel.borrow_mut().take();
                form.model_picker.set_activatable(true);
                form.profiles.set_sensitive(true);
                window.set_deletable(true);
                update_model_manager(&form);
                toast_error(&overlay, "Model download stopped unexpectedly.");
                return glib::ControlFlow::Break;
            }
        }
    });
}

fn confirm_delete_model(
    window: &adw::ApplicationWindow,
    overlay: &adw::ToastOverlay,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
    spec: &'static ModelSpec,
) {
    let models_dir = default_models_dir();
    let Some(target) = managed_model_path(&models_dir, spec) else {
        toast_error(
            overlay,
            "The selected model has an invalid managed filename.",
        );
        return;
    };
    match profiles_using_model(state, &target) {
        Ok(profiles) if !profiles.is_empty() => {
            toast_error(
                overlay,
                &format!(
                    "{} is used by: {}. Select another installed model in those profiles and save before deleting.",
                    spec.name,
                    profiles.join(", ")
                ),
            );
            return;
        }
        Err(error) => {
            toast_error(
                overlay,
                &format!("Could not check model usage before deletion: {error:#}"),
            );
            return;
        }
        Ok(_) => {}
    }

    let dialog = adw::AlertDialog::new(
        Some(&format!("Delete {}?", spec.name)),
        Some("The verified model file will be removed from the app's model directory."),
    );
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("delete", "Delete");
    dialog.set_response_appearance("delete", adw::ResponseAppearance::Destructive);
    dialog.set_default_response(Some("cancel"));
    dialog.set_close_response("cancel");

    let form = Rc::clone(form);
    let state = Rc::clone(state);
    let overlay = overlay.clone();
    let target = target.clone();
    dialog.connect_response(None, move |_, response| {
        if response != "delete" || form.model_busy.get() {
            return;
        }
        let models_dir = default_models_dir();
        match delete_managed_model(&models_dir, &target) {
            Ok(()) => {
                let current = PathBuf::from(form.model_path.text().as_str());
                if current == target {
                    let saved = PathBuf::from(&state.borrow().config.model.path);
                    let fallback = if saved != target && saved.is_file() {
                        Some(saved)
                    } else {
                        MODEL_CATALOG.iter().find_map(|candidate| {
                            let path = managed_model_path(&models_dir, candidate)?;
                            (path != target
                                && install_state(&models_dir, candidate) == InstallState::Installed)
                                .then_some(path)
                        })
                    };
                    if let Some(path) = fallback {
                        form.model_path.set_text(path.to_string_lossy().as_ref());
                        if let Some(candidate) = MODEL_CATALOG.iter().find(|candidate| {
                            managed_model_path(&models_dir, candidate).as_ref() == Some(&path)
                        }) {
                            if form.language.text().trim().is_empty() {
                                form.language.set_text(candidate.default_language);
                            }
                        }
                    } else {
                        form.model_path.set_text("");
                    }
                }
                let mut cfg = state.borrow().config.clone();
                cfg.model.path = form.model_path.text().to_string();
                refresh_models(&form, &cfg);
                toast(&overlay, &format!("Deleted {}.", spec.name));
            }
            Err(error) => toast_error(
                &overlay,
                &format!("Could not delete {}: {error:#}", spec.name),
            ),
        }
    });
    dialog.present(Some(window));
}

fn profiles_using_model(state: &Rc<RefCell<EditorState>>, target: &Path) -> Result<Vec<String>> {
    let store = state.borrow().store.clone();
    let mut profiles = Vec::new();
    for name in store.list().context("listing profiles")? {
        let cfg = store
            .load(&name)
            .with_context(|| format!("loading profile '{name}'"))?;
        if cfg.model.model_path() == target {
            profiles.push(name);
        }
    }
    Ok(profiles)
}

// ─── Post-processing model management ───────────────────────────────────────

fn refresh_pp_models(form: &Form, cfg: &Config) {
    let guard = UpdatingGuard::new(form);
    let models_dir = default_models_dir();
    let list = gtk4::StringList::new(&[]);
    let mut choices = Vec::with_capacity(POST_PROCESS_CATALOG.len());
    for spec in POST_PROCESS_CATALOG {
        list.append(&pp_catalog_label(
            spec,
            post_process_install_state(&models_dir, spec),
        ));
        choices.push(spec);
    }
    form.pp_model_picker.set_model(Some(&list));
    *form.pp_model_choices.borrow_mut() = choices;

    let selected = form
        .pp_model_choices
        .borrow()
        .iter()
        .position(|spec| spec.id == cfg.post_process.model)
        .map_or(gtk4::INVALID_LIST_POSITION, |index| index as u32);
    form.pp_model_picker.set_selected(selected);
    drop(guard);
    update_pp_model_manager(form);
}

fn selected_pp_spec(form: &Form) -> Option<&'static model_catalog::PostProcessModelSpec> {
    let selected = form.pp_model_picker.selected();
    if selected == gtk4::INVALID_LIST_POSITION {
        return None;
    }
    form.pp_model_choices.borrow().get(selected as usize).copied()
}

fn pp_catalog_label(
    spec: &model_catalog::PostProcessModelSpec,
    install: InstallState,
) -> String {
    let recommendation = if spec.recommended {
        "Recommended, "
    } else {
        ""
    };
    let status = match install {
        InstallState::Installed => "Installed",
        InstallState::Missing => "Download",
        InstallState::Invalid => "Invalid file",
    };
    format!(
        "{} ({}{} - {})",
        spec.name,
        recommendation,
        status,
        format_size(spec.size_bytes)
    )
}

fn update_pp_model_manager(form: &Form) {
    if form.pp_model_busy.get() {
        return;
    }

    form.pp_model_progress.set_opacity(0.0);
    form.pp_model_progress.set_text(None);
    match selected_pp_spec(form) {
        Some(spec) => {
            let install = post_process_install_state(&default_models_dir(), spec);
            let total = format_size(spec.size_bytes + spec.tokenizer_size_bytes);
            let status = match install {
                InstallState::Installed => "Installed",
                InstallState::Missing => "Download required",
                InstallState::Invalid => "Invalid file - delete before downloading",
            };
            form.pp_model_picker.set_subtitle(&format!(
                "{} | {} | {} | {}",
                spec.family, spec.languages, total, status
            ));
            form.pp_model_picker.set_tooltip_text(Some(&format!(
                "{} License: {}.",
                spec.description, spec.license
            )));
            form.pp_model_action_button.set_visible(true);
            form.pp_model_action_button.set_sensitive(true);
            if install != InstallState::Missing {
                form.pp_model_action_button
                    .set_icon_name("user-trash-symbolic");
                form.pp_model_action_button
                    .set_tooltip_text(Some("Delete selected model"));
                form.pp_model_action_button.add_css_class("destructive-action");
            } else {
                form.pp_model_action_button
                    .set_icon_name("folder-download-symbolic");
                form.pp_model_action_button
                    .set_tooltip_text(Some("Download selected model"));
                form.pp_model_action_button
                    .remove_css_class("destructive-action");
            }
        }
        None => {
            form.pp_model_picker
                .set_subtitle("Select an installed model or one to download");
            form.pp_model_picker.set_tooltip_text(None);
            form.pp_model_action_button.set_visible(false);
        }
    }
}

fn manage_selected_pp_model(
    window: &adw::ApplicationWindow,
    overlay: &adw::ToastOverlay,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
) {
    let Some(spec) = selected_pp_spec(form) else {
        return;
    };
    if form.pp_model_busy.get() {
        if let Some(cancel) = form.pp_model_cancel.borrow().as_ref() {
            cancel.store(true, Ordering::Release);
            form.pp_model_action_button.set_sensitive(false);
            form.pp_model_progress.set_text(Some("Cancelling"));
        }
        return;
    }
    match post_process_install_state(&default_models_dir(), spec) {
        InstallState::Installed | InstallState::Invalid => {
            confirm_delete_pp_model(window, overlay, form, state, spec)
        }
        InstallState::Missing => start_pp_model_download(window, overlay, form, state, spec),
    }
}

fn start_pp_model_download(
    window: &adw::ApplicationWindow,
    overlay: &adw::ToastOverlay,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
    spec: &'static model_catalog::PostProcessModelSpec,
) {
    if form.pp_model_busy.replace(true) {
        return;
    }

    form.pp_model_picker.set_activatable(false);
    form.profiles.set_sensitive(false);
    window.set_deletable(false);
    form.pp_model_picker
        .set_subtitle(&format!("Downloading {}", spec.name));
    form.pp_model_action_button
        .set_icon_name("process-stop-symbolic");
    form.pp_model_action_button
        .set_tooltip_text(Some("Cancel model download"));
    form.pp_model_action_button
        .remove_css_class("destructive-action");
    form.pp_model_action_button.set_sensitive(true);
    form.pp_model_action_button.set_visible(true);
    form.pp_model_progress.set_fraction(0.0);
    form.pp_model_progress.set_text(Some("Starting"));
    form.pp_model_progress.set_opacity(1.0);

    let (sender, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    *form.pp_model_cancel.borrow_mut() = Some(Arc::clone(&cancel));
    let models_dir = default_models_dir();
    let worker = thread::Builder::new()
        .name(format!("sttd-post-process-{}", spec.id))
        .spawn(move || {
            let progress_sender = sender.clone();
            let result =
                download_post_process_model(&models_dir, spec, &cancel, move |downloaded, total| {
                    let _ = progress_sender.send(DownloadEvent::Progress { downloaded, total });
                })
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(DownloadEvent::Finished(result));
        });

    if let Err(error) = worker {
        form.pp_model_busy.set(false);
        form.pp_model_cancel.borrow_mut().take();
        form.pp_model_picker.set_activatable(true);
        form.profiles.set_sensitive(true);
        window.set_deletable(true);
        update_pp_model_manager(form);
        toast_error(overlay, &format!("Could not start model download: {error}"));
        return;
    }

    let overlay = overlay.clone();
    let window = window.clone();
    let form = Rc::clone(form);
    let state = Rc::clone(state);
    glib::timeout_add_local(Duration::from_millis(100), move || loop {
        match receiver.try_recv() {
            Ok(DownloadEvent::Progress { downloaded, total }) => {
                let fraction = if total == 0 {
                    0.0
                } else {
                    downloaded.min(total) as f64 / total as f64
                };
                form.pp_model_progress.set_fraction(fraction);
                form.pp_model_progress
                    .set_text(Some(&format!("{:.0}%", fraction * 100.0)));
            }
            Ok(DownloadEvent::Finished(result)) => {
                form.pp_model_busy.set(false);
                let was_cancelled = form
                    .pp_model_cancel
                    .borrow_mut()
                    .take()
                    .is_some_and(|cancel| cancel.load(Ordering::Acquire));
                form.pp_model_picker.set_activatable(true);
                form.profiles.set_sensitive(true);
                window.set_deletable(true);
                match result {
                    Ok(_gguf_path) => {
                        let mut cfg = state.borrow().config.clone();
                        cfg.post_process.model = spec.id.to_string();
                        refresh_pp_models(&form, &cfg);
                        toast(&overlay, &format!("Installed {}.", spec.name));
                    }
                    Err(error) => {
                        update_pp_model_manager(&form);
                        if was_cancelled {
                            toast(&overlay, "Post-processing model download cancelled.");
                        } else {
                            toast_error(
                                &overlay,
                                &format!("Could not download {}: {error}", spec.name),
                            );
                        }
                    }
                }
                return glib::ControlFlow::Break;
            }
            Err(mpsc::TryRecvError::Empty) => return glib::ControlFlow::Continue,
            Err(mpsc::TryRecvError::Disconnected) => {
                form.pp_model_busy.set(false);
                form.pp_model_cancel.borrow_mut().take();
                form.pp_model_picker.set_activatable(true);
                form.profiles.set_sensitive(true);
                window.set_deletable(true);
                update_pp_model_manager(&form);
                toast_error(
                    &overlay,
                    "Post-processing model download stopped unexpectedly.",
                );
                return glib::ControlFlow::Break;
            }
        }
    });
}

fn confirm_delete_pp_model(
    window: &adw::ApplicationWindow,
    overlay: &adw::ToastOverlay,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
    spec: &'static model_catalog::PostProcessModelSpec,
) {
    match profiles_using_pp_model(state, spec) {
        Ok(profiles) if !profiles.is_empty() => {
            toast_error(
                overlay,
                &format!(
                    "{} is enabled in: {}. Disable post-processing or pick another model in those profiles, then save before deleting.",
                    spec.name,
                    profiles.join(", ")
                ),
            );
            return;
        }
        Err(error) => {
            toast_error(
                overlay,
                &format!("Could not check model usage before deletion: {error:#}"),
            );
            return;
        }
        Ok(_) => {}
    }

    let dialog = adw::AlertDialog::new(
        Some(&format!("Delete {}?", spec.name)),
        Some("The verified model and tokenizer files will be removed from the app's data directory."),
    );
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("delete", "Delete");
    dialog.set_response_appearance("delete", adw::ResponseAppearance::Destructive);
    dialog.set_default_response(Some("cancel"));
    dialog.set_close_response("cancel");

    let form = Rc::clone(form);
    let state = Rc::clone(state);
    let overlay = overlay.clone();
    dialog.connect_response(None, move |_, response| {
        if response != "delete" || form.pp_model_busy.get() {
            return;
        }
        let models_dir = default_models_dir();
        match delete_post_process_model(&models_dir, spec) {
            Ok(()) => {
                let cfg = state.borrow().config.clone();
                refresh_pp_models(&form, &cfg);
                toast(&overlay, &format!("Deleted {}.", spec.name));
            }
            Err(error) => toast_error(
                &overlay,
                &format!("Could not delete {}: {error:#}", spec.name),
            ),
        }
    });
    dialog.present(Some(window));
}

fn profiles_using_pp_model(
    state: &Rc<RefCell<EditorState>>,
    spec: &model_catalog::PostProcessModelSpec,
) -> Result<Vec<String>> {
    let store = state.borrow().store.clone();
    let mut profiles = Vec::new();
    for name in store.list().context("listing profiles")? {
        let cfg = store
            .load(&name)
            .with_context(|| format!("loading profile '{name}'"))?;
        if cfg.post_process.enabled && cfg.post_process.model == spec.id {
            profiles.push(name);
        }
    }
    Ok(profiles)
}

fn sync_injection_controls(form: &Form) {
    let enabled = form.injection_enabled.is_active();
    form.batch_mode.set_sensitive(enabled);
    form.char_delay_ms
        .set_sensitive(enabled && !form.batch_mode.is_active());
}

fn sync_overlay_controls(form: &Form) {
    let enabled = form.overlay_enabled.is_active();
    form.overlay_anchor.set_sensitive(enabled);
    form.overlay_margin.set_sensitive(enabled);
    form.crossfade_ms.set_sensitive(enabled);
}

fn sync_pp_controls(form: &Form) {
    let enabled = form.pp_enabled.is_active();
    form.pp_styling.set_sensitive(enabled);
    form.pp_structure.set_sensitive(enabled);
    form.pp_context.set_sensitive(enabled);
}

fn collect_config(state: &Rc<RefCell<EditorState>>, form: &Form) -> Config {
    let mut cfg = state.borrow().config.clone();
    let selected_path = selected_model_choice(form)
        .and_then(|choice| choice.path())
        .map(|path| path.to_string_lossy().into_owned());
    cfg.model.path = selected_path.unwrap_or_else(|| form.model_path.text().trim().to_string());
    if let Some(name) = Path::new(&cfg.model.path).file_name() {
        cfg.model.name = name.to_string_lossy().into_owned();
    }
    cfg.model.residency = if form.residency.selected() == 1 {
        ModelResidency::Lazy
    } else {
        ModelResidency::Preload
    };
    cfg.model.threads = spin_u32(&form.threads);
    cfg.model.language = form.language.text().to_string();

    // Capture is normalized to the pipeline format; these fields are kept in
    // the profile for TOML compatibility, not exposed as user controls.
    cfg.audio.sample_rate = PIPELINE_SAMPLE_RATE_HZ;
    cfg.audio.channels = 1;
    cfg.audio.vad_silence_ms = spin_u32(&form.vad_silence_ms);
    cfg.audio.silence_gap_ms = cfg.audio.vad_silence_ms;
    cfg.audio.max_edit_seconds = spin_u32(&form.max_edit_seconds);
    cfg.audio.stt_channel_capacity = spin_u32(&form.stt_channel_capacity) as usize;
    cfg.audio.node_name = nonempty(form.node_name.text().as_str());

    cfg.overlay.enabled = form.overlay_enabled.is_active();
    cfg.overlay.anchor = ANCHORS
        .get(form.overlay_anchor.selected() as usize)
        .map(|(anchor, _)| *anchor)
        .unwrap_or(OverlayAnchor::Bottom);
    cfg.overlay.margin = spin_u32(&form.overlay_margin) as i32;
    cfg.overlay.crossfade_ms = spin_u32(&form.crossfade_ms);
    cfg.overlay.preview_crossfade_ms = cfg.overlay.crossfade_ms;
    cfg.overlay.edit_buffer_cap_secs = cfg.audio.max_edit_seconds;

    cfg.injection.enabled = form.injection_enabled.is_active();
    cfg.injection.batch_mode = form.batch_mode.is_active();
    cfg.injection.char_delay_ms = spin_u32(&form.char_delay_ms);
    cfg.tray.enabled = form.tray_enabled.is_active();

    cfg.post_process.enabled = form.pp_enabled.is_active();
    if let Some(spec) = selected_pp_spec(form) {
        cfg.post_process.model = spec.id.to_string();
    }
    let axis_value = |axis: &'static [(&'static str, &'static str)],
                      picker: &adw::ComboRow,
                      fallback: &str| {
        axis.get(picker.selected() as usize)
            .map(|(value, _)| (*value).to_string())
            .unwrap_or_else(|| fallback.to_string())
    };
    cfg.post_process.styling = axis_value(
        PostProcessConfig::STYLINGS,
        &form.pp_styling,
        &cfg.post_process.styling,
    );
    cfg.post_process.structure = axis_value(
        PostProcessConfig::STRUCTURES,
        &form.pp_structure,
        &cfg.post_process.structure,
    );
    cfg.post_process.context = axis_value(
        PostProcessConfig::CONTEXTS,
        &form.pp_context,
        &cfg.post_process.context,
    );
    cfg.normalize();
    cfg
}

fn populate_form(form: &Form, cfg: &Config) {
    let guard = UpdatingGuard::new(form);
    form.model_path.set_text(&cfg.model.path);
    form.residency.set_selected(match cfg.model.residency {
        ModelResidency::Preload => 0,
        ModelResidency::Lazy => 1,
    });
    form.threads.set_value(cfg.model.threads as f64);
    form.language.set_text(&cfg.model.language);

    form.vad_silence_ms
        .set_value(cfg.audio.vad_silence_ms as f64);
    form.max_edit_seconds
        .set_value(cfg.audio.max_edit_seconds as f64);
    form.stt_channel_capacity
        .set_value(cfg.audio.stt_channel_capacity as f64);
    form.node_name
        .set_text(cfg.audio.node_name.as_deref().unwrap_or(""));

    form.overlay_enabled.set_active(cfg.overlay.enabled);
    let anchor_index = ANCHORS
        .iter()
        .position(|(anchor, _)| *anchor == cfg.overlay.anchor)
        .unwrap_or(1) as u32;
    form.overlay_anchor.set_selected(anchor_index);
    form.overlay_margin
        .set_value(cfg.overlay.margin.max(0) as f64);
    form.crossfade_ms.set_value(cfg.overlay.crossfade_ms as f64);

    form.injection_enabled.set_active(cfg.injection.enabled);
    form.batch_mode.set_active(cfg.injection.batch_mode);
    form.char_delay_ms
        .set_value(cfg.injection.char_delay_ms as f64);
    form.tray_enabled.set_active(cfg.tray.enabled);

    form.pp_enabled.set_active(cfg.post_process.enabled);
    let axis_index = |axis: &[(&str, &str)], value: &str| -> u32 {
        axis.iter()
            .position(|(name, _)| *name == value)
            .unwrap_or(0) as u32
    };
    form.pp_styling
        .set_selected(axis_index(PostProcessConfig::STYLINGS, &cfg.post_process.styling));
    form.pp_structure.set_selected(axis_index(
        PostProcessConfig::STRUCTURES,
        &cfg.post_process.structure,
    ));
    form.pp_context.set_selected(axis_index(
        PostProcessConfig::CONTEXTS,
        &cfg.post_process.context,
    ));
    refresh_pp_models(form, cfg);
    drop(guard);
    sync_injection_controls(form);
    sync_overlay_controls(form);
    sync_pp_controls(form);
}

fn spin_u32(spin: &adw::SpinRow) -> u32 {
    spin.value().max(0.0) as u32
}

fn nonempty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn unique_profile_name(store: &ProfileStore, base: &str) -> String {
    let names = store.list().unwrap_or_default();
    if !names.iter().any(|name| name == base) {
        return base.to_string();
    }
    for index in 2..=999 {
        let candidate = format!("{base} {index}");
        if !names.iter().any(|name| name == &candidate) {
            return candidate;
        }
    }
    format!("Profile {}", names.len() + 1)
}

fn toast(overlay: &adw::ToastOverlay, message: &str) {
    overlay.add_toast(adw::Toast::new(message));
}

fn toast_error(overlay: &adw::ToastOverlay, message: &str) {
    overlay.add_toast(adw::Toast::builder().title(message).timeout(6).build());
}

fn show_startup_error(app: &adw::Application, error: &anyhow::Error) {
    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("STT Settings")
        .default_width(480)
        .default_height(320)
        .build();
    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&adw::HeaderBar::new());
    let status = adw::StatusPage::new();
    status.set_icon_name(Some("dialog-error-symbolic"));
    status.set_title("Could Not Open Settings");
    status.set_description(Some(&format!("{error:#}")));
    toolbar.set_content(Some(&status));
    window.set_content(Some(&toolbar));
    window.present();
}
