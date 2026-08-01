//! GTK4 + libadwaita settings application for `sttd`.
//!
//! The app edits the daemon's complete TOML configuration through named
//! profiles. Saving updates only the selected profile; applying a profile also
//! mirrors it to `~/.config/sttd/config.toml`, which keeps the existing
//! systemd user unit compatible. Restart `sttd` after applying changes.

#[path = "config.rs"]
mod config;

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use adw::prelude::*;
use anyhow::{Context, Result};
use gtk4::gio;

use config::{default_models_dir, Config, ModelResidency, OverlayAnchor, ProfileStore};

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

/// Every editable row, kept flat so `collect_config` / `populate_form` can
/// read and write them without walking the widget tree.
struct Form {
    /// Guards signal handlers while rows are rebuilt programmatically.
    updating: Cell<bool>,
    /// Paths backing the rows of `model_picker`.
    model_paths: RefCell<Vec<PathBuf>>,
    profiles: adw::ComboRow,
    model_picker: adw::ComboRow,
    model_path: adw::EntryRow,
    residency: adw::ComboRow,
    threads: adw::SpinRow,
    language: adw::EntryRow,
    sample_rate: adw::SpinRow,
    channels: adw::SpinRow,
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
}

impl Form {
    fn new() -> Self {
        let profiles = adw::ComboRow::new();
        profiles.set_title("Profile");

        let model_picker = adw::ComboRow::new();
        model_picker.set_title("Installed Model");
        model_picker.set_subtitle("Discovered in the models directory");

        let model_path = entry_row("Model Path", "Path to the ggml Whisper model file (.bin)");

        let residency = string_combo(&RESIDENCIES);
        residency.set_title("Residency");
        residency.set_subtitle("Preload at startup, or load lazily on first use");

        let threads = spin_row("Threads", "Inference threads (0 = auto)", 0.0, 128.0, 1.0);
        let language = entry_row(
            "Language",
            "Whisper language code such as \"en\" (empty = auto-detect)",
        );

        let sample_rate = spin_row(
            "Sample Rate",
            "Capture rate in Hz — Whisper expects 16000",
            8_000.0,
            96_000.0,
            1_000.0,
        );
        let channels = spin_row(
            "Channels",
            "Captured audio channels (1 = mono)",
            1.0,
            8.0,
            1.0,
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
            "PipeWire Node",
            "Optional PipeWire target node name (empty = default source)",
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
            "Batch Committed Text",
            "Inject committed text at once instead of per keystroke",
        );
        let char_delay_ms = spin_row(
            "Character Delay",
            "Delay between injected characters in milliseconds",
            0.0,
            1_000.0,
            1.0,
        );

        let tray_enabled = switch_row("Enable Tray Icon", "Show a status icon in the system tray");

        Self {
            updating: Cell::new(false),
            model_paths: RefCell::new(Vec::new()),
            profiles,
            model_picker,
            model_path,
            residency,
            threads,
            language,
            sample_rate,
            channels,
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
        }
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
        .default_width(680)
        .default_height(840)
        .build();

    // ── Header bar: window title plus the primary actions ────────────
    let header = adw::HeaderBar::new();
    header.set_title_widget(Some(&adw::WindowTitle::new("STT Settings", "")));

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
    banner.set_button_label(Some("Got It"));
    banner.set_revealed(false);
    banner.connect_button_clicked(|banner| banner.set_revealed(false));
    toolbar.add_top_bar(&banner);

    // ── Preferences page ─────────────────────────────────────────────
    let page = adw::PreferencesPage::new();

    let profile_group = adw::PreferencesGroup::new();
    profile_group.set_title("Profile");
    profile_group.set_description(Some(
        "Each profile holds a complete configuration. Save writes changes to the \
         selected profile; Apply also makes it the daemon's active configuration.",
    ));
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
    page.add(&profile_group);

    let model_group = adw::PreferencesGroup::new();
    model_group.set_title("Model");
    let refresh_button = gtk4::Button::from_icon_name("view-refresh-symbolic");
    refresh_button.set_tooltip_text(Some("Refresh the discovered model list"));
    refresh_button.set_valign(gtk4::Align::Center);
    refresh_button.add_css_class("flat");
    form.model_picker.add_suffix(&refresh_button);
    let browse_button = gtk4::Button::from_icon_name("document-open-symbolic");
    browse_button.set_tooltip_text(Some("Browse for a model file"));
    browse_button.set_valign(gtk4::Align::Center);
    browse_button.add_css_class("flat");
    form.model_path.add_suffix(&browse_button);
    model_group.add(&form.model_picker);
    model_group.add(&form.model_path);
    model_group.add(&form.residency);
    model_group.add(&form.threads);
    model_group.add(&form.language);
    page.add(&model_group);

    let audio_group = adw::PreferencesGroup::new();
    audio_group.set_title("Audio");
    audio_group.add(&form.sample_rate);
    audio_group.add(&form.channels);
    audio_group.add(&form.vad_silence_ms);
    audio_group.add(&form.max_edit_seconds);
    audio_group.add(&form.stt_channel_capacity);
    audio_group.add(&form.node_name);
    page.add(&audio_group);

    let overlay_group = adw::PreferencesGroup::new();
    overlay_group.set_title("Overlay");
    overlay_group.add(&form.overlay_enabled);
    overlay_group.add(&form.overlay_anchor);
    overlay_group.add(&form.overlay_margin);
    overlay_group.add(&form.crossfade_ms);
    page.add(&overlay_group);

    let injection_group = adw::PreferencesGroup::new();
    injection_group.set_title("Injection");
    injection_group.add(&form.injection_enabled);
    injection_group.add(&form.batch_mode);
    injection_group.add(&form.char_delay_ms);
    page.add(&injection_group);

    let tray_group = adw::PreferencesGroup::new();
    tray_group.set_title("Tray");
    tray_group.add(&form.tray_enabled);
    page.add(&tray_group);

    toolbar.set_content(Some(&page));

    let toast_overlay = adw::ToastOverlay::new();
    toast_overlay.set_child(Some(&toolbar));
    window.set_content(Some(&toast_overlay));

    connect_signals(
        &window,
        &toast_overlay,
        &banner,
        &form,
        &state,
        &save_button,
        &apply_button,
        &refresh_button,
        &browse_button,
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
    refresh_button: &gtk4::Button,
    browse_button: &gtk4::Button,
) {
    // Selecting a profile loads it into the form.
    {
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
            load_profile(&form, &state, &overlay, &name);
        });
    }

    // Picking a discovered model fills in the model path entry.
    {
        let form = Rc::clone(form);
        let picker = form.model_picker.clone();
        picker.connect_selected_notify(move |row| {
            if form.updating.get() {
                return;
            }
            let index = row.selected() as usize;
            let path = form.model_paths.borrow().get(index).cloned();
            if let Some(path) = path {
                let text = path.to_string_lossy().into_owned();
                if form.model_path.text().as_str() != text {
                    form.model_path.set_text(&text);
                }
            }
        });
    }

    // Typing a path that matches a discovered model selects it.
    {
        let form = Rc::clone(form);
        let entry = form.model_path.clone();
        entry.connect_changed(move |row| {
            if form.updating.get() {
                return;
            }
            let text = row.text();
            let paths = form.model_paths.borrow();
            let index = paths
                .iter()
                .position(|path| path.to_string_lossy() == text.as_str());
            let selected = index.map_or(gtk4::INVALID_LIST_POSITION, |index| index as u32);
            if form.model_picker.selected() != selected {
                form.model_picker.set_selected(selected);
            }
        });
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let overlay = overlay.clone();
        let refresh_button = refresh_button.clone();
        refresh_button.connect_clicked(move |_| {
            let mut cfg = state.borrow().config.clone();
            cfg.model.path = form.model_path.text().to_string();
            refresh_models(&form, &cfg);
            toast(&overlay, "Model list refreshed.");
        });
    }

    {
        let form = Rc::clone(form);
        let window = window.clone();
        browse_button.connect_clicked(move |_| {
            let dialog = gtk4::FileDialog::new();
            dialog.set_title("Select a Whisper Model");
            dialog.set_modal(true);
            let filter = gtk4::FileFilter::new();
            filter.add_suffix("bin");
            filter.set_name(Some("Whisper models (*.bin)"));
            dialog.set_default_filter(Some(&filter));
            let form = Rc::clone(&form);
            dialog.open(Some(&window), gio::Cancellable::NONE, move |result| {
                if let Ok(file) = result {
                    if let Some(path) = file.path() {
                        form.model_path.set_text(path.to_string_lossy().as_ref());
                    }
                }
            });
        });
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let overlay = overlay.clone();
        save_button.connect_clicked(move |_| {
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
        Some("This permanently removes the profile from disk."),
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

    form.updating.set(true);
    let model = gtk4::StringList::new(&[]);
    for name in &names {
        model.append(name);
    }
    form.profiles.set_model(Some(&model));
    let index = names.iter().position(|name| name == &current).unwrap_or(0) as u32;
    form.profiles.set_selected(index);
    form.updating.set(false);

    form.profiles.set_subtitle(
        &active
            .map(|name| format!("Active profile: {name}"))
            .unwrap_or_else(|| "No active profile".into()),
    );
}

fn refresh_models(form: &Form, cfg: &Config) {
    let current = cfg.model.path.clone();
    let mut paths = BTreeSet::new();
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
            if path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
            {
                paths.insert(path);
            }
        }
    }
    if !current.is_empty() {
        paths.insert(PathBuf::from(&current));
    }

    form.updating.set(true);
    let model = gtk4::StringList::new(&[]);
    let mut list = Vec::with_capacity(paths.len());
    if paths.is_empty() {
        model.append("No .bin models found");
    } else {
        for path in &paths {
            let label = path
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
                .unwrap_or_else(|| path.to_string_lossy().into_owned());
            model.append(&label);
            list.push(path.clone());
        }
    }
    form.model_picker.set_model(Some(&model));
    *form.model_paths.borrow_mut() = list;

    let selected = form
        .model_paths
        .borrow()
        .iter()
        .position(|path| path.to_string_lossy().as_ref() == current)
        .map_or(gtk4::INVALID_LIST_POSITION, |index| index as u32);
    form.model_picker.set_selected(selected);
    form.updating.set(false);
}

fn collect_config(state: &Rc<RefCell<EditorState>>, form: &Form) -> Config {
    let mut cfg = state.borrow().config.clone();
    cfg.model.path = form.model_path.text().to_string();
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

    cfg.audio.sample_rate = spin_u32(&form.sample_rate);
    cfg.audio.channels = spin_u32(&form.channels) as u16;
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
    cfg.normalize();
    cfg
}

fn populate_form(form: &Form, cfg: &Config) {
    form.updating.set(true);
    form.model_path.set_text(&cfg.model.path);
    form.residency.set_selected(match cfg.model.residency {
        ModelResidency::Preload => 0,
        ModelResidency::Lazy => 1,
    });
    form.threads.set_value(cfg.model.threads as f64);
    form.language.set_text(&cfg.model.language);

    form.sample_rate.set_value(cfg.audio.sample_rate as f64);
    form.channels.set_value(cfg.audio.channels as f64);
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
    form.updating.set(false);
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
