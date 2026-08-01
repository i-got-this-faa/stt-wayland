//! GTK4 settings application for `sttd`.
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

use anyhow::{Context, Result};
use gtk4::prelude::*;
use gtk4::{
    self, Application, ApplicationWindow, Box as GtkBox, Button, CheckButton, ComboBoxText,
    DialogFlags, Entry, Frame, Grid, HeaderBar, Label, MessageDialog, MessageType, Orientation,
    PolicyType, ResponseType, ScrolledWindow, SpinButton,
};

use config::{default_models_dir, Config, ModelResidency, OverlayAnchor, ProfileStore};

fn main() {
    let app = Application::builder()
        .application_id("io.github.stt_wayland.Settings")
        .build();

    app.connect_activate(|app| {
        if let Err(error) = build_ui(app) {
            show_startup_error(app, &error);
        }
    });
    app.run();
}

#[derive(Debug)]
struct EditorState {
    store: ProfileStore,
    current: String,
    active: Option<String>,
    config: Config,
    loading_profile: Cell<bool>,
}

struct Form {
    profiles: ComboBoxText,
    active_badge: Label,
    model_picker: ComboBoxText,
    model_path: Entry,
    refresh_models: Button,
    residency: ComboBoxText,
    threads: SpinButton,
    language: Entry,
    sample_rate: SpinButton,
    channels: SpinButton,
    vad_silence_ms: SpinButton,
    max_edit_seconds: SpinButton,
    stt_channel_capacity: SpinButton,
    node_name: Entry,
    overlay_enabled: CheckButton,
    overlay_anchor: ComboBoxText,
    overlay_margin: SpinButton,
    crossfade_ms: SpinButton,
    injection_enabled: CheckButton,
    batch_mode: CheckButton,
    char_delay_ms: SpinButton,
    tray_enabled: CheckButton,
    status: Label,
}

struct ProfileButtons {
    new: Button,
    duplicate: Button,
    delete: Button,
    save: Button,
    apply: Button,
}

impl Form {
    fn new() -> Self {
        Self {
            profiles: ComboBoxText::new(),
            active_badge: Label::new(None),
            model_picker: combo(&[("", "No models discovered")]),
            model_path: Entry::builder().hexpand(true).build(),
            refresh_models: Button::with_label("Refresh models"),
            residency: combo(&[("preload", "Preload"), ("lazy", "Lazy")]),
            threads: SpinButton::with_range(0.0, 128.0, 1.0),
            language: Entry::builder().hexpand(true).build(),
            sample_rate: SpinButton::with_range(8_000.0, 96_000.0, 1_000.0),
            channels: SpinButton::with_range(1.0, 8.0, 1.0),
            vad_silence_ms: SpinButton::with_range(50.0, 10_000.0, 50.0),
            max_edit_seconds: SpinButton::with_range(1.0, 3_600.0, 1.0),
            stt_channel_capacity: SpinButton::with_range(1.0, 256.0, 1.0),
            node_name: Entry::builder().hexpand(true).build(),
            overlay_enabled: CheckButton::new(),
            overlay_anchor: combo(&[
                ("top", "Top"),
                ("bottom", "Bottom"),
                ("left", "Left"),
                ("right", "Right"),
                ("top_left", "Top left"),
                ("top_right", "Top right"),
                ("bottom_left", "Bottom left"),
                ("bottom_right", "Bottom right"),
                ("center", "Center"),
            ]),
            overlay_margin: SpinButton::with_range(0.0, 2_000.0, 1.0),
            crossfade_ms: SpinButton::with_range(0.0, 5_000.0, 10.0),
            injection_enabled: CheckButton::new(),
            batch_mode: CheckButton::new(),
            char_delay_ms: SpinButton::with_range(0.0, 1_000.0, 1.0),
            tray_enabled: CheckButton::new(),
            status: Label::new(None),
        }
    }
}

fn build_ui(app: &Application) -> Result<()> {
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
        loading_profile: Cell::new(false),
    }));
    let active_pointer = state.borrow().store.active()?;
    state.borrow_mut().active = active_pointer;

    let form = Rc::new(Form::new());
    populate_form(&form, &state.borrow().config);
    refresh_models(&form, &state.borrow().config);

    let window = ApplicationWindow::builder()
        .application(app)
        .title("stt-settings")
        .default_width(1_040)
        .default_height(760)
        .build();

    let header = HeaderBar::new();
    header.set_title_widget(Some(&Label::new(Some("stt-settings"))));
    window.set_titlebar(Some(&header));

    let root = GtkBox::new(Orientation::Vertical, 0);
    root.set_vexpand(true);

    let profile_bar = GtkBox::new(Orientation::Horizontal, 8);
    profile_bar.set_margin_start(16);
    profile_bar.set_margin_end(16);
    profile_bar.set_margin_top(12);
    profile_bar.set_margin_bottom(12);
    profile_bar.append(&Label::new(Some("Profile")));
    form.profiles.set_hexpand(false);
    form.profiles.set_size_request(220, -1);
    profile_bar.append(&form.profiles);
    form.active_badge.set_xalign(0.0);
    form.active_badge.set_hexpand(true);
    profile_bar.append(&form.active_badge);

    let buttons = ProfileButtons {
        new: Button::with_label("New"),
        duplicate: Button::with_label("Duplicate"),
        delete: Button::with_label("Delete"),
        save: Button::with_label("Save"),
        apply: Button::with_label("Apply profile"),
    };
    profile_bar.append(&buttons.new);
    profile_bar.append(&buttons.duplicate);
    profile_bar.append(&buttons.delete);
    profile_bar.append(&buttons.save);
    profile_bar.append(&buttons.apply);
    root.append(&profile_bar);

    let scrolled = ScrolledWindow::builder()
        .hscrollbar_policy(PolicyType::Never)
        .vexpand(true)
        .build();
    let content = GtkBox::new(Orientation::Vertical, 14);
    content.set_margin_start(16);
    content.set_margin_end(16);
    content.set_margin_bottom(20);

    let intro = Label::new(Some(
        "Edit a profile, save it independently, then apply it to the daemon's active config.\nRestart sttd after applying changes.",
    ));
    intro.set_xalign(0.0);
    intro.set_wrap(true);
    content.append(&intro);

    let model_grid = grid();
    add_row(&model_grid, 0, "Installed models", &form.model_picker);
    let model_path_row = GtkBox::new(Orientation::Horizontal, 8);
    model_path_row.append(&form.model_path);
    model_path_row.append(&form.refresh_models);
    add_row(&model_grid, 1, "Model path", &model_path_row);
    add_row(&model_grid, 2, "Residency", &form.residency);
    add_row(&model_grid, 3, "Whisper threads (0 = auto)", &form.threads);
    add_row(&model_grid, 4, "Language", &form.language);
    content.append(&section("Model", &model_grid));

    let audio_grid = grid();
    add_row(&audio_grid, 0, "Sample rate (Hz)", &form.sample_rate);
    add_row(&audio_grid, 1, "Channels", &form.channels);
    add_row(&audio_grid, 2, "VAD silence gap (ms)", &form.vad_silence_ms);
    add_row(
        &audio_grid,
        3,
        "Edit capture limit (seconds)",
        &form.max_edit_seconds,
    );
    add_row(
        &audio_grid,
        4,
        "STT queue capacity",
        &form.stt_channel_capacity,
    );
    add_row(&audio_grid, 5, "PipeWire node (optional)", &form.node_name);
    content.append(&section("Audio", &audio_grid));

    let overlay_grid = grid();
    add_row(&overlay_grid, 0, "Enable overlay", &form.overlay_enabled);
    add_row(&overlay_grid, 1, "Anchor", &form.overlay_anchor);
    add_row(&overlay_grid, 2, "Margin (px)", &form.overlay_margin);
    add_row(
        &overlay_grid,
        3,
        "Preview crossfade (ms)",
        &form.crossfade_ms,
    );
    content.append(&section("Overlay", &overlay_grid));

    let injection_grid = grid();
    add_row(
        &injection_grid,
        0,
        "Enable text injection",
        &form.injection_enabled,
    );
    add_row(&injection_grid, 1, "Batch committed text", &form.batch_mode);
    add_row(
        &injection_grid,
        2,
        "Character delay (ms)",
        &form.char_delay_ms,
    );
    content.append(&section("Injection", &injection_grid));

    let tray_grid = grid();
    add_row(&tray_grid, 0, "Enable tray icon", &form.tray_enabled);
    content.append(&section("Tray", &tray_grid));

    form.status.set_xalign(0.0);
    form.status.set_wrap(true);
    form.status.add_css_class("dim-label");
    content.append(&form.status);

    scrolled.set_child(Some(&content));
    root.append(&scrolled);
    window.set_child(Some(&root));

    connect_profile_signals(&window, &form, &state, &buttons);
    refresh_profiles(&form, &state);
    set_status(
        &form,
        "Ready. Select a profile or edit the current one.",
        false,
    );
    window.present();
    Ok(())
}

fn connect_profile_signals(
    window: &ApplicationWindow,
    form: &Rc<Form>,
    state: &Rc<RefCell<EditorState>>,
    buttons: &ProfileButtons,
) {
    {
        let state = Rc::clone(state);
        let form = Rc::clone(form);
        let profile_combo = form.profiles.clone();
        profile_combo.connect_changed(move |combo| {
            if state.borrow().loading_profile.get() {
                return;
            }
            let Some(name) = combo.active_id().map(|id| id.to_string()) else {
                return;
            };
            load_profile(&form, &state, &name);
        });
    }

    {
        let form = Rc::clone(form);
        let model_picker = form.model_picker.clone();
        model_picker.connect_changed(move |combo| {
            if let Some(path) = combo.active_id() {
                if !path.is_empty() {
                    form.model_path.set_text(path.as_str());
                }
            }
        });
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let refresh_button = form.refresh_models.clone();
        refresh_button.connect_clicked(move |_| {
            let mut cfg = state.borrow().config.clone();
            cfg.model.path = form.model_path.text().to_string();
            refresh_models(&form, &cfg);
            set_status(&form, "Model list refreshed.", false);
        });
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        buttons.new.connect_clicked(move |_| {
            let store = state.borrow().store.clone();
            let name = unique_profile_name(&store, "New profile");
            let cfg = collect_config(&state, &form);
            match store.save(&name, &cfg) {
                Ok(()) => {
                    {
                        let mut editor = state.borrow_mut();
                        editor.current = name.clone();
                        editor.config = cfg;
                    }
                    refresh_profiles(&form, &state);
                    set_status(&form, &format!("Created profile '{name}'."), false);
                }
                Err(error) => {
                    set_status(&form, &format!("Could not create profile: {error:#}"), true)
                }
            }
        });
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        buttons.duplicate.connect_clicked(move |_| {
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
                    set_status(&form, &format!("Duplicated profile as '{name}'."), false);
                }
                Err(error) => set_status(
                    &form,
                    &format!("Could not duplicate profile: {error:#}"),
                    true,
                ),
            }
        });
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        buttons.save.connect_clicked(move |_| {
            let name = state.borrow().current.clone();
            let cfg = collect_config(&state, &form);
            let store = state.borrow().store.clone();
            match store.save(&name, &cfg) {
                Ok(()) => {
                    state.borrow_mut().config = cfg;
                    set_status(
                        &form,
                        &format!("Saved '{name}'. Apply it to make it active."),
                        false,
                    );
                }
                Err(error) => {
                    set_status(&form, &format!("Could not save profile: {error:#}"), true)
                }
            }
        });
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        buttons.apply.connect_clicked(move |_| {
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
                    set_status(
                        &form,
                        &format!("Applied '{name}' to config.toml. Restart sttd to use it."),
                        false,
                    );
                }
                Err(error) => {
                    set_status(&form, &format!("Could not apply profile: {error:#}"), true)
                }
            }
        });
    }

    {
        let form = Rc::clone(form);
        let state = Rc::clone(state);
        let window = window.clone();
        buttons.delete.connect_clicked(move |_| {
            let name = state.borrow().current.clone();
            let dialog = MessageDialog::new(
                Some(&window),
                DialogFlags::MODAL,
                MessageType::Question,
                gtk4::ButtonsType::OkCancel,
                format!("Delete profile '{name}'?"),
            );
            let form = Rc::clone(&form);
            let state = Rc::clone(&state);
            dialog.connect_response(move |dialog, response| {
                if response == ResponseType::Ok {
                    let store = state.borrow().store.clone();
                    match store.delete(&name) {
                        Ok(()) => {
                            let next = store.active().ok().flatten().or_else(|| {
                                store.list().ok().and_then(|names| names.into_iter().next())
                            });
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
                            set_status(&form, "Profile deleted.", false);
                        }
                        Err(error) => {
                            set_status(&form, &format!("Could not delete profile: {error:#}"), true)
                        }
                    }
                }
                dialog.close();
            });
            dialog.present();
        });
    }
}

fn load_profile(form: &Form, state: &Rc<RefCell<EditorState>>, name: &str) {
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
            set_status(form, &format!("Loaded '{name}'."), false);
        }
        Err(error) => set_status(form, &format!("Could not load profile: {error:#}"), true),
    }
}

fn refresh_profiles(form: &Form, state: &Rc<RefCell<EditorState>>) {
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
            set_status(form, &format!("Could not list profiles: {error:#}"), true);
            return;
        }
    };

    state.borrow().loading_profile.set(true);
    form.profiles.remove_all();
    for name in names {
        let label = if active.as_deref() == Some(name.as_str()) {
            format!("{name} (active)")
        } else {
            name.clone()
        };
        form.profiles.append(Some(&name), &label);
    }
    form.profiles.set_active_id(Some(&current));
    state.borrow().loading_profile.set(false);
    form.active_badge.set_text(
        &active
            .map(|name| format!("Active: {name}"))
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

    form.model_picker.remove_all();
    if paths.is_empty() {
        form.model_picker.append(Some(""), "No .bin models found");
    } else {
        for path in paths {
            let id = path.to_string_lossy().into_owned();
            let label = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&id)
                .to_string();
            form.model_picker.append(Some(&id), &label);
        }
    }
    if !current.is_empty() {
        form.model_picker.set_active_id(Some(&current));
    }
}

fn collect_config(state: &Rc<RefCell<EditorState>>, form: &Form) -> Config {
    let mut cfg = state.borrow().config.clone();
    cfg.model.path = form.model_path.text().to_string();
    if let Some(name) = Path::new(&cfg.model.path).file_name() {
        cfg.model.name = name.to_string_lossy().into_owned();
    }
    cfg.model.residency = match combo_id(&form.residency).as_deref() {
        Some("lazy") => ModelResidency::Lazy,
        _ => ModelResidency::Preload,
    };
    cfg.model.threads = nonnegative_u32(&form.threads);
    cfg.model.language = form.language.text().to_string();

    cfg.audio.sample_rate = nonnegative_u32(&form.sample_rate);
    cfg.audio.channels = nonnegative_u32(&form.channels) as u16;
    cfg.audio.vad_silence_ms = nonnegative_u32(&form.vad_silence_ms);
    cfg.audio.silence_gap_ms = cfg.audio.vad_silence_ms;
    cfg.audio.max_edit_seconds = nonnegative_u32(&form.max_edit_seconds);
    cfg.audio.stt_channel_capacity = nonnegative_u32(&form.stt_channel_capacity) as usize;
    cfg.audio.node_name = nonempty(form.node_name.text().as_str());

    cfg.overlay.enabled = form.overlay_enabled.is_active();
    cfg.overlay.anchor = overlay_anchor(combo_id(&form.overlay_anchor).as_deref());
    cfg.overlay.margin = nonnegative_u32(&form.overlay_margin) as i32;
    cfg.overlay.crossfade_ms = nonnegative_u32(&form.crossfade_ms);
    cfg.overlay.preview_crossfade_ms = cfg.overlay.crossfade_ms;
    cfg.overlay.edit_buffer_cap_secs = cfg.audio.max_edit_seconds;

    cfg.injection.enabled = form.injection_enabled.is_active();
    cfg.injection.batch_mode = form.batch_mode.is_active();
    cfg.injection.char_delay_ms = nonnegative_u32(&form.char_delay_ms);
    cfg.tray.enabled = form.tray_enabled.is_active();
    cfg.normalize();
    cfg
}

fn populate_form(form: &Form, cfg: &Config) {
    form.model_path.set_text(&cfg.model.path);
    form.residency
        .set_active_id(Some(match cfg.model.residency {
            ModelResidency::Preload => "preload",
            ModelResidency::Lazy => "lazy",
        }));
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
    form.overlay_anchor
        .set_active_id(Some(overlay_anchor_id(cfg.overlay.anchor)));
    form.overlay_margin
        .set_value(cfg.overlay.margin.max(0) as f64);
    form.crossfade_ms.set_value(cfg.overlay.crossfade_ms as f64);

    form.injection_enabled.set_active(cfg.injection.enabled);
    form.batch_mode.set_active(cfg.injection.batch_mode);
    form.char_delay_ms
        .set_value(cfg.injection.char_delay_ms as f64);
    form.tray_enabled.set_active(cfg.tray.enabled);
}

fn combo(items: &[(&str, &str)]) -> ComboBoxText {
    let combo = ComboBoxText::new();
    for (id, label) in items {
        combo.append(Some(id), label);
    }
    if let Some((id, _)) = items.first() {
        combo.set_active_id(Some(id));
    }
    combo
}

fn grid() -> Grid {
    let grid = Grid::new();
    grid.set_column_spacing(24);
    grid.set_row_spacing(8);
    grid.set_column_homogeneous(false);
    grid
}

fn add_row<W: IsA<gtk4::Widget>>(grid: &Grid, row: i32, text: &str, widget: &W) {
    let label = Label::new(Some(text));
    label.set_xalign(0.0);
    label.set_hexpand(true);
    grid.attach(&label, 0, row, 1, 1);
    grid.attach(widget, 1, row, 1, 1);
}

fn section(title: &str, grid: &Grid) -> Frame {
    let frame = Frame::new(Some(title));
    frame.set_margin_top(2);
    frame.set_margin_bottom(2);
    frame.set_margin_start(2);
    frame.set_margin_end(2);
    frame.set_child(Some(grid));
    frame
}

fn combo_id(combo: &ComboBoxText) -> Option<String> {
    combo.active_id().map(|id| id.to_string())
}

fn nonnegative_u32(spin: &SpinButton) -> u32 {
    spin.value_as_int().max(0) as u32
}

fn nonempty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn overlay_anchor(value: Option<&str>) -> OverlayAnchor {
    match value {
        Some("top") => OverlayAnchor::Top,
        Some("left") => OverlayAnchor::Left,
        Some("right") => OverlayAnchor::Right,
        Some("top_left") => OverlayAnchor::TopLeft,
        Some("top_right") => OverlayAnchor::TopRight,
        Some("bottom_left") => OverlayAnchor::BottomLeft,
        Some("bottom_right") => OverlayAnchor::BottomRight,
        Some("center") => OverlayAnchor::Center,
        _ => OverlayAnchor::Bottom,
    }
}

fn overlay_anchor_id(anchor: OverlayAnchor) -> &'static str {
    match anchor {
        OverlayAnchor::Top => "top",
        OverlayAnchor::Bottom => "bottom",
        OverlayAnchor::Left => "left",
        OverlayAnchor::Right => "right",
        OverlayAnchor::TopLeft => "top_left",
        OverlayAnchor::TopRight => "top_right",
        OverlayAnchor::BottomLeft => "bottom_left",
        OverlayAnchor::BottomRight => "bottom_right",
        OverlayAnchor::Center => "center",
    }
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

fn set_status(form: &Form, message: &str, error: bool) {
    form.status.set_text(message);
    if error {
        form.status.add_css_class("error");
    } else {
        form.status.remove_css_class("error");
    }
}

fn show_startup_error(app: &Application, error: &anyhow::Error) {
    let window = ApplicationWindow::builder()
        .application(app)
        .title("stt-settings error")
        .default_width(520)
        .default_height(180)
        .build();
    let box_ = GtkBox::new(Orientation::Vertical, 12);
    box_.set_margin_start(20);
    box_.set_margin_end(20);
    box_.set_margin_top(20);
    box_.set_margin_bottom(20);
    let label = Label::new(Some(&format!("Could not open stt-settings:\n{error:#}")));
    label.set_wrap(true);
    label.set_xalign(0.0);
    box_.append(&label);
    window.set_child(Some(&box_));
    window.present();
}
