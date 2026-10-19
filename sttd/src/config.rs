//! Configuration loading for `sttd`.
//!
//! Default path: `~/.config/sttd/config.toml` (XDG).
//! Models live under `~/.local/share/sttd/models/`.
//!
//! ## Schema (architecture §11)
//!
//! | Section     | Fields |
//! |-------------|--------|
//! | `audio`     | `sample_rate`, `vad_silence_ms`, `max_edit_seconds` |
//! | `model`     | `path`, `residency` (`preload` \| `lazy`) |
//! | `overlay`   | `anchor`, `margin`, `crossfade_ms` |
//! | `injection` | `batch_mode` |
//! | `tray`      | `enabled` |
//!
//! Legacy TOML keys (`silence_gap_ms`, `preview_crossfade_ms`, `name`+`directory`, …)
//! are accepted via serde aliases / post-load normalization.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Capture rate required by the VAD, preview, and ASR pipeline.
///
/// Capture backends may deliver another native rate; they resample to this
/// value before samples enter the daemon.
pub const PIPELINE_SAMPLE_RATE_HZ: u32 = 16_000;

// ─── Top-level ──────────────────────────────────────────────────────────────

/// Top-level daemon configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub model: ModelConfig,
    pub overlay: OverlayConfig,
    pub injection: InjectionConfig,
    pub tray: TrayConfig,
    pub post_process: PostProcessConfig,
}

impl Default for Config {
    fn default() -> Self {
        let mut cfg = Self {
            audio: AudioConfig::default(),
            model: ModelConfig::default(),
            overlay: OverlayConfig::default(),
            injection: InjectionConfig::default(),
            tray: TrayConfig::default(),
            post_process: PostProcessConfig::default(),
        };
        cfg.normalize();
        cfg
    }
}

// ─── Audio ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Capture sample rate (Hz). Kept for TOML compatibility; normalization
    /// always pins this to [`PIPELINE_SAMPLE_RATE_HZ`].
    pub sample_rate: u32,

    /// VAD silence gap that closes a speech chunk (milliseconds).
    /// TOML key: `vad_silence_ms` (alias: `silence_gap_ms`).
    #[serde(alias = "silence_gap_ms")]
    pub vad_silence_ms: u32,

    /// Max edit-mode buffer duration in seconds (default 300 ≈ 5 minutes).
    pub max_edit_seconds: u32,

    /// Number of capture channels. Kept for TOML compatibility; the pipeline
    /// always downmixes to mono during normalization.
    pub channels: u16,

    /// Bounded mpsc capacity (number of `Vec<f32>` chunks) for STT backpressure.
    pub stt_channel_capacity: usize,

    /// Optional PipeWire target node name (`None` = default source).
    pub node_name: Option<String>,

    /// Legacy name for [`Self::vad_silence_ms`]; kept in sync by [`Config::normalize`].
    /// Existing modules (`vad`) read this field.
    #[serde(skip)]
    pub silence_gap_ms: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        let vad_silence_ms = 500;
        Self {
            sample_rate: 16_000,
            vad_silence_ms,
            max_edit_seconds: 300,
            channels: 1,
            stt_channel_capacity: 8,
            node_name: None,
            silence_gap_ms: vad_silence_ms,
        }
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// When the transcription model is loaded into memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelResidency {
    /// Load model at daemon start (default).
    #[default]
    Preload,
    /// Load on first Start / Toggle that needs STT.
    Lazy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Path to the audio.cpp GGUF model file. Supports leading `~`.
    ///
    /// Empty means “derive from [`Self::name`] + [`Self::directory`]”
    /// (legacy TOML). After [`Config::load`], always the expanded path.
    /// Default resolved path: `~/.local/share/sttd/models/qwen3-asr-0.6b-q8_0.gguf`.
    pub path: String,

    /// `preload` (default) or `lazy`.
    pub residency: ModelResidency,

    /// Whisper threads (`0` = library default / auto).
    pub threads: u32,

    /// Language hint (e.g. `"en"`); empty = auto.
    pub language: String,

    /// Model file name (status display / legacy TOML). Used when `path` is empty.
    pub name: String,

    /// Directory containing model files (legacy). Default: `~/.local/share/sttd/models/`.
    pub directory: Option<PathBuf>,
}

fn default_model_name() -> String {
    "qwen3-asr-0.6b-q8_0.gguf".into()
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            // Empty until normalize/model_path so legacy `name`-only TOML works
            // (serde default for missing `path` is also empty).
            path: String::new(),
            residency: ModelResidency::Preload,
            threads: 0,
            language: "en".into(),
            name: default_model_name(),
            directory: None,
        }
    }
}

impl ModelConfig {
    /// Resolve the model file path.
    ///
    /// Prefers `path` (with `~` expansion) when non-empty; otherwise
    /// `directory`/`name` (default dir: `~/.local/share/sttd/models/`).
    pub fn model_path(&self) -> PathBuf {
        if !self.path.is_empty() {
            return expand_tilde(&self.path);
        }
        let dir = self.directory.clone().unwrap_or_else(default_models_dir);
        let name = Path::new(&self.name);
        if name.is_absolute() {
            expand_tilde_path(name)
        } else {
            dir.join(name)
        }
    }
}

// ─── Overlay ────────────────────────────────────────────────────────────────

/// Layer-shell edge / corner anchor for the overlay surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OverlayAnchor {
    Top,
    #[default]
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OverlayConfig {
    /// Show live preview card while recording.
    pub enabled: bool,

    /// Screen edge / corner for the overlay.
    pub anchor: OverlayAnchor,

    /// Margin from the anchored edge(s), in pixels.
    pub margin: i32,

    /// In-progress preview crossfade duration (ms); Live mode overlay-only.
    /// TOML key: `crossfade_ms` (alias: `preview_crossfade_ms`).
    #[serde(alias = "preview_crossfade_ms")]
    pub crossfade_ms: u32,

    /// Legacy name for [`Self::crossfade_ms`]; kept in sync by [`Config::normalize`].
    #[serde(skip)]
    pub preview_crossfade_ms: u32,

    /// Max edit buffer duration (seconds). Synced from `audio.max_edit_seconds` on load.
    #[serde(alias = "edit_buffer_cap_secs")]
    pub edit_buffer_cap_secs: u32,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        let crossfade_ms = 200;
        Self {
            enabled: true,
            anchor: OverlayAnchor::Bottom,
            margin: 16,
            crossfade_ms,
            preview_crossfade_ms: crossfade_ms,
            edit_buffer_cap_secs: 300,
        }
    }
}

// ─── Injection ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InjectionConfig {
    /// Enable virtual-keyboard text injection.
    pub enabled: bool,

    /// Per-chunk batch injection of committed text (default `true`).
    pub batch_mode: bool,

    /// Delay between injected characters (ms); `0` = as fast as the protocol allows.
    /// Only relevant when `batch_mode` is `false`.
    pub char_delay_ms: u32,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            batch_mode: true,
            char_delay_ms: 0,
        }
    }
}

// ─── Tray ───────────────────────────────────────────────────────────────────

/// StatusNotifierItem tray configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrayConfig {
    /// Register a tray icon when a session D-Bus bus is available.
    pub enabled: bool,
}

impl Default for TrayConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

// ─── Post-processing ────────────────────────────────────────────────────────

/// Post-processing phase configuration (text normalization on final
/// transcripts: punctuation, casing, inverse text normalization).
///
/// Models are local GGUF files from the post-processing catalog
/// (`model_catalog::POST_PROCESS_CATALOG`), managed by `stt-settings`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PostProcessConfig {
    /// Run final transcripts through the normalization model.
    pub enabled: bool,

    /// Post-processing model id from the catalog (default: `s1-mini`).
    pub model: String,

    /// Control line — register: `casual`, `semi-casual`, `semi-formal`,
    /// `formal` (model card's trained values; clamped by normalize).
    pub styling: String,

    /// Control line — layout: `prose` or `lists`.
    pub structure: String,

    /// Control line — destination conventions: `general` or `email`.
    pub context: String,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "s1-mini".into(),
            styling: "semi-formal".into(),
            structure: "prose".into(),
            context: "general".into(),
        }
    }
}

impl PostProcessConfig {
    /// `(config value, display label)` pairs for the styling axis.
    pub const STYLINGS: &'static [(&'static str, &'static str)] = &[
        ("casual", "Casual"),
        ("semi-casual", "Semi-Casual"),
        ("semi-formal", "Semi-Formal"),
        ("formal", "Formal"),
    ];

    /// `(config value, display label)` pairs for the structure axis.
    pub const STRUCTURES: &'static [(&'static str, &'static str)] =
        &[("prose", "Prose"), ("lists", "Lists")];

    /// `(config value, display label)` pairs for the context axis.
    pub const CONTEXTS: &'static [(&'static str, &'static str)] =
        &[("general", "General"), ("email", "Email")];

    /// Clamp unknown values to trained defaults so a hand-edited config can
    /// never feed out-of-distribution control lines to the model.
    pub fn normalize(&mut self) {
        self.styling = normalize_axis(&self.styling, Self::STYLINGS, "semi-formal");
        self.structure = normalize_axis(&self.structure, Self::STRUCTURES, "prose");
        self.context = normalize_axis(&self.context, Self::CONTEXTS, "general");
    }
}

fn normalize_axis(value: &str, allowed: &[(&str, &str)], fallback: &str) -> String {
    let trimmed = value.trim().to_ascii_lowercase();
    if allowed.iter().any(|(name, _)| *name == trimmed) {
        trimmed
    } else {
        fallback.to_string()
    }
}

// ─── Load / save ────────────────────────────────────────────────────────────

impl Config {
    /// Load config from `path`, or `~/.config/sttd/config.toml` if `path` is `None`.
    ///
    /// Missing file → defaults (not an error). Expands `~` in `model.path`.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = match path {
            Some(p) => expand_tilde_path(p),
            None => default_config_path(),
        };

        if !path.exists() {
            tracing::info!(
                path = %path.display(),
                "config file not found; using defaults"
            );
            return Ok(Self::default());
        }

        let text =
            fs::read_to_string(&path).with_context(|| format!("read config {}", path.display()))?;
        let mut cfg: Config =
            toml::from_str(&text).with_context(|| format!("parse config {}", path.display()))?;
        cfg.normalize();
        Ok(cfg)
    }

    /// Normalize after deserialize / default construction:
    /// - expand `~` in model path
    /// - derive path from legacy `name`+`directory` when `path` empty
    /// - sync legacy field aliases used by other modules
    pub fn normalize(&mut self) {
        // Resolve path (handles empty path → name+directory, and `~` expansion).
        let expanded = self.model.model_path();
        self.model.path = expanded.to_string_lossy().into_owned();

        // Keep display name aligned with the file name.
        if let Some(file_name) = Path::new(&self.model.path).file_name() {
            let file_name = file_name.to_string_lossy();
            if !file_name.is_empty() {
                self.model.name = file_name.into_owned();
            }
        }

        // Sync legacy audio field used by `vad` module.
        self.audio.silence_gap_ms = self.audio.vad_silence_ms;

        // The VAD and every ASR backend operate on mono 16 kHz PCM. Hand-edited
        // or legacy configs may still contain another rate/channel count; pin
        // them here instead of feeding mislabeled samples to the pipeline.
        self.audio.sample_rate = PIPELINE_SAMPLE_RATE_HZ;
        self.audio.channels = 1;
        self.audio.vad_silence_ms = self.audio.vad_silence_ms.clamp(50, 10_000);
        self.audio.silence_gap_ms = self.audio.vad_silence_ms;
        self.audio.max_edit_seconds = self.audio.max_edit_seconds.max(1);
        self.audio.stt_channel_capacity = self.audio.stt_channel_capacity.clamp(1, 256);

        // Sync legacy overlay crossfade field used by `overlay` module.
        self.overlay.preview_crossfade_ms = self.overlay.crossfade_ms;

        // Clamp post-processing control-line values to the trained sets.
        self.post_process.normalize();

        // Edit buffer cap: `audio.max_edit_seconds` is canonical (§11).
        // Legacy TOML may only set `overlay.edit_buffer_cap_secs` — mirror that
        // into audio when audio still has the default and overlay differs.
        if self.audio.max_edit_seconds != 300 {
            self.overlay.edit_buffer_cap_secs = self.audio.max_edit_seconds;
        } else if self.overlay.edit_buffer_cap_secs != 300 {
            self.audio.max_edit_seconds = self.overlay.edit_buffer_cap_secs;
        } else {
            self.overlay.edit_buffer_cap_secs = self.audio.max_edit_seconds;
        }
    }

    /// Ensure parent dirs exist and write current config (for first-run helpers).
    #[allow(dead_code)]
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create config dir {}", parent.display()))?;
        }
        let text = toml::to_string_pretty(self).context("serialize config")?;
        fs::write(path, text).with_context(|| format!("write config {}", path.display()))?;
        Ok(())
    }
}

// ─── Paths ──────────────────────────────────────────────────────────────────

/// The XDG configuration directory for `sttd`.
pub fn default_config_dir() -> PathBuf {
    if let Some(proj) = directories::ProjectDirs::from("", "", "sttd") {
        proj.config_dir().to_path_buf()
    } else {
        PathBuf::from(".")
    }
}

/// `~/.config/sttd/config.toml`
pub fn default_config_path() -> PathBuf {
    default_config_dir().join("config.toml")
}

/// `~/.local/share/sttd/models/`
pub fn default_models_dir() -> PathBuf {
    if let Some(proj) = directories::ProjectDirs::from("", "", "sttd") {
        proj.data_local_dir().join("models")
    } else {
        PathBuf::from("models")
    }
}

/// Expand a leading `~` or `~/` using the user's home directory.
pub fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

/// Persistent named configuration profiles used by `stt-settings` and
/// `sttd --profile`.
#[derive(Debug, Clone)]
pub struct ProfileStore {
    root: PathBuf,
}

impl Default for ProfileStore {
    fn default() -> Self {
        Self::new(default_config_dir())
    }
}

impl ProfileStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn profiles_dir(&self) -> PathBuf {
        self.root.join("profiles")
    }

    fn profile_path(&self, name: &str) -> Result<PathBuf> {
        validate_profile_name(name)?;
        Ok(self.profiles_dir().join(format!("{name}.toml")))
    }

    fn active_path(&self) -> PathBuf {
        self.root.join("active-profile")
    }

    /// List profile names in stable, alphabetical order.
    pub fn list(&self) -> Result<Vec<String>> {
        let dir = self.profiles_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut names = Vec::new();
        for entry in fs::read_dir(&dir)
            .with_context(|| format!("read profile directory {}", dir.display()))?
        {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
                continue;
            }
            if let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) {
                if validate_profile_name(name).is_ok() {
                    names.push(name.to_string());
                }
            }
        }
        names.sort();
        Ok(names)
    }

    pub fn load(&self, name: &str) -> Result<Config> {
        let path = self.profile_path(name)?;
        if !path.exists() {
            bail!("profile '{name}' does not exist");
        }
        Config::load(Some(&path))
    }

    /// Save a profile atomically and preserve the normalized config schema.
    pub fn save(&self, name: &str, cfg: &Config) -> Result<()> {
        let path = self.profile_path(name)?;
        fs::create_dir_all(self.profiles_dir()).with_context(|| {
            format!("create profile directory {}", self.profiles_dir().display())
        })?;

        let mut normalized = cfg.clone();
        normalized.normalize();
        let text = toml::to_string_pretty(&normalized).context("serialize profile")?;
        atomic_write(&path, text.as_bytes())
            .with_context(|| format!("write profile '{name}' at {}", path.display()))
    }

    /// Return the selected profile, if the pointer exists and is valid.
    pub fn active(&self) -> Result<Option<String>> {
        let path = self.active_path();
        if !path.exists() {
            return Ok(None);
        }
        let name = fs::read_to_string(&path)
            .with_context(|| format!("read active profile {}", path.display()))?;
        let name = name.trim();
        if name.is_empty() {
            return Ok(None);
        }
        validate_profile_name(name)?;
        Ok(Some(name.to_string()))
    }

    /// Make a profile active and mirror it to the daemon's legacy config path.
    /// Existing systemd units therefore pick it up on their next restart.
    pub fn activate(&self, name: &str) -> Result<()> {
        let cfg = self.load(name)?;
        let config_path = self.root.join("config.toml");
        let mut normalized = cfg;
        normalized.normalize();
        let text = toml::to_string_pretty(&normalized).context("serialize active profile")?;
        atomic_write(&config_path, text.as_bytes())
            .with_context(|| format!("write active config {}", config_path.display()))?;
        atomic_write(&self.active_path(), name.as_bytes()).with_context(|| {
            format!(
                "write active profile pointer {}",
                self.active_path().display()
            )
        })
    }

    /// Delete a profile while keeping at least one usable profile available.
    pub fn delete(&self, name: &str) -> Result<()> {
        let names = self.list()?;
        if !names.iter().any(|candidate| candidate == name) {
            bail!("profile '{name}' does not exist");
        }
        if names.len() == 1 {
            bail!("cannot delete the last profile");
        }

        let path = self.profile_path(name)?;
        fs::remove_file(&path)
            .with_context(|| format!("delete profile '{name}' at {}", path.display()))?;

        if self.active()?.as_deref() == Some(name) {
            let next = names
                .into_iter()
                .find(|candidate| candidate != name)
                .expect("profile count checked above");
            self.activate(&next)?;
        }
        Ok(())
    }

    /// Import the current single-config setup into the profile store on first
    /// use, then repair a missing or stale active-profile pointer.
    pub fn ensure_initialized(&self) -> Result<()> {
        fs::create_dir_all(self.profiles_dir()).with_context(|| {
            format!("create profile directory {}", self.profiles_dir().display())
        })?;

        let mut names = self.list()?;
        if names.is_empty() {
            let config_path = self.root.join("config.toml");
            let cfg = if config_path.exists() {
                Config::load(Some(&config_path))?
            } else {
                Config::default()
            };
            self.save("Default", &cfg)?;
            names.push("Default".into());
        }

        let active = self.active().ok().flatten();
        let active_is_valid = active
            .as_deref()
            .is_some_and(|name| names.iter().any(|candidate| candidate == name));
        if !active_is_valid {
            self.activate(&names[0])?;
        }
        Ok(())
    }
}

fn validate_profile_name(name: &str) -> Result<()> {
    if name.is_empty() || name == "." || name == ".." || name.len() > 64 {
        bail!("invalid profile name '{name}'");
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ' ' | '-' | '_' | '.'))
    {
        bail!("invalid profile name '{name}': use letters, numbers, spaces, '-', '_' or '.'");
    }
    Ok(())
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("path {} has no parent", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("create directory {}", parent.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("sttd-config");
    let tmp = parent.join(format!(".{file_name}.tmp-{}", std::process::id()));
    fs::write(&tmp, contents).with_context(|| format!("write temporary file {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("replace configuration file {}", path.display()))?;
    Ok(())
}

fn expand_tilde_path(path: &Path) -> PathBuf {
    match path.to_str() {
        Some(s) => expand_tilde(s),
        None => path.to_path_buf(),
    }
}

fn home_dir() -> Option<PathBuf> {
    directories::UserDirs::new().map(|u| u.home_dir().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_architecture() {
        let cfg = Config::default();
        assert_eq!(cfg.audio.sample_rate, 16_000);
        assert_eq!(cfg.audio.vad_silence_ms, 500);
        assert_eq!(cfg.audio.silence_gap_ms, 500);
        assert_eq!(cfg.audio.max_edit_seconds, 300);
        assert_eq!(cfg.model.residency, ModelResidency::Preload);
        assert!(cfg.model.path.ends_with("qwen3-asr-0.6b-q8_0.gguf"));
        assert!(!cfg.model.path.starts_with('~'));
        assert_eq!(cfg.overlay.anchor, OverlayAnchor::Bottom);
        assert_eq!(cfg.overlay.margin, 16);
        assert_eq!(cfg.overlay.crossfade_ms, 200);
        assert_eq!(cfg.overlay.preview_crossfade_ms, 200);
        assert_eq!(cfg.overlay.edit_buffer_cap_secs, 300);
        assert!(cfg.injection.batch_mode);
        assert!(cfg.tray.enabled);
    }

    #[test]
    fn normalize_forces_pipeline_audio_format() {
        let mut cfg = Config::default();
        cfg.audio.sample_rate = 48_000;
        cfg.audio.channels = 2;
        cfg.audio.vad_silence_ms = 0;
        cfg.audio.max_edit_seconds = 0;
        cfg.audio.stt_channel_capacity = 0;

        cfg.normalize();

        assert_eq!(cfg.audio.sample_rate, PIPELINE_SAMPLE_RATE_HZ);
        assert_eq!(cfg.audio.channels, 1);
        assert_eq!(cfg.audio.vad_silence_ms, 50);
        assert_eq!(cfg.audio.silence_gap_ms, 50);
        assert_eq!(cfg.audio.max_edit_seconds, 1);
        assert_eq!(cfg.audio.stt_channel_capacity, 1);
    }

    #[test]
    fn expand_tilde_home() {
        let home = home_dir().expect("home");
        assert_eq!(expand_tilde("~"), home);
        assert_eq!(expand_tilde("~/models/x.bin"), home.join("models/x.bin"));
        assert_eq!(expand_tilde("/abs/path"), PathBuf::from("/abs/path"));
    }

    #[test]
    fn parse_section_11_toml() {
        let toml = r#"
[audio]
sample_rate = 16000
vad_silence_ms = 600
max_edit_seconds = 120

[model]
path = "~/custom/models/ggml-small.en.bin"
residency = "lazy"

[overlay]
anchor = "top_right"
margin = 24
crossfade_ms = 150

[injection]
batch_mode = true

[tray]
enabled = false
"#;
        let mut cfg: Config = toml::from_str(toml).unwrap();
        cfg.normalize();

        assert_eq!(cfg.audio.vad_silence_ms, 600);
        assert_eq!(cfg.audio.silence_gap_ms, 600);
        assert_eq!(cfg.audio.max_edit_seconds, 120);
        assert_eq!(cfg.overlay.edit_buffer_cap_secs, 120);
        assert_eq!(cfg.model.residency, ModelResidency::Lazy);
        assert!(cfg.model.path.ends_with("ggml-small.en.bin"));
        assert!(!cfg.model.path.contains('~'));
        assert_eq!(cfg.model.name, "ggml-small.en.bin");
        assert_eq!(cfg.overlay.anchor, OverlayAnchor::TopRight);
        assert_eq!(cfg.overlay.margin, 24);
        assert_eq!(cfg.overlay.crossfade_ms, 150);
        assert_eq!(cfg.overlay.preview_crossfade_ms, 150);
        assert!(cfg.injection.batch_mode);
        assert!(!cfg.tray.enabled);
        // Post-processing defaults: off, s1-mini, semi-formal/prose/general.
        assert!(!cfg.post_process.enabled);
        assert_eq!(cfg.post_process.model, "s1-mini");
        assert_eq!(cfg.post_process.styling, "semi-formal");
        assert_eq!(cfg.post_process.structure, "prose");
        assert_eq!(cfg.post_process.context, "general");
    }

    #[test]
    fn post_process_toml_and_axis_clamping() {
        let toml = r#"
[post_process]
enabled = true
model = "s1-mini"
styling = "Formal"
structure = "lists"
context = "email"
"#;
        let mut cfg: Config = toml::from_str(toml).unwrap();
        cfg.normalize();

        assert!(cfg.post_process.enabled);
        assert_eq!(cfg.post_process.model, "s1-mini");
        assert_eq!(cfg.post_process.styling, "formal");
        assert_eq!(cfg.post_process.structure, "lists");
        assert_eq!(cfg.post_process.context, "email");

        // Unknown axis values fall back to the trained defaults.
        let mut cfg: Config = toml::from_str(
            r#"
[post_process]
styling = "shouty"
structure = "haiku"
context = "slack"
"#,
        )
        .unwrap();
        cfg.normalize();
        assert_eq!(cfg.post_process.styling, "semi-formal");
        assert_eq!(cfg.post_process.structure, "prose");
        assert_eq!(cfg.post_process.context, "general");
    }

    #[test]
    fn legacy_toml_aliases() {
        let toml = r#"
[audio]
silence_gap_ms = 400

[model]
name = "ggml-small.en.bin"
residency = "preload"

[overlay]
preview_crossfade_ms = 90
edit_buffer_cap_secs = 60
"#;
        let mut cfg: Config = toml::from_str(toml).unwrap();
        cfg.normalize();

        assert_eq!(cfg.audio.vad_silence_ms, 400);
        assert_eq!(cfg.audio.silence_gap_ms, 400);
        assert!(cfg.model.path.ends_with("ggml-small.en.bin"));
        assert_eq!(cfg.overlay.crossfade_ms, 90);
        assert_eq!(cfg.audio.max_edit_seconds, 60);
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let path = PathBuf::from("/tmp/sttd-config-definitely-missing-xyz.toml");
        let cfg = Config::load(Some(&path)).unwrap();
        assert_eq!(cfg.audio.sample_rate, 16_000);
        assert_eq!(cfg.model.residency, ModelResidency::Preload);
    }

    #[test]
    fn profile_store_imports_switches_and_deletes_profiles() {
        let root = std::env::temp_dir().join(format!(
            "sttd-profile-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = ProfileStore::new(root.clone());

        store.ensure_initialized().unwrap();
        assert_eq!(store.list().unwrap(), vec!["Default"]);
        assert_eq!(store.active().unwrap().as_deref(), Some("Default"));

        let mut cfg = Config::default();
        cfg.model.path = "~/models/ggml-small.en.bin".into();
        store.save("Work", &cfg).unwrap();
        store.activate("Work").unwrap();
        assert_eq!(store.active().unwrap().as_deref(), Some("Work"));
        assert!(Config::load(Some(&root.join("config.toml")))
            .unwrap()
            .model
            .path
            .ends_with("ggml-small.en.bin"));

        store.delete("Work").unwrap();
        assert_eq!(store.active().unwrap().as_deref(), Some("Default"));
        assert!(store.delete("Default").is_err());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn profile_names_cannot_escape_profile_directory() {
        let root = std::env::temp_dir().join(format!(
            "sttd-profile-name-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = ProfileStore::new(root.clone());
        assert!(store.save("../escape", &Config::default()).is_err());
        assert!(store.save("valid-profile", &Config::default()).is_ok());
        let _ = std::fs::remove_dir_all(root);
    }
}
