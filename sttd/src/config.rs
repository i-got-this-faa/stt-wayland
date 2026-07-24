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
//!
//! Legacy TOML keys (`silence_gap_ms`, `preview_crossfade_ms`, `name`+`directory`, …)
//! are accepted via serde aliases / post-load normalization.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ─── Top-level ──────────────────────────────────────────────────────────────

/// Top-level daemon configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub model: ModelConfig,
    pub overlay: OverlayConfig,
    pub injection: InjectionConfig,
}

impl Default for Config {
    fn default() -> Self {
        let mut cfg = Self {
            audio: AudioConfig::default(),
            model: ModelConfig::default(),
            overlay: OverlayConfig::default(),
            injection: InjectionConfig::default(),
        };
        cfg.normalize();
        cfg
    }
}

// ─── Audio ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Capture sample rate (Hz). Whisper + webrtc-vad expect 16_000.
    pub sample_rate: u32,

    /// VAD silence gap that closes a speech chunk (milliseconds).
    /// TOML key: `vad_silence_ms` (alias: `silence_gap_ms`).
    #[serde(alias = "silence_gap_ms")]
    pub vad_silence_ms: u32,

    /// Max edit-mode buffer duration in seconds (default 300 ≈ 5 minutes).
    pub max_edit_seconds: u32,

    /// Number of channels (mono).
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

/// When the whisper model is loaded into memory.
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
    /// Path to the ggml model file. Supports leading `~`.
    ///
    /// Empty means “derive from [`Self::name`] + [`Self::directory`]”
    /// (legacy TOML). After [`Config::load`], always the expanded path.
    /// Default resolved path: `~/.local/share/sttd/models/ggml-base.en.bin`.
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
    "ggml-base.en.bin".into()
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
        let dir = self
            .directory
            .clone()
            .unwrap_or_else(default_models_dir);
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

        let text = fs::read_to_string(&path)
            .with_context(|| format!("read config {}", path.display()))?;
        let mut cfg: Config = toml::from_str(&text)
            .with_context(|| format!("parse config {}", path.display()))?;
        cfg.normalize();
        Ok(cfg)
    }

    /// Normalize after deserialize / default construction:
    /// - expand `~` in model path
    /// - derive path from legacy `name`+`directory` when `path` empty
    /// - sync legacy field aliases used by other modules
    fn normalize(&mut self) {
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

        // Sync legacy overlay crossfade field used by `overlay` module.
        self.overlay.preview_crossfade_ms = self.overlay.crossfade_ms;

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

/// `~/.config/sttd/config.toml`
pub fn default_config_path() -> PathBuf {
    if let Some(proj) = directories::ProjectDirs::from("", "", "sttd") {
        proj.config_dir().join("config.toml")
    } else {
        PathBuf::from("config.toml")
    }
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
        assert!(cfg.model.path.ends_with("ggml-base.en.bin"));
        assert!(!cfg.model.path.starts_with('~'));
        assert_eq!(cfg.overlay.anchor, OverlayAnchor::Bottom);
        assert_eq!(cfg.overlay.margin, 16);
        assert_eq!(cfg.overlay.crossfade_ms, 200);
        assert_eq!(cfg.overlay.preview_crossfade_ms, 200);
        assert_eq!(cfg.overlay.edit_buffer_cap_secs, 300);
        assert!(cfg.injection.batch_mode);
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
}
