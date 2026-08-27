//! Curated audio.cpp-compatible speech recognition models and safe local
//! model-file management.

use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    pub family: &'static str,
    pub languages: &'static str,
    pub description: &'static str,
    pub default_language: &'static str,
    pub license: &'static str,
    pub streaming: bool,
    pub recommended: bool,
}

pub const MODEL_CATALOG: &[ModelSpec] = &[
    ModelSpec {
        id: "qwen3-asr-0.6b-q8",
        name: "Qwen3 ASR 0.6B (Q8_0)",
        filename: "qwen3-asr-0.6b-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/c3857f1ec35cfea8993924e7c2a6f682b5dc060b/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf",
        sha256: "6c44ec2fb4cee513892d7863c1fcc3ea6b699ffa4d899b0ef4ab19956d9544f7",
        size_bytes: 1_151_272_416,
        family: "Qwen3 ASR",
        languages: "30 languages",
        description: "Modern multilingual model; the best all-round default.",
        default_language: "auto",
        license: "Apache-2.0",
        streaming: true,
        recommended: true,
    },
    ModelSpec {
        id: "parakeet-tdt-0.6b-v3-q8",
        name: "Parakeet TDT 0.6B v3 (Q8_0)",
        filename: "parakeet-tdt-0.6b-v3-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/c3857f1ec35cfea8993924e7c2a6f682b5dc060b/Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf",
        sha256: "074e61ac1abd3d3efcfc10798d17bf9a975b31768b466fc2578364d206dde64c",
        size_bytes: 915_733_744,
        family: "NVIDIA FastConformer TDT",
        languages: "25 European languages",
        description: "Fast multilingual transcription with word-level timestamps.",
        default_language: "en",
        license: "CC-BY-4.0",
        streaming: false,
        recommended: false,
    },
    ModelSpec {
        id: "nemotron-3.5-asr-streaming-0.6b-q8",
        name: "Nemotron 3.5 ASR Streaming 0.6B (Q8_0)",
        filename: "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/c3857f1ec35cfea8993924e7c2a6f682b5dc060b/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
        sha256: "7026c80d8b94b5184c3acc20b34c410787a35babab7ec33931bd2913cd022afe",
        size_bytes: 930_620_256,
        family: "NVIDIA FastConformer RNN-T",
        languages: "32 locales",
        description: "Low-latency multilingual model with a streaming architecture.",
        default_language: "en-US",
        license: "NVIDIA Open Model License",
        streaming: true,
        recommended: false,
    },
    ModelSpec {
        id: "qwen3-asr-1.7b-q8",
        name: "Qwen3 ASR 1.7B (Q8_0)",
        filename: "qwen3-asr-1.7b-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/c3857f1ec35cfea8993924e7c2a6f682b5dc060b/Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf",
        sha256: "da4fc2ac7f24dee784d1684eb1f35836cdbf559519452ae11777670734c0a4f8",
        size_bytes: 2_473_010_048,
        family: "Qwen3 ASR",
        languages: "30 languages",
        description: "Higher-accuracy Qwen3 variant for systems with more memory.",
        default_language: "auto",
        license: "Apache-2.0",
        streaming: true,
        recommended: false,
    },
    ModelSpec {
        id: "citrinet-asr-q8",
        name: "Citrinet ASR (Q8_0)",
        filename: "citrinet-asr-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/c3857f1ec35cfea8993924e7c2a6f682b5dc060b/Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf",
        sha256: "e1b9187ecc1c5770f5785f1d1e797ff2bf1f3692c78a5328684275aa729c6368",
        size_bytes: 40_574_432,
        family: "NVIDIA Citrinet",
        languages: "English",
        description: "Tiny 40 MB English model for low-memory systems.",
        default_language: "en",
        license: "CC-BY-4.0",
        streaming: false,
        recommended: false,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallState {
    Missing,
    Installed,
    Invalid,
}

pub fn managed_model_path(models_dir: &Path, spec: &ModelSpec) -> Option<PathBuf> {
    let filename = Path::new(spec.filename);
    let mut components = filename.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Some(models_dir.join(filename)),
        _ => None,
    }
}

pub fn model_for_filename(filename: &str) -> Option<&'static ModelSpec> {
    MODEL_CATALOG.iter().find(|spec| spec.filename == filename)
}

pub fn install_state(models_dir: &Path, spec: &ModelSpec) -> InstallState {
    let Some(path) = managed_model_path(models_dir, spec) else {
        return InstallState::Invalid;
    };
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() && metadata.len() == spec.size_bytes => {
            InstallState::Installed
        }
        Ok(_) => InstallState::Invalid,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => InstallState::Missing,
        Err(_) => InstallState::Invalid,
    }
}

pub fn format_size(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    format!("{:.0} MiB", bytes as f64 / MIB)
}

pub fn download_model(
    models_dir: &Path,
    spec: &ModelSpec,
    cancel: &AtomicBool,
    progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    let destination = managed_model_path(models_dir, spec)
        .with_context(|| format!("invalid model filename {}", spec.filename))?;
    download_verified_file(
        models_dir,
        destination,
        spec.filename,
        spec.url,
        spec.sha256,
        spec.size_bytes,
        spec.name,
        cancel,
        progress,
    )
}

/// Shared download core for both catalogs: verifies an already-installed file,
/// or streams `url` into a hidden partial file and atomically hard-links it to
/// `destination` after size + SHA-256 verification.
fn download_verified_file(
    dir: &Path,
    destination: PathBuf,
    filename: &str,
    url: &str,
    sha256: &str,
    size_bytes: u64,
    display_name: &str,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("creating model directory {}", dir.display()))?;

    match fs::symlink_metadata(&destination) {
        Ok(metadata) if metadata.file_type().is_file() && metadata.len() == size_bytes => {
            verify_file_digest(&destination, sha256, size_bytes)?;
            return Ok(destination);
        }
        Ok(_) => {
            bail!(
                "{} already exists but does not match the catalog; delete it before downloading",
                destination.display()
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => {
            bail!(
                "{} already exists but does not match the catalog; delete it before downloading",
                destination.display()
            );
        }
    }

    if cancel.load(Ordering::Acquire) {
        bail!("download cancelled");
    }

    static PART_COUNTER: AtomicU64 = AtomicU64::new(0);
    let part_number = PART_COUNTER.fetch_add(1, Ordering::Relaxed);
    let part_name = format!(".{filename}.part-{}-{part_number}", std::process::id());
    let part_path = dir.join(part_name);

    let result = (|| -> Result<()> {
        let response = ureq::get(url)
            .set("User-Agent", concat!("sttd/", env!("CARGO_PKG_VERSION")))
            .call()
            .with_context(|| format!("downloading {display_name}"))?;
        if let Some(length) = response
            .header("Content-Length")
            .and_then(|value| value.parse::<u64>().ok())
        {
            if length != size_bytes {
                bail!(
                    "server reported {length} bytes for {display_name}, expected {size_bytes}"
                );
            }
        }

        let mut reader = response.into_reader();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&part_path)
            .with_context(|| format!("creating partial file {}", part_path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        let mut downloaded = 0_u64;

        loop {
            if cancel.load(Ordering::Acquire) {
                bail!("download cancelled");
            }
            let count = reader.read(&mut buffer).context("reading model download")?;
            if count == 0 {
                break;
            }
            downloaded += count as u64;
            if downloaded > size_bytes {
                bail!("download exceeded the expected size for {display_name}");
            }
            file.write_all(&buffer[..count])
                .context("writing partial model")?;
            hasher.update(&buffer[..count]);
            progress(downloaded, size_bytes);
        }

        if downloaded != size_bytes {
            bail!("downloaded {downloaded} bytes for {display_name}, expected {size_bytes}");
        }
        let actual_hash = format!("{:x}", hasher.finalize());
        if actual_hash != sha256 {
            bail!(
                "SHA-256 mismatch for {display_name} (expected {sha256}, got {actual_hash})"
            );
        }

        file.flush().context("flushing partial model")?;
        file.sync_all().context("syncing partial model")?;
        drop(file);

        // Ensure the destination's parent exists before creating the hard
        // link (models in the post-processing catalog live in subdirectories).
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating destination directory {}", parent.display()))?;
        }
        // Both paths are in the same directory. Creating a hard link exposes
        // only the complete file and fails instead of replacing an existing one.
        fs::hard_link(&part_path, &destination)
            .with_context(|| format!("installing model at {}", destination.display()))?;
        fs::remove_file(&part_path)
            .with_context(|| format!("removing partial file {}", part_path.display()))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&part_path);
    }
    result?;
    Ok(destination)
}

pub fn delete_managed_model(models_dir: &Path, path: &Path) -> Result<()> {
    let filename = path
        .file_name()
        .filter(|name| is_model_filename(name))
        .context("model path does not have a supported filename")?;
    let expected = models_dir.join(filename);
    if path != expected {
        bail!("refusing to delete a model outside the managed directory");
    }

    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("reading model metadata for {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("refusing to delete a symlink or non-regular model file");
    }

    let canonical_dir = models_dir
        .canonicalize()
        .with_context(|| format!("resolving model directory {}", models_dir.display()))?;
    let canonical_parent = path
        .parent()
        .context("model path has no parent")?
        .canonicalize()
        .with_context(|| format!("resolving model parent for {}", path.display()))?;
    if canonical_parent != canonical_dir {
        bail!("refusing to delete a model outside the managed directory");
    }

    fs::remove_file(path).with_context(|| format!("deleting model {}", path.display()))
}

fn is_model_filename(name: &OsStr) -> bool {
    let path = Path::new(name);
    path.components().count() == 1
        && path
            .extension()
            .and_then(OsStr::to_str)
            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin") || ext.eq_ignore_ascii_case("gguf"))
}

fn verify_file_digest(path: &Path, sha256: &str, size_bytes: u64) -> Result<()> {
    let mut file =
        File::open(path).with_context(|| format!("opening installed model {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("reading installed model metadata for {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.len() != size_bytes {
        bail!("installed model does not match the expected size");
    }

    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).context("verifying model file")?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual_hash = format!("{:x}", hasher.finalize());
    if actual_hash != sha256 {
        bail!("installed model failed SHA-256 verification");
    }
    Ok(())
}

// ─── Post-processing catalog ────────────────────────────────────────────────

/// A curated post-processing (text normalization) model.
///
/// Each entry ships two verified files: quantized GGUF weights and the
/// matching `tokenizer.json`. Files live under
/// `~/.local/share/sttd/models/post-process/<id>/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostProcessModelSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    pub tokenizer_filename: &'static str,
    pub tokenizer_url: &'static str,
    pub tokenizer_sha256: &'static str,
    pub tokenizer_size_bytes: u64,
    pub family: &'static str,
    pub languages: &'static str,
    pub description: &'static str,
    pub license: &'static str,
    pub recommended: bool,
}

pub const POST_PROCESS_CATALOG: &[PostProcessModelSpec] = &[PostProcessModelSpec {
    id: "s1-mini",
    name: "S1-mini by Superwhisper (Q4_K_M)",
    filename: "s1-mini-q4_k_m.gguf",
    url: "https://huggingface.co/superwhisper/s1-mini-GGUF/resolve/ee2c0f56e56345f475749a44ff2893e21c3cb292/s1-mini-q4_k_m.gguf",
    sha256: "3b41ebe2502cbd03e811d5d16b022f5ab551eda58d62597d152f89535003c634",
    size_bytes: 484_219_808,
    tokenizer_filename: "tokenizer.json",
    tokenizer_url: "https://huggingface.co/superwhisper/s1-mini/resolve/ae016a2cc6e42d0298932503b0874f048a947f40/tokenizer.json",
    tokenizer_sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    tokenizer_size_bytes: 11_422_654,
    family: "Qwen3 0.6B (text normalizer)",
    languages: "English",
    description: "Adds punctuation and casing, expands spoken numbers/dates/emails, and removes fillers after recognition.",
    license: "Apache-2.0 + naming clause",
    recommended: true,
}];

pub fn post_process_spec_for_id(id: &str) -> Option<&'static PostProcessModelSpec> {
    POST_PROCESS_CATALOG.iter().find(|spec| spec.id == id)
}

/// `~/.local/share/sttd/models/post-process/`
pub fn post_process_models_dir() -> PathBuf {
    crate::config::default_models_dir().join("post-process")
}

fn post_process_models_dir_in(models_dir: &Path) -> PathBuf {
    models_dir.join("post-process")
}

/// Per-model directory holding the GGUF and tokenizer files.
pub fn post_process_model_dir(
    models_dir: &Path,
    spec: &PostProcessModelSpec,
) -> Option<PathBuf> {
    if spec.id.is_empty()
        || !spec
            .id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        return None;
    }
    Some(post_process_models_dir_in(models_dir).join(spec.id))
}

fn single_component_path(parent: &Path, filename: &str) -> Option<PathBuf> {
    let filename = Path::new(filename);
    let mut components = filename.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Some(parent.join(filename)),
        _ => None,
    }
}

pub fn post_process_model_path(
    models_dir: &Path,
    spec: &PostProcessModelSpec,
) -> Option<PathBuf> {
    post_process_model_dir(models_dir, spec)
        .and_then(|dir| single_component_path(&dir, spec.filename))
}

pub fn post_process_tokenizer_path(
    models_dir: &Path,
    spec: &PostProcessModelSpec,
) -> Option<PathBuf> {
    post_process_model_dir(models_dir, spec)
        .and_then(|dir| single_component_path(&dir, spec.tokenizer_filename))
}

/// Installed only when both the GGUF and the tokenizer are present with the
/// expected sizes.
pub fn post_process_install_state(
    models_dir: &Path,
    spec: &PostProcessModelSpec,
) -> InstallState {
    let (Some(model_path), Some(tokenizer_path)) = (
        post_process_model_path(models_dir, spec),
        post_process_tokenizer_path(models_dir, spec),
    ) else {
        return InstallState::Invalid;
    };

    let state_for = |path: &Path, size_bytes: u64| match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() && metadata.len() == size_bytes => {
            InstallState::Installed
        }
        Ok(_) => InstallState::Invalid,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => InstallState::Missing,
        Err(_) => InstallState::Invalid,
    };

    let model_state = state_for(&model_path, spec.size_bytes);
    let tokenizer_state = state_for(&tokenizer_path, spec.tokenizer_size_bytes);
    if model_state == InstallState::Installed && tokenizer_state == InstallState::Installed {
        InstallState::Installed
    } else if model_state == InstallState::Invalid || tokenizer_state == InstallState::Invalid {
        InstallState::Invalid
    } else {
        InstallState::Missing
    }
}

/// Download (or verify) the GGUF and tokenizer, then return the GGUF path.
///
/// Progress is reported over the combined byte total of both files.
pub fn download_post_process_model(
    models_dir: &Path,
    spec: &PostProcessModelSpec,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    let model_path = post_process_model_path(models_dir, spec)
        .with_context(|| format!("invalid post-processing model filename {}", spec.filename))?;
    let tokenizer_path = post_process_tokenizer_path(models_dir, spec).with_context(|| {
        format!(
            "invalid post-processing tokenizer filename {}",
            spec.tokenizer_filename
        )
    })?;

    // Create the per-model subdirectory so the partial files' parent exists.
    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating model directory {}", parent.display()))?;
    }

    let total = spec.size_bytes + spec.tokenizer_size_bytes;
    let tokenizer_installed = fs::symlink_metadata(&tokenizer_path)
        .map(|metadata| {
            metadata.file_type().is_file() && metadata.len() == spec.tokenizer_size_bytes
        })
        .unwrap_or(false);

    // Tokenizer first: it is small, so a cancelled GGUF fetch leaves the
    // install in an easy-to-resume state instead of a long partial download.
    if tokenizer_installed {
        verify_file_digest(
            &tokenizer_path,
            spec.tokenizer_sha256,
            spec.tokenizer_size_bytes,
        )?;
        progress(spec.tokenizer_size_bytes, total);
    } else {
        download_verified_file(
            models_dir,
            tokenizer_path,
            spec.tokenizer_filename,
            spec.tokenizer_url,
            spec.tokenizer_sha256,
            spec.tokenizer_size_bytes,
            &format!("{} tokenizer", spec.name),
            cancel,
            |downloaded, _| progress(downloaded, total),
        )?;
    }

    download_verified_file(
        models_dir,
        model_path.clone(),
        spec.filename,
        spec.url,
        spec.sha256,
        spec.size_bytes,
        spec.name,
        cancel,
        move |downloaded, _| progress(spec.tokenizer_size_bytes + downloaded, total),
    )?;
    Ok(model_path)
}

/// Delete a post-processing model directory (both files) after checking it is
/// the managed location and contains only regular files.
pub fn delete_post_process_model(models_dir: &Path, spec: &PostProcessModelSpec) -> Result<()> {
    let Some(dir) = post_process_model_dir(models_dir, spec) else {
        bail!("post-processing model has an invalid managed id");
    };
    let canonical_dir = dir
        .canonicalize()
        .with_context(|| format!("resolving model directory {}", dir.display()))?;
    for entry in fs::read_dir(&canonical_dir)
        .with_context(|| format!("reading model directory {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("reading model directory {}", dir.display()))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("reading metadata for {}", path.display()))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            bail!("refusing to delete a symlink or non-regular model file");
        }
    }
    fs::remove_dir_all(&canonical_dir)
        .with_context(|| format!("deleting model directory {}", dir.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_entries_are_unique_and_pinned() {
        for (index, spec) in MODEL_CATALOG.iter().enumerate() {
            assert!(!spec.id.is_empty());
            assert!(MODEL_CATALOG[..index]
                .iter()
                .all(|other| other.id != spec.id));
            assert!(MODEL_CATALOG[..index]
                .iter()
                .all(|other| other.filename != spec.filename));
            assert!(managed_model_path(Path::new("/models"), spec).is_some());
            assert_eq!(spec.sha256.len(), 64);
            assert!(spec.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert!(spec.url.starts_with("https://huggingface.co/"));
            assert!(spec.url.contains("/resolve/"));
            assert!(!spec.url.contains("/resolve/main/"));
            // Citrinet q8 is legitimately ~40 MB; anything smaller is a typo.
            assert!(spec.size_bytes > 40_000_000);
            assert!(!spec.license.is_empty());
        }
    }

    #[test]
    fn filename_lookup_is_exact() {
        assert_eq!(
            model_for_filename("qwen3-asr-0.6b-q8_0.gguf").map(|spec| spec.id),
            Some("qwen3-asr-0.6b-q8")
        );
        assert!(model_for_filename("../qwen3-asr-0.6b-q8_0.gguf").is_none());
        assert!(model_for_filename("ggml-base.en.bin").is_none());
    }

    #[test]
    fn delete_rejects_paths_outside_managed_directory() {
        let root = test_dir("outside-delete");
        let models = root.join("models");
        let outside = root.join("outside.gguf");
        fs::create_dir_all(&models).unwrap();
        fs::write(&outside, b"model").unwrap();

        let error = delete_managed_model(&models, &outside).unwrap_err();
        assert!(error.to_string().contains("outside the managed directory"));
        assert!(outside.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn delete_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let root = test_dir("symlink-delete");
        let models = root.join("models");
        let outside = root.join("outside.gguf");
        let link = models.join("linked.gguf");
        fs::create_dir_all(&models).unwrap();
        fs::write(&outside, b"model").unwrap();
        symlink(&outside, &link).unwrap();

        let error = delete_managed_model(&models, &link).unwrap_err();
        assert!(error.to_string().contains("symlink"));
        assert!(outside.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn download_verifies_and_installs_without_exposing_a_partial_file() {
        let body = b"tiny verified model fixture";
        let root = test_dir("download-success");
        let models = root.join("models");
        let (url, server) = serve_once(body);
        let hash: &'static str = Box::leak(format!("{:x}", Sha256::digest(body)).into_boxed_str());
        let spec = fixture_spec(url, hash, body.len() as u64);
        let cancel = AtomicBool::new(false);
        let mut last_progress = 0;

        let installed = download_model(&models, &spec, &cancel, |downloaded, total| {
            assert_eq!(total, body.len() as u64);
            last_progress = downloaded;
        })
        .unwrap();
        server.join().unwrap();

        assert_eq!(fs::read(&installed).unwrap(), body);
        assert_eq!(last_progress, body.len() as u64);
        assert_eq!(install_state(&models, &spec), InstallState::Installed);
        assert_eq!(fs::read_dir(&models).unwrap().count(), 1);
        delete_managed_model(&models, &installed).unwrap();
        assert!(!installed.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn download_cancellation_leaves_no_file() {
        let root = test_dir("download-cancel");
        let models = root.join("models");
        let spec = fixture_spec(
            "http://127.0.0.1:9/not-requested.gguf",
            "0000000000000000000000000000000000000000000000000000000000000000",
            1,
        );
        let cancel = AtomicBool::new(true);

        let error = download_model(&models, &spec, &cancel, |_, _| {}).unwrap_err();

        assert!(error.to_string().contains("cancelled"));
        assert_eq!(fs::read_dir(&models).unwrap().count(), 0);
        let _ = fs::remove_dir_all(root);
    }

    fn fixture_spec(url: &'static str, sha256: &'static str, size_bytes: u64) -> ModelSpec {
        ModelSpec {
            id: "fixture",
            name: "Fixture",
            filename: "fixture.gguf",
            url,
            sha256,
            size_bytes,
            family: "fixture",
            languages: "fixture",
            description: "fixture",
            default_language: "auto",
            license: "fixture",
            streaming: false,
            recommended: false,
        }
    }

    fn serve_once(body: &'static [u8]) -> (&'static str, std::thread::JoinHandle<()>) {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let url = Box::leak(format!("http://{address}/fixture.gguf").into_boxed_str());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(body).unwrap();
            stream.flush().unwrap();
        });
        (url, server)
    }

    fn test_dir(name: &str) -> PathBuf {
        static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
        let suffix = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("sttd-{name}-{}-{suffix}", std::process::id()))
    }

    fn pp_fixture_spec(
        model_url: &'static str,
        model_sha256: &'static str,
        tokenizer_url: &'static str,
        tokenizer_sha256: &'static str,
        model_size: u64,
        tokenizer_size: u64,
    ) -> PostProcessModelSpec {
        PostProcessModelSpec {
            id: "fixture",
            name: "Fixture",
            filename: "fixture.gguf",
            url: model_url,
            sha256: model_sha256,
            size_bytes: model_size,
            tokenizer_filename: "tokenizer.json",
            tokenizer_url,
            tokenizer_sha256,
            tokenizer_size_bytes: tokenizer_size,
            family: "fixture",
            languages: "fixture",
            description: "fixture",
            license: "fixture",
            recommended: false,
        }
    }

    #[test]
    fn post_process_catalog_entries_are_unique_and_pinned() {
        for (index, spec) in POST_PROCESS_CATALOG.iter().enumerate() {
            assert!(!spec.id.is_empty());
            assert!(POST_PROCESS_CATALOG[..index]
                .iter()
                .all(|other| other.id != spec.id));
            assert_eq!(spec.sha256.len(), 64);
            assert!(spec.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert_eq!(spec.tokenizer_sha256.len(), 64);
            assert!(spec.url.starts_with("https://huggingface.co/"));
            assert!(spec.url.contains("/resolve/"));
            assert!(!spec.url.contains("/resolve/main/"));
            assert!(spec.tokenizer_url.starts_with("https://huggingface.co/"));
            assert!(spec.tokenizer_url.contains("/resolve/"));
            assert!(!spec.tokenizer_url.contains("/resolve/main/"));
            assert!(post_process_model_path(Path::new("/models"), spec).is_some());
            assert!(post_process_tokenizer_path(Path::new("/models"), spec).is_some());
            assert!(spec.size_bytes > 100_000_000);
            assert!(spec.tokenizer_size_bytes > 1_000_000);
            assert!(!spec.license.is_empty());
            // The s1-mini license requires the exact name and attribution.
            assert!(spec.name.contains("S1-mini"));
            assert!(spec.name.contains("Superwhisper"));
        }
    }

    #[test]
    fn post_process_install_state_requires_both_files() {
        let root = test_dir("pp-install-state");
        let models = root.join("models");
        let spec = pp_fixture_spec(
            "http://127.0.0.1:9/model.gguf",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "http://127.0.0.1:9/tokenizer.json",
            "0000000000000000000000000000000000000000000000000000000000000000",
            8,
            8,
        );

        assert_eq!(
            post_process_install_state(&models, &spec),
            InstallState::Missing
        );
        let tokenizer = post_process_tokenizer_path(&models, &spec).unwrap();
        std::fs::create_dir_all(tokenizer.parent().unwrap()).unwrap();
        std::fs::write(&tokenizer, b"12345678").unwrap();
        assert_eq!(
            post_process_install_state(&models, &spec),
            InstallState::Missing
        );
        let model = post_process_model_path(&models, &spec).unwrap();
        std::fs::write(&model, b"12345678").unwrap();
        assert_eq!(
            post_process_install_state(&models, &spec),
            InstallState::Installed
        );
        std::fs::write(&model, b"wrong!!").unwrap();
        assert_eq!(
            post_process_install_state(&models, &spec),
            InstallState::Invalid
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn post_process_download_installs_both_files() {
        let model_body: &'static [u8] = Box::leak(b"tiny model fixture".to_vec().into_boxed_slice());
        let tokenizer_body: &'static [u8] =
            Box::leak(b"{\"tiny\":true}".to_vec().into_boxed_slice());
        let (model_url, model_server) = serve_once(model_body);
        let (tokenizer_url, tokenizer_server) = serve_once(tokenizer_body);
        let model_hash: &'static str =
            Box::leak(format!("{:x}", Sha256::digest(model_body)).into_boxed_str());
        let tokenizer_hash: &'static str =
            Box::leak(format!("{:x}", Sha256::digest(tokenizer_body)).into_boxed_str());
        let spec = pp_fixture_spec(
            model_url,
            model_hash,
            tokenizer_url,
            tokenizer_hash,
            model_body.len() as u64,
            tokenizer_body.len() as u64,
        );

        let root = test_dir("pp-download-success");
        let models = root.join("models");
        let cancel = AtomicBool::new(false);
        let mut last_progress = (0_u64, 0_u64);
        let installed =
            download_post_process_model(&models, &spec, &cancel, |downloaded, total| {
                last_progress = (downloaded, total);
            })
            .unwrap();
        tokenizer_server.join().unwrap();
        model_server.join().unwrap();

        assert_eq!(std::fs::read(&installed).unwrap(), model_body);
        let tokenizer = post_process_tokenizer_path(&models, &spec).unwrap();
        assert_eq!(std::fs::read(&tokenizer).unwrap(), tokenizer_body);
        let total = (model_body.len() + tokenizer_body.len()) as u64;
        assert_eq!(last_progress, (total, total));
        assert_eq!(
            post_process_install_state(&models, &spec),
            InstallState::Installed
        );

        delete_post_process_model(&models, &spec).unwrap();
        assert!(!post_process_model_dir(&models, &spec).unwrap().exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
