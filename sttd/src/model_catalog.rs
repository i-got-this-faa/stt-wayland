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
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    let destination = managed_model_path(models_dir, spec)
        .with_context(|| format!("invalid model filename {}", spec.filename))?;
    fs::create_dir_all(models_dir)
        .with_context(|| format!("creating model directory {}", models_dir.display()))?;

    match install_state(models_dir, spec) {
        InstallState::Installed => {
            verify_model_file(&destination, spec)?;
            return Ok(destination);
        }
        InstallState::Invalid => {
            bail!(
                "{} already exists but does not match the catalog; delete it before downloading",
                destination.display()
            );
        }
        InstallState::Missing => {}
    }

    if cancel.load(Ordering::Acquire) {
        bail!("download cancelled");
    }

    static PART_COUNTER: AtomicU64 = AtomicU64::new(0);
    let part_number = PART_COUNTER.fetch_add(1, Ordering::Relaxed);
    let part_name = format!(
        ".{}.part-{}-{part_number}",
        spec.filename,
        std::process::id()
    );
    let part_path = models_dir.join(part_name);

    let result = (|| -> Result<()> {
        let response = ureq::get(spec.url)
            .set("User-Agent", concat!("sttd/", env!("CARGO_PKG_VERSION")))
            .call()
            .with_context(|| format!("downloading {}", spec.name))?;
        if let Some(length) = response
            .header("Content-Length")
            .and_then(|value| value.parse::<u64>().ok())
        {
            if length != spec.size_bytes {
                bail!(
                    "server reported {length} bytes for {}, expected {}",
                    spec.name,
                    spec.size_bytes
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
            if downloaded > spec.size_bytes {
                bail!("download exceeded the expected size for {}", spec.name);
            }
            file.write_all(&buffer[..count])
                .context("writing partial model")?;
            hasher.update(&buffer[..count]);
            progress(downloaded, spec.size_bytes);
        }

        if downloaded != spec.size_bytes {
            bail!(
                "downloaded {downloaded} bytes for {}, expected {}",
                spec.name,
                spec.size_bytes
            );
        }
        let actual_hash = format!("{:x}", hasher.finalize());
        if actual_hash != spec.sha256 {
            bail!(
                "SHA-256 mismatch for {} (expected {}, got {actual_hash})",
                spec.name,
                spec.sha256
            );
        }

        file.flush().context("flushing partial model")?;
        file.sync_all().context("syncing partial model")?;
        drop(file);

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

fn verify_model_file(path: &Path, spec: &ModelSpec) -> Result<()> {
    let mut file =
        File::open(path).with_context(|| format!("opening installed model {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("reading installed model metadata for {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.len() != spec.size_bytes {
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
    if actual_hash != spec.sha256 {
        bail!("installed model failed SHA-256 verification");
    }
    Ok(())
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
}
