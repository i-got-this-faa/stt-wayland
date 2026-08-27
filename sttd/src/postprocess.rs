//! Post-processing phase: transcript text normalization.
//!
//! Pipeline position: `audio → ASR (stt) → post-process → committed text`.
//!
//! A post-processor rewrites raw ASR output into clean written text:
//! punctuation and capitalization, inverse text normalization (spoken
//! numbers, dates, times, currency, email addresses), filler removal and
//! self-correction resolution. Models are local GGUF files from
//! [`crate::model_catalog::POST_PROCESS_CATALOG`], downloaded and managed by
//! `stt-settings`.
//!
//! # Models
//!
//! The first shipped model is **S1-mini by Superwhisper**
//! (`superwhisper/s1-mini`), a 0.6B Qwen3 fine-tune trained for a single
//! transformation task. Its input format is part of the training and must be
//! reproduced exactly:
//!
//! ```text
//! <|im_start|>system
//! {system prompt}<|im_end|>
//! <|im_start|>user
//! [Styling: …] [Structure: …] [Context: …]
//! {raw transcript}<|im_end|>
//! <|im_start|>assistant
//! <think>
//!
//! </think>
//!
//! ```
//!
//! — the documented system prompt, the three control-line values from
//! [`PostProcessConfig`], and the empty think block that disables Qwen3's
//! thinking mode. Decoding is greedy (`do_sample = false` upstream).
//!
//! Adding future models: append a `PostProcessModelSpec` to
//! `POST_PROCESS_CATALOG`. If a model uses a different prompt format, extend
//! [`chat_prompt`] to dispatch on the spec instead of assuming the s1 shape.
//!
//! # Failure policy
//!
//! Post-processing must never break dictation. Any missing file, load or
//! inference error falls back to the raw transcript (warned once), and an
//! empty model output is a *valid* result for filler-only input, matching
//! upstream guidance.

use tracing::{debug, info, warn};

use crate::config::PostProcessConfig;

/// The exact system prompt required by S1-mini (part of the trained format).
pub const S1_SYSTEM_PROMPT: &str = "You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.";

/// Assistant-turn prefix: Qwen3's non-thinking template. Two newlines inside
/// the think block and two after it — upstream warns blank output otherwise.
const ASSISTANT_THINK_PREFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Target maximum prompt tokens per pass. The model card recommends passes
/// under ~1,000 tokens; dictation chunks stay far below, but Edit-mode
/// buffers can hold minutes of text and are split to fit.
const MAX_INPUT_TOKENS: usize = 768;

/// Hard ceiling on generated tokens per pass.
const MAX_NEW_TOKENS: usize = 1024;

/// Control line steering S1-mini, built from the configured axis values.
pub fn control_line(config: &PostProcessConfig) -> String {
    format!(
        "[Styling: {}] [Structure: {}] [Context: {}]",
        config.styling, config.structure, config.context
    )
}

/// Full chat prompt in the exact format S1-mini was trained on.
pub fn chat_prompt(config: &PostProcessConfig, transcript: &str) -> String {
    format!(
        "<|im_start|>system\n{S1_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{}\n{transcript}<|im_end|>\n{ASSISTANT_THINK_PREFIX}",
        control_line(config)
    )
}

/// Defensive output cleanup: drop a stray `<think>…</think>` block (only
/// possible when a model ignores the prefilled prefix) and trim whitespace.
pub fn clean_output(text: &str) -> String {
    let mut out = text;
    if let Some(start) = out.find("<think>") {
        if let Some(end_rel) = out[start..].find("</think>") {
            let end = start + end_rel + "</think>".len();
            out = &out[end..];
        }
    }
    out.trim().to_string()
}

/// `1.3 × input_tokens + 32`, clamped — the model card's safe ceiling.
fn max_new_tokens_for(input_tokens: usize) -> usize {
    ((input_tokens as f64 * 1.3) as usize + 32)
        .clamp(16, MAX_NEW_TOKENS)
}

/// Sentence splitter used to keep single passes within the token budget.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        current.push(character);
        if matches!(character, '.' | '!' | '?' | ';' | '\n') {
            sentences.push(current.trim().to_string());
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences.into_iter().filter(|s| !s.is_empty()).collect()
}

// ─── Worker-facing handle ───────────────────────────────────────────────────

/// Handle owned by the STT worker; applies the phase to final transcripts.
pub(crate) struct PostProcessor {
    config: PostProcessConfig,
    #[cfg(feature = "post-process")]
    engine: Option<engine::TextNormalizer>,
    warned_problem: bool,
}

impl PostProcessor {
    pub fn new(config: PostProcessConfig) -> Self {
        if config.enabled {
            if crate::model_catalog::post_process_spec_for_id(&config.model).is_none() {
                warn!(
                    model = %config.model,
                    "post-process: unknown model id; transcripts stay raw \
                     (pick a model in stt-settings)"
                );
            } else {
                info!(
                    model = %config.model,
                    styling = %config.styling,
                    structure = %config.structure,
                    context = %config.context,
                    "post-process: enabled (model loads lazily on first transcript)"
                );
            }
        }
        Self {
            config,
            #[cfg(feature = "post-process")]
            engine: None,
            warned_problem: false,
        }
    }

    /// Whether the phase is configured on (regardless of engine state).
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Apply the post-processing phase to a final transcript.
    ///
    /// Infallible: any failure returns the raw transcript so dictation keeps
    /// flowing. An empty result is valid (filler-only input) and propagates.
    pub fn process(&mut self, text: &str) -> String {
        if !self.config.enabled || text.trim().is_empty() {
            return text.to_string();
        }

        #[cfg(not(feature = "post-process"))]
        {
            self.warn_once(
                "post-process enabled but sttd was built without the \
                 `post-process` feature; rebuild with default features",
            );
            return text.to_string();
        }

        #[cfg(feature = "post-process")]
        {
            let Some(spec) = crate::model_catalog::post_process_spec_for_id(&self.config.model)
            else {
                self.warn_once(&format!(
                    "post-process model '{}' is not in the catalog; transcripts stay raw",
                    self.config.model
                ));
                return text.to_string();
            };

            if self.engine.is_none() {
                match self.load_engine(spec) {
                    Ok(engine) => self.engine = Some(engine),
                    Err(error) => {
                        self.warn_once(&format!(
                            "post-process model unavailable; transcripts stay raw ({error:#})"
                        ));
                        return text.to_string();
                    }
                }
            }

            let engine = self.engine.as_mut().expect("engine loaded above");
            match engine.normalize(&self.config, text) {
                Ok(cleaned) => {
                    debug!(
                        raw_chars = text.len(),
                        clean_chars = cleaned.len(),
                        "post-process: normalized transcript"
                    );
                    cleaned
                }
                Err(error) => {
                    warn!(error = %error, "post-process: normalization failed; using raw transcript");
                    text.to_string()
                }
            }
        }
    }

    #[cfg(feature = "post-process")]
    fn load_engine(
        &self,
        spec: &'static crate::model_catalog::PostProcessModelSpec,
    ) -> anyhow::Result<engine::TextNormalizer> {
        use crate::model_catalog::{
            post_process_install_state, post_process_model_path, post_process_tokenizer_path,
            InstallState,
        };

        let models_dir = crate::model_catalog::post_process_models_dir();
        if post_process_install_state(&models_dir, spec) != InstallState::Installed {
            anyhow::bail!(
                "model '{}' is not downloaded (install it in stt-settings → Post-Processing)",
                spec.name
            );
        }
        let gguf = post_process_model_path(&models_dir, spec)
            .ok_or_else(|| anyhow::anyhow!("invalid model filename in catalog"))?;
        let tokenizer = post_process_tokenizer_path(&models_dir, spec)
            .ok_or_else(|| anyhow::anyhow!("invalid tokenizer filename in catalog"))?;
        engine::TextNormalizer::load(&gguf, &tokenizer)
    }

    fn warn_once(&mut self, message: &str) {
        if self.warned_problem {
            debug!("{}", message);
        } else {
            self.warned_problem = true;
            warn!("{}", message);
        }
    }
}

// ─── Inference engine (candle, quantized Qwen3) ─────────────────────────────

#[cfg(feature = "post-process")]
mod engine {
    use std::path::Path;

    use anyhow::{Context, Result};
    use candle_core::quantized::gguf_file;
    use candle_core::{Device, Tensor};
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    use candle_transformers::models::quantized_qwen3::ModelWeights;
    use tracing::info;

    use super::{chat_prompt, clean_output, max_new_tokens_for, split_sentences, MAX_INPUT_TOKENS};
    use crate::config::PostProcessConfig;

    /// Greedy text generator over a quantized Qwen3 GGUF.
    pub(super) struct TextNormalizer {
        model: ModelWeights,
        tokenizer: tokenizers::Tokenizer,
        logits: LogitsProcessor,
        device: Device,
        eos_tokens: Vec<u32>,
    }

    impl TextNormalizer {
        /// Select the best compute device available at runtime.
        ///
        /// `cuda_if_available` / `metal_if_available` are always compiled into
        /// candle-core; they fall back to `Ok(Device::Cpu)` when the backend
        /// is absent, so we inspect the returned device before accepting it.
        fn best_device() -> Result<Device> {
            if let Ok(device) = Device::cuda_if_available(0) {
                if device.is_cuda() {
                    return Ok(device);
                }
            }
            if let Ok(device) = Device::metal_if_available(0) {
                if device.is_metal() {
                    return Ok(device);
                }
            }
            Ok(Device::Cpu)
        }

        pub(super) fn load(gguf: &Path, tokenizer_json: &Path) -> Result<Self> {
            let start = std::time::Instant::now();
            let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json).map_err(|error| {
                anyhow::anyhow!("loading tokenizer {}: {error}", tokenizer_json.display())
            })?;
            let device = Self::best_device()?;
            let mut file = std::fs::File::open(gguf)
                .with_context(|| format!("opening model {}", gguf.display()))?;
            let content = gguf_file::Content::read(&mut file)
                .map_err(|error| error.with_path(gguf))
                .with_context(|| format!("reading GGUF header from {}", gguf.display()))?;
            let model = ModelWeights::from_gguf(content, &mut file, &device)
                .with_context(|| format!("loading GGUF weights from {}", gguf.display()))?;
            let eos_tokens = ["<|im_end|>", "<|endoftext|>"]
                .iter()
                .filter_map(|token| tokenizer.get_vocab(true).get(*token).copied())
                .collect();
            info!(
                model = %gguf.display(),
                device = ?device,
                load_ms = start.elapsed().as_millis() as u64,
                "post-process: normalizer ready"
            );
            Ok(Self {
                model,
                tokenizer,
                logits: LogitsProcessor::from_sampling(0, Sampling::ArgMax),
                device,
                eos_tokens,
            })
        }

        /// Token count of `text` (0 when encoding fails — treated as short).
        fn token_len(&self, text: &str) -> usize {
            self.tokenizer
                .encode(text, true)
                .map(|encoding| encoding.get_ids().len())
                .unwrap_or(0)
        }

        pub(super) fn normalize(
            &mut self,
            config: &PostProcessConfig,
            transcript: &str,
        ) -> Result<String> {
            let pieces = self.split_long_transcript(transcript);
            let mut outputs = Vec::with_capacity(pieces.len());
            for piece in &pieces {
                let prompt = chat_prompt(config, piece);
                let max_new_tokens = max_new_tokens_for(self.token_len(&prompt));
                let generated = self.generate(&prompt, max_new_tokens)?;
                let cleaned = clean_output(&generated);
                // Filler-only pieces legitimately normalize to nothing.
                if !cleaned.is_empty() {
                    outputs.push(cleaned);
                }
            }
            Ok(outputs.join(" "))
        }

        /// Split transcripts exceeding the per-pass token budget at sentence
        /// boundaries (word boundaries as fallback) per the model card's
        /// "chunk at sentence boundaries" guidance.
        fn split_long_transcript(&self, transcript: &str) -> Vec<String> {
            if self.token_len(transcript) <= MAX_INPUT_TOKENS {
                return vec![transcript.to_string()];
            }

            let mut pieces: Vec<String> = Vec::new();
            let mut current = String::new();
            let mut current_tokens = 0usize;

            for sentence in split_sentences(transcript) {
                let sentence_tokens = self.token_len(&sentence);
                if sentence_tokens > MAX_INPUT_TOKENS {
                    // A single oversized sentence: hard-split at word gaps.
                    if !current.trim().is_empty() {
                        pieces.push(current.trim().to_string());
                        current.clear();
                        current_tokens = 0;
                    }
                    for word in sentence.split_whitespace() {
                        let candidate = if current.is_empty() {
                            word.to_string()
                        } else {
                            format!("{current} {word}")
                        };
                        let candidate_tokens = self.token_len(&candidate);
                        if candidate_tokens > MAX_INPUT_TOKENS && !current.is_empty() {
                            pieces.push(current.trim().to_string());
                            current = word.to_string();
                            current_tokens = self.token_len(&current);
                        } else {
                            current = candidate;
                            current_tokens = candidate_tokens;
                        }
                    }
                    continue;
                }
                if current_tokens + sentence_tokens > MAX_INPUT_TOKENS
                    && !current.trim().is_empty()
                {
                    pieces.push(current.trim().to_string());
                    current.clear();
                    current_tokens = 0;
                }
                if current.is_empty() {
                    current = sentence;
                } else {
                    current.push(' ');
                    current.push_str(&sentence);
                }
                current_tokens += sentence_tokens;
            }
            if !current.trim().is_empty() {
                pieces.push(current.trim().to_string());
            }
            if pieces.is_empty() {
                pieces.push(transcript.to_string());
            }
            pieces
        }

        /// Greedy generation; returns the decoded completion (may be empty).
        fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String> {
            let encoding = self
                .tokenizer
                .encode(prompt, true)
                .map_err(|error| anyhow::anyhow!("encoding prompt: {error}"))?;
            let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
            if prompt_tokens.is_empty() {
                return Ok(String::new());
            }

            let mut next = {
                let input = Tensor::new(&prompt_tokens[..], &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, 0)?.squeeze(0)?;
                self.logits.sample(&logits)?
            };

            let mut generated: Vec<u32> = vec![next];
            if !self.eos_tokens.contains(&next) {
                for index in 0..max_new_tokens.saturating_sub(1) {
                    let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                    let logits = self
                        .model
                        .forward(&input, prompt_tokens.len() + index)?
                        .squeeze(0)?;
                    next = self.logits.sample(&logits)?;
                    if self.eos_tokens.contains(&next) {
                        break;
                    }
                    generated.push(next);
                }
            }

            self.tokenizer
                .decode(&generated, true)
                .map_err(|error| anyhow::anyhow!("decoding output: {error}"))
        }
    }
}

#[cfg(test)]
mod tests;
