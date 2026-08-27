//! Tests for the post-processing phase (prompt format + graceful fallback).

use std::path::Path;

use super::*;
use crate::config::PostProcessConfig;

fn default_config() -> PostProcessConfig {
    PostProcessConfig::default()
}

#[test]
fn control_line_matches_trained_format() {
    assert_eq!(
        control_line(&default_config()),
        "[Styling: semi-formal] [Structure: prose] [Context: general]"
    );
}

#[test]
fn chat_prompt_reproduces_the_s1_template_exactly() {
    let config = PostProcessConfig {
        styling: "casual".into(),
        structure: "lists".into(),
        context: "email".into(),
        ..PostProcessConfig::default()
    };
    let prompt = chat_prompt(&config, "hello world");
    assert_eq!(
        prompt,
        "<|im_start|>system\n\
         You are a text normalizer for speech-to-text transcripts. The input begins \
         with a control line specifying the styling, structure, and context settings; \
         clean the transcript to match those settings and output only the cleaned \
         text.<|im_end|>\n\
         <|im_start|>user\n\
         [Styling: casual] [Structure: lists] [Context: email]\n\
         hello world<|im_end|>\n\
         <|im_start|>assistant\n\
         <think>\n\n</think>\n\n"
    );
    // The assistant turn must open with the empty think block.
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

#[test]
fn clean_output_strips_stray_think_blocks_and_whitespace() {
    assert_eq!(clean_output("  Hello there.  "), "Hello there.");
    assert_eq!(
        clean_output("<think>\nreasoning\n</think>\n\nHello."),
        "Hello."
    );
    // Empty output is a valid normalization (filler-only input).
    assert_eq!(clean_output("   "), "");
}

#[test]
fn max_new_tokens_tracks_input_size() {
    assert_eq!(max_new_tokens_for(0), 32);
    assert_eq!(max_new_tokens_for(100), 162);
    // Clamped ceiling.
    assert_eq!(max_new_tokens_for(100_000), MAX_NEW_TOKENS);
}

#[test]
fn sentences_split_on_terminators() {
    let sentences = split_sentences("One. Two! Three? trailing");
    assert_eq!(sentences, vec!["One.", "Two!", "Three?", "trailing"]);
}

#[test]
fn processor_passes_through_when_disabled() {
    let mut processor = PostProcessor::new(default_config());
    assert!(!processor.is_enabled());
    assert_eq!(processor.process("raw text"), "raw text");
}

#[test]
fn processor_passes_through_empty_input_when_enabled() {
    let config = PostProcessConfig {
        enabled: true,
        ..PostProcessConfig::default()
    };
    let mut processor = PostProcessor::new(config);
    assert_eq!(processor.process("   "), "   ");
}

#[cfg(feature = "post-process")]
#[test]
fn processor_falls_back_to_raw_when_model_missing() {
    // Enabled but nothing downloaded: must return the raw transcript and
    // never panic — dictation keeps flowing.
    let config = PostProcessConfig {
        enabled: true,
        model: "s1-mini".into(),
        ..PostProcessConfig::default()
    };
    let mut processor = PostProcessor::new(config);
    assert_eq!(processor.process("raw text"), "raw text");
}

#[cfg(feature = "post-process")]
#[test]
fn processor_falls_back_for_unknown_model_id() {
    let config = PostProcessConfig {
        enabled: true,
        model: "does-not-exist".into(),
        ..PostProcessConfig::default()
    };
    let mut processor = PostProcessor::new(config);
    assert_eq!(processor.process("raw text"), "raw text");
}

// End-to-end normalization with the real model files; skipped unless the
// environment points at them (run locally, not in CI):
//   STTD_PP_TEST_GGUF=…/s1-mini-q4_k_m.gguf \
//   STTD_PP_TEST_TOKENIZER=…/tokenizer.json \
//   cargo test -p sttd post_process_live -- --nocapture
#[cfg(feature = "post-process")]
#[test]
fn post_process_live_normalization() {
    let (Ok(gguf), Ok(tokenizer)) = (
        std::env::var("STTD_PP_TEST_GGUF"),
        std::env::var("STTD_PP_TEST_TOKENIZER"),
    ) else {
        eprintln!("skipping: STTD_PP_TEST_GGUF / STTD_PP_TEST_TOKENIZER not set");
        return;
    };
    let mut normalizer =
        engine::TextNormalizer::load(Path::new(&gguf), Path::new(&tokenizer)).unwrap();

    let config = default_config();
    let out = normalizer
        .normalize(
            &config,
            "so um i need to like send the the report by uh friday no wait make that thursday",
        )
        .unwrap();
    println!("normalized: {out:?}");
    assert_eq!(out, "I need to send the report by Thursday.");

    // Filler-only input normalizes to the empty string.
    let out = normalizer
        .normalize(&config, "um uh um")
        .unwrap();
    println!("filler-only normalized: {out:?}");
    assert_eq!(out, "");
}
