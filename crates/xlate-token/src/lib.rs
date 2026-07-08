use std::collections::HashMap;

use xlate_core::types::{ContentPartType, Message};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingType {
    O200kBase,
    Cl100kBase,
    ClaudeTokens,
    Fallback,
}

pub struct TokenEstimator {
    encodings: HashMap<String, EncodingType>,
}

impl TokenEstimator {
    pub fn new() -> Self {
        Self {
            encodings: HashMap::new(),
        }
    }

    pub fn register_encoding(&mut self, model_prefix: &str, encoding: EncodingType) {
        self.encodings
            .insert(model_prefix.to_string(), encoding);
    }

    pub fn resolve_encoding(&self, model: &str) -> EncodingType {
        for (prefix, enc) in &self.encodings {
            if model.starts_with(prefix) {
                return *enc;
            }
        }
        if model.starts_with("gpt-4o") || model.starts_with("gpt-4.1") || model.starts_with("gpt-5") || model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4") {
            return EncodingType::O200kBase;
        }
        if model.starts_with("gpt-3") || model.starts_with("gpt-4") {
            return EncodingType::Cl100kBase;
        }
        if model.starts_with("claude") {
            return EncodingType::ClaudeTokens;
        }
        EncodingType::Fallback
    }

    pub fn estimate_input(&self, messages: &[Message], model: &str) -> i64 {
        let encoding = self.resolve_encoding(model);
        let mut total: i64 = 0;
        for msg in messages {
            total += 3;
            total += self.count_text_tokens(&msg.content, encoding);
            total += self.count_text_tokens(&msg.reasoning_content, encoding);
            for part in &msg.content_parts {
                total += 1;
                match part.kind {
                    ContentPartType::Text => {
                        total += self.count_text_tokens(&part.text, encoding);
                    }
                    ContentPartType::Image => {
                        total += 1024;
                    }
                }
            }
            for tc in &msg.tool_calls {
                total += self.count_text_tokens(&tc.function.name, encoding);
                total += self.count_text_tokens(&tc.function.arguments, encoding);
            }
        }
        total + 3
    }

    pub fn estimate_from_char_count(&self, char_count: usize) -> i64 {
        ((char_count as f64) / 4.0).ceil() as i64
    }

    fn count_text_tokens(&self, text: &str, _encoding: EncodingType) -> i64 {
        self.heuristic_count(text)
    }

    fn heuristic_count(&self, text: &str) -> i64 {
        if text.is_empty() {
            return 0;
        }
        let mut latin_chars = 0i64;
        let mut cjk_chars = 0i64;
        for c in text.chars() {
            if is_cjk(c) {
                cjk_chars += 1;
            } else {
                latin_chars += 1;
            }
        }
        let latin_tokens = ((latin_chars as f64) / 4.0).ceil() as i64;
        let cjk_tokens = ((cjk_chars as f64) / 1.5).ceil() as i64;
        latin_tokens + cjk_tokens
    }
}

impl Default for TokenEstimator {
    fn default() -> Self {
        Self::new()
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{3040}'..='\u{309F}'
        | '\u{30A0}'..='\u{30FF}'
        | '\u{AC00}'..='\u{D7AF}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_estimation() {
        let est = TokenEstimator::new();
        assert!(est.heuristic_count("hello world") >= 2);
        assert!(est.heuristic_count("hello world") <= 4);
    }

    #[test]
    fn cjk_estimation() {
        let est = TokenEstimator::new();
        assert!(est.heuristic_count("你好世界测试") >= 3);
        assert!(est.heuristic_count("你好世界测试") <= 5);
    }

    #[test]
    fn empty_string() {
        let est = TokenEstimator::new();
        assert_eq!(est.heuristic_count(""), 0);
    }

    #[test]
    fn resolve_encoding_gpt4o() {
        let est = TokenEstimator::new();
        assert_eq!(est.resolve_encoding("gpt-4o"), EncodingType::O200kBase);
        assert_eq!(est.resolve_encoding("gpt-4o-mini"), EncodingType::O200kBase);
    }

    #[test]
    fn resolve_encoding_claude() {
        let est = TokenEstimator::new();
        assert_eq!(est.resolve_encoding("claude-sonnet-4-20250514"), EncodingType::ClaudeTokens);
    }

    #[test]
    fn resolve_encoding_fallback() {
        let est = TokenEstimator::new();
        assert_eq!(est.resolve_encoding("unknown-model"), EncodingType::Fallback);
    }

    #[test]
    fn custom_encoding_override() {
        let mut est = TokenEstimator::new();
        est.register_encoding("my-model", EncodingType::O200kBase);
        assert_eq!(est.resolve_encoding("my-model-v2"), EncodingType::O200kBase);
    }
}
