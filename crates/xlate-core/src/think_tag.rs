//! Incremental `<think>...</think>` tag parser for provider streams.

const THINK_OPEN_TAG: &str = "<think>";
const THINK_CLOSE_TAG: &str = "</think>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentPartKind {
    Text,
    Reasoning,
    ThinkingCompleted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentPart {
    pub kind: ContentPartKind,
    pub text: String,
}

pub struct ThinkTagParser {
    carry: String,
    in_think: bool,
}

impl Default for ThinkTagParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkTagParser {
    pub fn new() -> Self {
        Self {
            carry: String::new(),
            in_think: false,
        }
    }

    pub fn consume(&mut self, text: &str) -> Vec<ContentPart> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut input = format!("{}{}", self.carry, text);
        self.carry.clear();
        let mut parts = Vec::with_capacity(4);

        while !input.is_empty() {
            if self.in_think {
                if let Some(close_idx) = input.find(THINK_CLOSE_TAG) {
                    if close_idx > 0 {
                        parts.push(ContentPart {
                            kind: ContentPartKind::Reasoning,
                            text: input[..close_idx].to_string(),
                        });
                    }
                    parts.push(ContentPart {
                        kind: ContentPartKind::ThinkingCompleted,
                        text: String::new(),
                    });
                    self.in_think = false;
                    input = input[close_idx + THINK_CLOSE_TAG.len()..].to_string();
                    continue;
                }

                let carry_len = trailing_tag_prefix_length(&input, THINK_CLOSE_TAG);
                let emit_end = input.len() - carry_len;
                if emit_end > 0 {
                    parts.push(ContentPart {
                        kind: ContentPartKind::Reasoning,
                        text: input[..emit_end].to_string(),
                    });
                }
                self.carry = input[emit_end..].to_string();
                break;
            }

            if let Some(open_idx) = input.find(THINK_OPEN_TAG) {
                if open_idx > 0 {
                    parts.push(ContentPart {
                        kind: ContentPartKind::Text,
                        text: input[..open_idx].to_string(),
                    });
                }
                self.in_think = true;
                input = input[open_idx + THINK_OPEN_TAG.len()..].to_string();
                continue;
            }

            let carry_len = trailing_tag_prefix_length(&input, THINK_OPEN_TAG);
            let emit_end = input.len() - carry_len;
            if emit_end > 0 {
                parts.push(ContentPart {
                    kind: ContentPartKind::Text,
                    text: input[..emit_end].to_string(),
                });
            }
            self.carry = input[emit_end..].to_string();
            break;
        }

        parts
    }

    pub fn flush(&mut self) -> Vec<ContentPart> {
        if self.carry.is_empty() {
            return Vec::new();
        }

        let kind = if self.in_think {
            ContentPartKind::Reasoning
        } else {
            ContentPartKind::Text
        };

        let text = std::mem::take(&mut self.carry);
        vec![ContentPart { kind, text }]
    }
}

fn trailing_tag_prefix_length(text: &str, tag: &str) -> usize {
    let max_len = text.len().min(tag.len() - 1);
    for size in (1..=max_len).rev() {
        if text.ends_with(&tag[..size]) {
            return size;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tags_passes_through() {
        let mut parser = ThinkTagParser::new();
        let parts = parser.consume("hello world");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].kind, ContentPartKind::Text);
        assert_eq!(parts[0].text, "hello world");
    }

    #[test]
    fn complete_think_tag() {
        let mut parser = ThinkTagParser::new();
        let parts = parser.consume("before<think>reasoning</think>after");
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0].kind, ContentPartKind::Text);
        assert_eq!(parts[0].text, "before");
        assert_eq!(parts[1].kind, ContentPartKind::Reasoning);
        assert_eq!(parts[1].text, "reasoning");
        assert_eq!(parts[2].kind, ContentPartKind::ThinkingCompleted);
        assert_eq!(parts[3].kind, ContentPartKind::Text);
        assert_eq!(parts[3].text, "after");
    }

    #[test]
    fn split_across_chunks() {
        let mut parser = ThinkTagParser::new();
        let p1 = parser.consume("<think>rea");
        assert_eq!(p1.len(), 1);
        assert_eq!(p1[0].kind, ContentPartKind::Reasoning);
        assert_eq!(p1[0].text, "rea");

        let p2 = parser.consume("soning</think>done");
        assert_eq!(p2.len(), 3);
        assert_eq!(p2[0].kind, ContentPartKind::Reasoning);
        assert_eq!(p2[0].text, "soning");
        assert_eq!(p2[1].kind, ContentPartKind::ThinkingCompleted);
        assert_eq!(p2[2].kind, ContentPartKind::Text);
        assert_eq!(p2[2].text, "done");
    }

    #[test]
    fn tag_split_at_boundary() {
        let mut parser = ThinkTagParser::new();
        let p1 = parser.consume("hello<thin");
        assert_eq!(p1.len(), 1);
        assert_eq!(p1[0].text, "hello");

        let p2 = parser.consume("k>inside</think>");
        assert_eq!(p2.len(), 2);
        assert_eq!(p2[0].kind, ContentPartKind::Reasoning);
        assert_eq!(p2[0].text, "inside");
        assert_eq!(p2[1].kind, ContentPartKind::ThinkingCompleted);
    }

    #[test]
    fn flush_remaining_carry() {
        let mut parser = ThinkTagParser::new();
        let parts = parser.consume("<think>partial");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].kind, ContentPartKind::Reasoning);
        assert_eq!(parts[0].text, "partial");
        let flushed = parser.flush();
        assert!(flushed.is_empty());
    }

    #[test]
    fn flush_carry_inside_think() {
        let mut parser = ThinkTagParser::new();
        let parts = parser.consume("<think>reasoning</thin");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].kind, ContentPartKind::Reasoning);
        assert_eq!(parts[0].text, "reasoning");
        let flushed = parser.flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].kind, ContentPartKind::Reasoning);
        assert_eq!(flushed[0].text, "</thin");
    }

    #[test]
    fn flush_carry_outside_think() {
        let mut parser = ThinkTagParser::new();
        let p1 = parser.consume("hello<th");
        assert_eq!(p1.len(), 1);
        assert_eq!(p1[0].text, "hello");

        let flushed = parser.flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].kind, ContentPartKind::Text);
        assert_eq!(flushed[0].text, "<th");
    }

    #[test]
    fn trailing_tag_prefix_length_cases() {
        assert_eq!(trailing_tag_prefix_length("abc<thi", "<think>"), 4);
        assert_eq!(trailing_tag_prefix_length("abc<", "<think>"), 1);
        assert_eq!(trailing_tag_prefix_length("abcdef", "<think>"), 0);
        assert_eq!(trailing_tag_prefix_length("</thin", "</think>"), 6);
    }

    #[test]
    fn empty_input_returns_empty() {
        let mut parser = ThinkTagParser::new();
        assert!(parser.consume("").is_empty());
    }
}
