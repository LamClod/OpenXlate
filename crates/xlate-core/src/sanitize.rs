use crate::types::{Message, ToolCallDescriptor};

/// Cleans a message list before sending to any provider:
/// 1. Remove empty assistant placeholder messages
/// 2. Merge adjacent assistant messages that only contain tool_calls
/// 3. Trim dangling tool_calls that have no corresponding tool response
/// 4. Trim trailing assistant prefill messages
///
/// Ported from the original Go implementation's sanitizeProviderMessages (router.go:140-161).
pub fn sanitize_messages(input: Vec<Message>) -> Vec<Message> {
    sanitize_messages_with(input, true, true, true, true)
}

pub fn sanitize_messages_with(
    input: Vec<Message>,
    remove_empty_assistants: bool,
    merge_adjacent_tool_calls: bool,
    trim_dangling: bool,
    trim_trailing_prefill: bool,
) -> Vec<Message> {
    if input.is_empty() {
        return Vec::new();
    }

    let filtered: Vec<Message> = if remove_empty_assistants {
        input
            .into_iter()
            .filter(|m| !is_assistant_placeholder(m))
            .collect()
    } else {
        input
    };

    let merged = if merge_adjacent_tool_calls {
        merge_adjacent_assistant_tool_call_messages(filtered)
    } else {
        filtered
    };

    let mut trimmed = if trim_dangling {
        trim_dangling_assistant_tool_calls(merged)
    } else {
        merged
    };

    if trim_trailing_prefill {
        while trimmed.last().is_some_and(is_assistant_prefill) {
            trimmed.pop();
        }
    }

    trimmed
}

fn is_assistant_placeholder(m: &Message) -> bool {
    if m.role.trim() != "assistant" {
        return false;
    }
    if !m.tool_calls.is_empty() || !m.content_parts.is_empty() {
        return false;
    }
    if !m.tool_call_id.trim().is_empty() || !m.name.trim().is_empty() {
        return false;
    }
    if !m.reasoning_content.trim().is_empty() {
        return false;
    }
    if !m.reasoning_signature.trim().is_empty() {
        return false;
    }
    m.content.trim().is_empty()
}

fn is_assistant_prefill(m: &Message) -> bool {
    if m.role.trim() != "assistant" {
        return false;
    }
    if !m.tool_calls.is_empty() {
        return false;
    }
    if !m.tool_call_id.trim().is_empty() || !m.name.trim().is_empty() {
        return false;
    }
    !m.content.trim().is_empty() || !m.reasoning_content.trim().is_empty()
}

fn merge_adjacent_assistant_tool_call_messages(input: Vec<Message>) -> Vec<Message> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut merged: Vec<Message> = Vec::with_capacity(input.len());
    for msg in input {
        if try_merge_into_last(&mut merged, &msg) {
            continue;
        }
        merged.push(msg);
    }
    merged
}

fn can_merge(last: &Message, current: &Message) -> bool {
    if last.role.trim() != "assistant" || current.role.trim() != "assistant" {
        return false;
    }
    if last.tool_calls.is_empty() || current.tool_calls.is_empty() {
        return false;
    }
    if !last.tool_call_id.trim().is_empty() || !last.name.trim().is_empty() {
        return false;
    }
    if !current.tool_call_id.trim().is_empty() || !current.name.trim().is_empty() {
        return false;
    }
    if !current.content.trim().is_empty() || !current.content_parts.is_empty() {
        return false;
    }
    true
}

fn try_merge_into_last(merged: &mut [Message], current: &Message) -> bool {
    let Some(last) = merged.last_mut() else {
        return false;
    };
    if !can_merge(last, current) {
        return false;
    }
    let start_index = last.tool_calls.len();
    for (i, tc) in current.tool_calls.iter().enumerate() {
        let mut item = tc.clone();
        item.index = start_index + i;
        last.tool_calls.push(item);
    }
    last.reasoning_content = merge_reasoning(&last.reasoning_content, &current.reasoning_content);
    merge_reasoning_metadata(last, current);
    true
}

fn merge_reasoning(left: &str, right: &str) -> String {
    let l = left.trim();
    let r = right.trim();
    match (l.is_empty(), r.is_empty(), l == r) {
        (true, _, _) => r.to_string(),
        (_, true, _) | (_, _, true) => l.to_string(),
        _ => format!("{l}\n\n{r}"),
    }
}

fn merge_reasoning_signature(left: &str, right: &str) -> String {
    let l = left.trim();
    let r = right.trim();
    match (l.is_empty(), r.is_empty(), l == r) {
        (true, _, _) => r.to_string(),
        (_, true, _) | (_, _, true) => l.to_string(),
        _ => String::new(),
    }
}

fn merge_reasoning_metadata(last: &mut Message, current: &Message) {
    let left_sig = last.reasoning_signature.trim().to_string();
    let right_sig = current.reasoning_signature.trim().to_string();
    let merged_sig = merge_reasoning_signature(&left_sig, &right_sig);
    last.reasoning_signature = merged_sig.clone();

    if merged_sig.is_empty() {
        last.reasoning_signature_source.clear();
        last.openai_responses_reasoning_id.clear();
        last.openai_responses_reasoning_status.clear();
        last.openai_responses_reasoning_summary = None;
        return;
    }

    if left_sig.is_empty() && !right_sig.is_empty() {
        last.reasoning_signature_source = current.reasoning_signature_source.trim().to_string();
        last.openai_responses_reasoning_id = current.openai_responses_reasoning_id.clone();
        last.openai_responses_reasoning_status = current.openai_responses_reasoning_status.clone();
        last.openai_responses_reasoning_summary =
            current.openai_responses_reasoning_summary.clone();
        return;
    }

    if left_sig == right_sig {
        if last.reasoning_signature_source.trim().is_empty() {
            last.reasoning_signature_source =
                current.reasoning_signature_source.trim().to_string();
        }
        if last.openai_responses_reasoning_id.trim().is_empty() {
            last.openai_responses_reasoning_id = current.openai_responses_reasoning_id.clone();
        }
        if last.openai_responses_reasoning_status.trim().is_empty() {
            last.openai_responses_reasoning_status =
                current.openai_responses_reasoning_status.clone();
        }
        if last.openai_responses_reasoning_summary.is_none() {
            last.openai_responses_reasoning_summary =
                current.openai_responses_reasoning_summary.clone();
        }
    }
}

fn trim_dangling_assistant_tool_calls(input: Vec<Message>) -> Vec<Message> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<Message> = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let msg = &input[i];
        if msg.role.trim() != "assistant" || msg.tool_calls.is_empty() {
            result.push(msg.clone());
            i += 1;
            continue;
        }

        let mut end = i + 1;
        let mut responded = std::collections::HashSet::new();
        while end < input.len() && input[end].role.trim() == "tool" {
            let tc_id = input[end].tool_call_id.trim();
            if !tc_id.is_empty() {
                responded.insert(tc_id.to_string());
            }
            end += 1;
        }

        let mut kept_calls: Vec<ToolCallDescriptor> = Vec::new();
        let mut allowed_ids = std::collections::HashSet::new();
        for tc in &msg.tool_calls {
            let tc_id = tc.id.trim();
            if !responded.contains(tc_id) {
                continue;
            }
            let mut item = tc.clone();
            item.index = kept_calls.len();
            kept_calls.push(item);
            allowed_ids.insert(tc_id.to_string());
        }

        if !kept_calls.is_empty() {
            let mut cleaned = msg.clone();
            cleaned.tool_calls = kept_calls;
            result.push(cleaned);
            for tool_msg in &input[(i + 1)..end] {
                if allowed_ids.contains(tool_msg.tool_call_id.trim()) {
                    result.push(tool_msg.clone());
                }
            }
        } else if !msg.content.trim().is_empty()
            || !msg.content_parts.is_empty()
            || !msg.reasoning_content.trim().is_empty()
        {
            let mut fallback = msg.clone();
            fallback.tool_calls = Vec::new();
            result.push(fallback);
        }

        i = end;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolCallFunctionShape;

    fn user_msg(text: &str) -> Message {
        Message {
            role: "user".into(),
            content: text.into(),
            ..Default::default()
        }
    }

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: "assistant".into(),
            content: text.into(),
            ..Default::default()
        }
    }

    fn assistant_with_tool_calls(ids: &[&str]) -> Message {
        Message {
            role: "assistant".into(),
            tool_calls: ids
                .iter()
                .enumerate()
                .map(|(i, id)| ToolCallDescriptor {
                    id: id.to_string(),
                    index: i,
                    kind: "function".into(),
                    function: ToolCallFunctionShape {
                        name: "test".into(),
                        arguments: "{}".into(),
                    },
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn tool_result(tc_id: &str) -> Message {
        Message {
            role: "tool".into(),
            tool_call_id: tc_id.into(),
            content: "result".into(),
            ..Default::default()
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(sanitize_messages(vec![]).is_empty());
    }

    #[test]
    fn preserves_normal_conversation() {
        let input = vec![user_msg("hi"), assistant_msg("hello"), user_msg("bye")];
        let output = sanitize_messages(input.clone());
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn removes_empty_assistant_placeholder() {
        let input = vec![
            user_msg("hi"),
            Message {
                role: "assistant".into(),
                ..Default::default()
            },
            user_msg("bye"),
        ];
        let output = sanitize_messages(input);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].content, "hi");
        assert_eq!(output[1].content, "bye");
    }

    #[test]
    fn keeps_assistant_with_reasoning_when_not_trailing() {
        let input = vec![
            user_msg("hi"),
            Message {
                role: "assistant".into(),
                reasoning_content: "thinking...".into(),
                ..Default::default()
            },
            user_msg("follow up"),
        ];
        let output = sanitize_messages(input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn trims_trailing_assistant_with_only_reasoning() {
        let input = vec![
            user_msg("hi"),
            Message {
                role: "assistant".into(),
                reasoning_content: "thinking...".into(),
                ..Default::default()
            },
        ];
        let output = sanitize_messages(input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn merges_adjacent_assistant_tool_calls() {
        let input = vec![
            user_msg("hi"),
            assistant_with_tool_calls(&["tc1"]),
            assistant_with_tool_calls(&["tc2"]),
            tool_result("tc1"),
            tool_result("tc2"),
        ];
        let output = sanitize_messages(input);
        assert_eq!(output.len(), 4);
        assert_eq!(output[1].tool_calls.len(), 2);
        assert_eq!(output[1].tool_calls[0].id, "tc1");
        assert_eq!(output[1].tool_calls[0].index, 0);
        assert_eq!(output[1].tool_calls[1].id, "tc2");
        assert_eq!(output[1].tool_calls[1].index, 1);
    }

    #[test]
    fn trims_dangling_tool_calls() {
        let input = vec![
            user_msg("hi"),
            assistant_with_tool_calls(&["tc1", "tc2"]),
            tool_result("tc1"),
        ];
        let output = sanitize_messages(input);
        assert_eq!(output[1].tool_calls.len(), 1);
        assert_eq!(output[1].tool_calls[0].id, "tc1");
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn all_dangling_falls_back_to_text_if_present() {
        let mut msg = assistant_with_tool_calls(&["tc1"]);
        msg.content = "some text".into();
        let input = vec![user_msg("hi"), msg, user_msg("follow up")];
        let output = sanitize_messages(input);
        assert_eq!(output.len(), 3);
        assert!(output[1].tool_calls.is_empty());
        assert_eq!(output[1].content, "some text");
    }

    #[test]
    fn all_dangling_no_text_drops_message() {
        let input = vec![user_msg("hi"), assistant_with_tool_calls(&["tc1"])];
        let output = sanitize_messages(input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn trims_trailing_assistant_prefill() {
        let input = vec![
            user_msg("hi"),
            assistant_msg("hello"),
            user_msg("bye"),
            assistant_msg("trailing prefill"),
        ];
        let output = sanitize_messages(input);
        assert_eq!(output.len(), 3);
        assert_eq!(output[2].content, "bye");
    }

    #[test]
    fn does_not_trim_trailing_assistant_with_tool_calls() {
        let input = vec![
            user_msg("hi"),
            assistant_with_tool_calls(&["tc1"]),
            tool_result("tc1"),
        ];
        let output = sanitize_messages(input);
        assert_eq!(output.len(), 3);
    }
}
