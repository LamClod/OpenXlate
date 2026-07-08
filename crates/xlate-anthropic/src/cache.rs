//! Prompt cache breakpoint algorithm.
//!
//! Places `cache_control: { type: "ephemeral" }` markers on:
//! 1. The last tool definition
//! 2. The last system block
//! 3. The last cacheable block in the stable message prefix
//! 4. The last cacheable block in the full message list

use crate::types::ephemeral_cache_control;
use serde_json::Value;

pub(crate) fn apply_cache_breakpoints_with_limit(
    body: &mut Value,
    stable_message_count: usize,
    max_breakpoints: usize,
) {
    let mut positions = compute_breakpoint_positions(body, stable_message_count);
    positions.truncate(max_breakpoints);
    for pos in &positions {
        apply_single_breakpoint(body, pos);
    }
    tracing::debug!(positions = ?positions, max = max_breakpoints, "applied anthropic cache breakpoints");
}

/// Compute the list of JSON-path-like breakpoint position strings.
fn compute_breakpoint_positions(body: &Value, stable_message_count: usize) -> Vec<String> {
    let mut positions: Vec<String> = Vec::with_capacity(4);

    // 1. Last tool
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        if !tools.is_empty() {
            positions.push(format!("tools[{}]", tools.len() - 1));
        }
    }

    // 2. Last system block
    if let Some(system) = body.get("system").and_then(|v| v.as_array()) {
        if !system.is_empty() {
            positions.push(format!("system[{}]", system.len() - 1));
        }
    }

    // 3. Stable message prefix
    if stable_message_count > 0 {
        if let Some(path) = message_cache_breakpoint_path(body, stable_message_count) {
            positions.push(path);
        }
    }

    // 4. Full message list
    let total = body
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    if total > 0 {
        if let Some(path) = message_cache_breakpoint_path(body, total) {
            positions.push(path);
        }
    }

    dedup_positions(positions)
}

/// Find the last cacheable block within the first `message_count` messages,
/// scanning backwards.
fn message_cache_breakpoint_path(body: &Value, message_count: usize) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    if messages.is_empty() || message_count == 0 {
        return None;
    }
    let count = message_count.min(messages.len());
    for msg_idx in (0..count).rev() {
        let content = messages[msg_idx].get("content")?.as_array()?;
        for blk_idx in (0..content.len()).rev() {
            if is_cacheable_block(&content[blk_idx]) {
                return Some(format!("messages[{msg_idx}].content[{blk_idx}]"));
            }
        }
    }
    None
}

fn is_cacheable_block(block: &Value) -> bool {
    let block_type = block
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    match block_type {
        "text" => {
            let text = block
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            !text.is_empty()
        }
        "tool_result" => {
            let content = block
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            !content.is_empty()
        }
        "tool_use" => {
            let id = block
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            let name = block
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            !id.is_empty() && !name.is_empty()
        }
        _ => false,
    }
}

fn dedup_positions(items: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let trimmed = item.trim().to_string();
        if trimmed.is_empty() || seen.contains(&trimmed) {
            continue;
        }
        seen.insert(trimmed.clone());
        out.push(trimmed);
    }
    out
}

/// Apply `cache_control: { type: "ephemeral" }` at the given position.
fn apply_single_breakpoint(body: &mut Value, position: &str) {
    let position = position.trim();

    // tools[N]
    if let Some(idx) = parse_bracket_index(position, "tools") {
        if let Some(tools) = body.get_mut("tools").and_then(|v| v.as_array_mut()) {
            if idx < tools.len() {
                if let Some(tool) = tools[idx].as_object_mut() {
                    tool.insert(
                        "cache_control".to_string(),
                        serde_json::to_value(ephemeral_cache_control()).unwrap(),
                    );
                }
            }
        }
        return;
    }

    // system[N]
    if let Some(idx) = parse_bracket_index(position, "system") {
        if let Some(system) = body.get_mut("system").and_then(|v| v.as_array_mut()) {
            if idx < system.len() {
                if let Some(block) = system[idx].as_object_mut() {
                    block.insert(
                        "cache_control".to_string(),
                        serde_json::to_value(ephemeral_cache_control()).unwrap(),
                    );
                }
            }
        }
        return;
    }

    // messages[M].content[B]
    if let Some((msg_idx, blk_idx)) = parse_message_block_path(position) {
        if let Some(messages) = body.get_mut("messages").and_then(|v| v.as_array_mut()) {
            if msg_idx < messages.len() {
                if let Some(content) = messages[msg_idx]
                    .get_mut("content")
                    .and_then(|v| v.as_array_mut())
                {
                    if blk_idx < content.len() {
                        if let Some(block) = content[blk_idx].as_object_mut() {
                            block.insert(
                                "cache_control".to_string(),
                                serde_json::to_value(ephemeral_cache_control()).unwrap(),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Parse `prefix[N]` and return N.
fn parse_bracket_index(position: &str, prefix: &str) -> Option<usize> {
    let start = format!("{prefix}[");
    if !position.starts_with(&start) || !position.ends_with(']') {
        return None;
    }
    let inner = &position[start.len()..position.len() - 1];
    inner.parse().ok()
}

/// Parse `messages[M].content[B]` and return (M, B).
fn parse_message_block_path(position: &str) -> Option<(usize, usize)> {
    let rest = position.strip_prefix("messages[")?;
    let (msg_str, rest) = rest.split_once("].content[")?;
    let blk_str = rest.strip_suffix(']')?;
    let msg_idx: usize = msg_str.parse().ok()?;
    let blk_idx: usize = blk_str.parse().ok()?;
    Some((msg_idx, blk_idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bracket_index() {
        assert_eq!(parse_bracket_index("tools[3]", "tools"), Some(3));
        assert_eq!(parse_bracket_index("system[0]", "system"), Some(0));
        assert_eq!(parse_bracket_index("tools[3]", "system"), None);
    }

    #[test]
    fn test_parse_message_block_path() {
        assert_eq!(
            parse_message_block_path("messages[2].content[1]"),
            Some((2, 1))
        );
        assert_eq!(parse_message_block_path("tools[0]"), None);
    }

    #[test]
    fn test_apply_cache_breakpoints_tools() {
        let mut body = serde_json::json!({
            "tools": [
                {"name": "a", "input_schema": {}},
                {"name": "b", "input_schema": {}},
            ],
            "system": [{"type": "text", "text": "hello"}],
            "messages": [],
        });
        apply_cache_breakpoints_with_limit(&mut body, 0, 4);
        // Last tool should have cache_control
        let last_tool = &body["tools"][1];
        assert!(last_tool.get("cache_control").is_some());
        // Last system should have cache_control
        let last_system = &body["system"][0];
        assert!(last_system.get("cache_control").is_some());
    }
}
