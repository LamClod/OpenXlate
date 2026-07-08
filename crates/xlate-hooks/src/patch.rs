use async_trait::async_trait;
use xlate_core::hook::{Hook, HookContext, HookVerdict};

#[derive(Debug, Clone)]
pub struct PatchOp {
    pub op: String,
    pub path: String,
    pub value: Option<serde_json::Value>,
    pub condition: Option<String>,
}

pub struct PatchRule {
    pub model: String,
    pub patches: Vec<PatchOp>,
}

pub struct PatchHook {
    rules: Vec<PatchRule>,
}

impl PatchHook {
    pub fn new() -> Self {
        Self { rules: vec![] }
    }

    pub fn with_rules(rules: Vec<PatchRule>) -> Self {
        Self { rules }
    }
}

impl Default for PatchHook {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_segments(path: &str) -> Option<Vec<&str>> {
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if segments.is_empty() || (segments.len() == 1 && segments[0].is_empty()) {
        return None;
    }
    Some(segments)
}

fn navigate_to_parent<'a>(
    doc: &'a mut serde_json::Value,
    segments: &[&str],
) -> Option<&'a mut serde_json::Value> {
    let mut current = doc;
    for seg in segments {
        let next = if current.is_array() {
            let idx: usize = seg.parse().ok()?;
            current.as_array_mut()?.get_mut(idx)?
        } else {
            current.as_object_mut()?.get_mut(*seg)?
        };
        current = next;
    }
    Some(current)
}

fn ensure_path(doc: &mut serde_json::Value, segments: &[&str]) {
    let mut current = doc;
    for seg in segments {
        if current.is_object() && !current.as_object().unwrap().contains_key(*seg) {
            current.as_object_mut().unwrap()
                .insert(seg.to_string(), serde_json::Value::Object(Default::default()));
        }
        let next = if current.is_array() {
            match seg.parse::<usize>().ok().and_then(|i| current.as_array_mut().and_then(|a| a.get_mut(i))) {
                Some(v) => v,
                None => return,
            }
        } else {
            match current.as_object_mut().and_then(|o| o.get_mut(*seg)) {
                Some(v) => v,
                None => return,
            }
        };
        current = next;
    }
}

fn pointer_set(doc: &mut serde_json::Value, path: &str, value: serde_json::Value) {
    let segments = match parse_segments(path) {
        Some(s) => s,
        None => return,
    };
    let (parent_segs, last) = segments.split_at(segments.len() - 1);

    if !parent_segs.is_empty() {
        ensure_path(doc, parent_segs);
    }

    let parent = if parent_segs.is_empty() {
        &mut *doc
    } else {
        match navigate_to_parent(doc, parent_segs) {
            Some(p) => p,
            None => return,
        }
    };

    let key = last[0];
    if let Some(arr) = parent.as_array_mut() {
        if key == "-" {
            arr.push(value);
        } else if let Ok(idx) = key.parse::<usize>() {
            if idx <= arr.len() {
                arr.insert(idx, value);
            }
        }
    } else if let Some(obj) = parent.as_object_mut() {
        obj.insert(key.to_string(), value);
    }
}

fn pointer_remove(doc: &mut serde_json::Value, path: &str) -> bool {
    let segments = match parse_segments(path) {
        Some(s) => s,
        None => return false,
    };
    let (parent_segs, last) = segments.split_at(segments.len() - 1);

    let parent = if parent_segs.is_empty() {
        &mut *doc
    } else {
        match navigate_to_parent(doc, parent_segs) {
            Some(p) => p,
            None => return false,
        }
    };

    let key = last[0];
    if let Some(arr) = parent.as_array_mut() {
        if let Ok(idx) = key.parse::<usize>() {
            if idx < arr.len() {
                arr.remove(idx);
                return true;
            }
        }
        false
    } else {
        parent
            .as_object_mut()
            .map(|o| o.remove(key).is_some())
            .unwrap_or(false)
    }
}

fn pointer_test(doc: &serde_json::Value, path: &str, expected: &serde_json::Value) -> bool {
    doc.pointer(path).is_some_and(|v| v == expected)
}

fn evaluate_condition(cond: &str, doc: &serde_json::Value, ctx: &HookContext) -> bool {
    let meta = ctx.extensions.get::<xlate_core::registry::ModelMeta>();
    match cond {
        "model_lacks_tool_use" => meta.map(|m| !m.capabilities.tool_use).unwrap_or(false),
        "model_has_tool_use" => meta.map(|m| m.capabilities.tool_use).unwrap_or(true),
        "model_lacks_vision" => meta.map(|m| !m.capabilities.vision).unwrap_or(false),
        "model_has_vision" => meta.map(|m| m.capabilities.vision).unwrap_or(false),
        "model_lacks_streaming" => meta.map(|m| !m.capabilities.streaming).unwrap_or(false),
        "model_has_streaming" => meta.map(|m| m.capabilities.streaming).unwrap_or(true),
        c if c.starts_with("has:") => doc.pointer(&c[4..]).is_some(),
        c if c.starts_with("!has:") => doc.pointer(&c[5..]).is_none(),
        _ => {
            tracing::warn!(condition = cond, "unknown patch condition, skipping patch op");
            false
        }
    }
}

#[async_trait]
impl Hook for PatchHook {
    fn name(&self) -> &str {
        "patch"
    }

    fn priority(&self) -> i32 {
        200
    }

    async fn pre_send(&self, ctx: &mut HookContext) -> HookVerdict {
        let model = ctx.request.model.clone();
        for rule in &self.rules {
            if !glob_match::glob_match(&rule.model, &model) {
                continue;
            }
            if rule.patches.is_empty() {
                continue;
            }

            let mut doc = match serde_json::to_value(&ctx.request) {
                Ok(v) => v,
                Err(_) => continue,
            };

            for patch in &rule.patches {
                if let Some(ref cond) = patch.condition {
                    if !evaluate_condition(cond, &doc, ctx) {
                        continue;
                    }
                }
                match patch.op.as_str() {
                    "add" | "replace" => {
                        if let Some(ref val) = patch.value {
                            pointer_set(&mut doc, &patch.path, val.clone());
                            tracing::debug!(
                                op = %patch.op,
                                path = %patch.path,
                                "patch applied"
                            );
                        }
                    }
                    "remove" => {
                        if pointer_remove(&mut doc, &patch.path) {
                            tracing::debug!(path = %patch.path, "patch removed");
                        }
                    }
                    "test" => {
                        if let Some(ref val) = patch.value {
                            if !pointer_test(&doc, &patch.path, val) {
                                tracing::debug!(path = %patch.path, "patch test failed, stopping rule");
                                break;
                            }
                        }
                    }
                    "copy" => {
                        if let Some(ref from) = patch.value.as_ref().and_then(|v| v.as_str()) {
                            if let Some(val) = doc.pointer(from).cloned() {
                                pointer_set(&mut doc, &patch.path, val);
                                tracing::debug!(from = %from, to = %patch.path, "patch copied");
                            }
                        }
                    }
                    "move" => {
                        if let Some(ref from) = patch.value.as_ref().and_then(|v| v.as_str()) {
                            if let Some(val) = doc.pointer(from).cloned() {
                                pointer_remove(&mut doc, from);
                                pointer_set(&mut doc, &patch.path, val);
                                tracing::debug!(from = %from, to = %patch.path, "patch moved");
                            }
                        }
                    }
                    _ => {
                        tracing::warn!(op = %patch.op, "unsupported patch op");
                    }
                }
            }

            if let Ok(patched) = serde_json::from_value(doc) {
                ctx.request = patched;
            }
        }

        HookVerdict::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_set_nested() {
        let mut doc = serde_json::json!({"a": {"b": 1}});
        pointer_set(&mut doc, "/a/c", serde_json::json!(2));
        assert_eq!(doc["a"]["c"], 2);
    }

    #[test]
    fn pointer_remove_nested() {
        let mut doc = serde_json::json!({"a": {"b": 1, "c": 2}});
        assert!(pointer_remove(&mut doc, "/a/b"));
        assert!(doc["a"].get("b").is_none());
        assert_eq!(doc["a"]["c"], 2);
    }

    #[test]
    fn pointer_remove_top() {
        let mut doc = serde_json::json!({"x": 1, "y": 2});
        assert!(pointer_remove(&mut doc, "/x"));
        assert!(doc.get("x").is_none());
    }
}
