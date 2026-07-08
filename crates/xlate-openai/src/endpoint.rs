//! Endpoint URL resolution for OpenAI-compatible providers.
//!
//! Ported from the original Go implementation's `OpenAIEndpointURL`, `trailingVersionSegment`,
//! `stripEndpointVersionPrefix`, `ResolveOpenAIEndpoint`, `OpenAIEndpointFromBaseURL`.
//! This rule chain is critical for compatibility with domestic relay gateways
//! (Z.AI /v4, DashScope, etc.).

/// Well-known OpenAI endpoint paths.
pub const ENDPOINT_RESPONSES: &str = "/v1/responses";
pub const ENDPOINT_CHAT_COMPLETIONS: &str = "/v1/chat/completions";
pub const ENDPOINT_CUSTOM: &str = "/custom";

/// Detect the protocol "shape" from an endpoint path.
/// Returns `"responses"` or `"chat/completions"`.
pub fn endpoint_shape(endpoint: &str) -> &'static str {
    let lower = endpoint.trim().to_lowercase();
    if lower.ends_with("/responses") {
        "responses"
    } else {
        "chat/completions"
    }
}

/// Normalize an OpenAI endpoint string.
/// Supports three presets: `/v1/responses`, `/v1/chat/completions`, `/custom`.
pub fn normalize_endpoint(endpoint: &str) -> Option<&'static str> {
    let normalized = endpoint.trim();
    match normalized {
        "" => Some(ENDPOINT_RESPONSES),
        s if s == ENDPOINT_RESPONSES => Some(ENDPOINT_RESPONSES),
        s if s == ENDPOINT_CHAT_COMPLETIONS => Some(ENDPOINT_CHAT_COMPLETIONS),
        s if s == ENDPOINT_CUSTOM => Some(ENDPOINT_CUSTOM),
        _ => None,
    }
}

/// Resolve which endpoint to use given a base URL and an optional endpoint hint.
/// If the base URL already ends with a known endpoint suffix, use that.
/// Otherwise, normalize the given endpoint string.
pub fn resolve_endpoint(base_url: &str, endpoint: &str) -> Option<String> {
    if let Some(ep) = endpoint_from_base_url(base_url) {
        return Some(ep.to_string());
    }
    normalize_endpoint(endpoint).map(|s| s.to_string())
}

/// Detect if a base URL already contains an endpoint suffix.
pub fn endpoint_from_base_url(base_url: &str) -> Option<&'static str> {
    let base = base_url.trim().trim_end_matches('/').to_lowercase();
    if base.ends_with("/responses") {
        Some(ENDPOINT_RESPONSES)
    } else if base.ends_with("/chat/completions") {
        Some(ENDPOINT_CHAT_COMPLETIONS)
    } else {
        None
    }
}

/// Build the full request URL from a base URL and an endpoint.
///
/// Rules (ported from Go `OpenAIEndpointURL`):
///  0. Custom endpoint: if baseURL already contains endpoint suffix, use as-is.
///     Otherwise append `/chat/completions`.
///  1. If baseURL already ends with endpoint suffix, use baseURL directly.
///  2. If baseURL ends with `/vN`, strip version prefix from endpoint to avoid
///     double-versioning (e.g. `.../v4` + `/v1/chat/completions` -> `.../v4/chat/completions`).
///  3. Fallback: concatenate base + endpoint.
pub fn build_endpoint_url(base_url: &str, endpoint: &str) -> String {
    let base = base_url.trim().trim_end_matches('/');
    let mut normalized_endpoint = endpoint.trim().to_string();
    if normalized_endpoint.is_empty() {
        normalized_endpoint = ENDPOINT_RESPONSES.to_string();
    }
    if !normalized_endpoint.starts_with('/') {
        normalized_endpoint = format!("/{normalized_endpoint}");
    }

    // Rule 0: custom path mode
    if normalized_endpoint == ENDPOINT_CUSTOM {
        if endpoint_from_base_url(base).is_some() {
            return base.to_string();
        }
        return format!("{base}/chat/completions");
    }

    // Rule 1: baseURL already contains endpoint suffix
    if endpoint_from_base_url(base).is_some() {
        return base.to_string();
    }

    // Rule 2: baseURL ends with /vN — strip version prefix from endpoint
    if trailing_version_segment(base).is_some() {
        if let Some(rest) = strip_endpoint_version_prefix(&normalized_endpoint) {
            return format!("{base}{rest}");
        }
    }

    // Rule 3: fallback
    format!("{base}{normalized_endpoint}")
}

/// Check if a URL path ends with `/vN` (N is a positive integer).
/// Returns the version segment (e.g. `"v4"`) if matched.
fn trailing_version_segment(base: &str) -> Option<&str> {
    let idx = base.rfind('/')?;
    let seg = &base[idx + 1..];
    if seg.len() < 2 || !seg.starts_with('v') {
        return None;
    }
    if seg[1..].chars().all(|c| c.is_ascii_digit()) && !seg[1..].is_empty() {
        Some(seg)
    } else {
        None
    }
}

/// Strip a leading `/vN/` prefix from an endpoint path.
/// `/v1/chat/completions` -> `Some("/chat/completions")`
/// `/chat/completions` -> `None`
fn strip_endpoint_version_prefix(endpoint: &str) -> Option<&str> {
    let bytes = endpoint.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'/' || bytes[1] != b'v' {
        return None;
    }
    let mut i = 2;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 2 || i >= bytes.len() || bytes[i] != b'/' {
        return None;
    }
    Some(&endpoint[i..])
}

/// Check if a provider URL ends with any of the given endpoint suffixes.
pub fn provider_url_has_endpoint(base_url: &str, endpoints: &[&str]) -> bool {
    let base = base_url.trim().trim_end_matches('/').to_lowercase();
    if base.is_empty() {
        return false;
    }
    for ep in endpoints {
        let mut normalized = ep.trim().trim_end_matches('/').to_lowercase();
        if normalized.is_empty() {
            continue;
        }
        if !normalized.starts_with('/') {
            normalized = format!("/{normalized}");
        }
        if base.ends_with(&normalized) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trailing_version_segment() {
        assert_eq!(trailing_version_segment("https://api.z.ai/v4"), Some("v4"));
        assert_eq!(trailing_version_segment("https://api.openai.com/v1"), Some("v1"));
        assert_eq!(trailing_version_segment("https://api.openai.com"), None);
        assert_eq!(trailing_version_segment("https://api.openai.com/va"), None);
    }

    #[test]
    fn test_strip_endpoint_version_prefix() {
        assert_eq!(
            strip_endpoint_version_prefix("/v1/chat/completions"),
            Some("/chat/completions")
        );
        assert_eq!(strip_endpoint_version_prefix("/chat/completions"), None);
        assert_eq!(strip_endpoint_version_prefix("/v/chat"), None);
    }

    #[test]
    fn test_build_endpoint_url_basic() {
        assert_eq!(
            build_endpoint_url("https://api.openai.com/v1", "/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_endpoint_url_version_dedup() {
        // Base ends with /v4, endpoint starts with /v1/ -> strip to avoid double version
        assert_eq!(
            build_endpoint_url("https://api.z.ai/v4", "/v1/chat/completions"),
            "https://api.z.ai/v4/chat/completions"
        );
    }

    #[test]
    fn test_build_endpoint_url_already_has_endpoint() {
        assert_eq!(
            build_endpoint_url("https://proxy.example.com/v1/chat/completions", "/v1/responses"),
            "https://proxy.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_endpoint_url_custom_with_endpoint_suffix() {
        assert_eq!(
            build_endpoint_url("https://proxy.example.com/v1/chat/completions", "/custom"),
            "https://proxy.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_endpoint_url_custom_without_endpoint_suffix() {
        assert_eq!(
            build_endpoint_url("https://api.example.com/v1", "/custom"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_endpoint_shape() {
        assert_eq!(endpoint_shape("/v1/responses"), "responses");
        assert_eq!(endpoint_shape("/v1/chat/completions"), "chat/completions");
        assert_eq!(endpoint_shape("/custom"), "chat/completions");
    }

    #[test]
    fn test_resolve_endpoint() {
        assert_eq!(
            resolve_endpoint("https://api.openai.com/v1", ""),
            Some(ENDPOINT_RESPONSES.to_string())
        );
        assert_eq!(
            resolve_endpoint("https://api.openai.com/v1/chat/completions", ""),
            Some(ENDPOINT_CHAT_COMPLETIONS.to_string())
        );
    }
}
