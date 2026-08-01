//! Local-only API gateway. It accepts one of the supported client protocols,
//! converts through the shared IR, then calls the configured upstream protocol.
//!
//! Local model ids are `{provider_name}-{upstream_model}`. `/v1/models` discovers
//! upstream models and assembles those ids; request routing strips the prefix.

use std::borrow::Cow;
use std::io;
use std::net::IpAddr;

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::Value;

use super::state::{GatewayState, GatewayStatus, ProviderConfig};
use crate::codec::{Codec, CodecFormat};

const LOOPBACK_ADDRESS: &str = "127.0.0.1:5150";

pub async fn serve(state: GatewayState) {
    let listener = match tokio::net::TcpListener::bind(LOOPBACK_ADDRESS).await {
        Ok(listener) => listener,
        Err(error) => {
            set_gateway_status(
                &state,
                GatewayStatus {
                    running: false,
                    port: 5150,
                    error: Some(format!("无法绑定本地网关 {LOOPBACK_ADDRESS}: {error}")),
                },
            );
            return;
        }
    };

    set_gateway_status(
        &state,
        GatewayStatus {
            running: true,
            port: 5150,
            error: None,
        },
    );

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models).options(cors_preflight))
        .route(
            "/v1/models/{model}",
            get(get_model).options(cors_preflight),
        )
        .route(
            "/v1/chat/completions",
            post(proxy_openai).options(cors_preflight),
        )
        .route(
            "/v1/responses",
            post(proxy_responses).options(cors_preflight),
        )
        .route(
            "/v1/messages",
            post(proxy_anthropic).options(cors_preflight),
        )
        .route(
            "/v1beta/models/{model_action}",
            post(proxy_gemini).options(cors_preflight),
        )
        .with_state(state)
        .layer(middleware::from_fn(add_cors_headers));

    if let Err(error) = axum::serve(listener, app).await {
        // This usually happens only while the desktop process is shutting down.
        eprintln!("local gateway stopped: {error}");
    }
}

async fn health(State(state): State<GatewayState>) -> Response {
    let provider_count = state
        .providers
        .read()
        .map(|providers| providers.iter().filter(|provider| provider.enabled).count())
        .unwrap_or_default();
    Json(serde_json::json!({
        "status": "ok",
        "providers": provider_count,
        "port": 5150,
    }))
    .into_response()
}

/// OpenAI-compatible model list. Pulls each enabled provider's upstream models
/// and returns local routing ids `{name}-{upstream_id}`.
async fn list_models(State(state): State<GatewayState>) -> Response {
    match collect_local_models(&state).await {
        Ok(data) => Json(serde_json::json!({
            "object": "list",
            "data": data,
        }))
        .into_response(),
        Err(message) => error_response(StatusCode::BAD_GATEWAY, message),
    }
}

async fn get_model(State(state): State<GatewayState>, Path(model): Path<String>) -> Response {
    let model = model.trim_start_matches("models/").to_string();
    if let Ok(data) = collect_local_models(&state).await {
        if let Some(entry) = data.into_iter().find(|item| {
            item.get("id")
                .and_then(|value| value.as_str())
                .is_some_and(|id| id == model)
        }) {
            return Json(entry).into_response();
        }
    }

    // Allow direct lookup of a valid routing id even if the upstream list is incomplete.
    match resolve_route(&state, &model) {
        Ok((provider, _)) => Json(serde_json::json!({
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": provider.name,
        }))
        .into_response(),
        Err(message) => error_response(StatusCode::NOT_FOUND, message),
    }
}

async fn collect_local_models(state: &GatewayState) -> Result<Vec<Value>, String> {
    let providers = enabled_providers(state)?;
    let mut entries = Vec::new();
    let mut errors = Vec::new();

    for provider in providers {
        match fetch_upstream_model_ids(state, &provider).await {
            Ok(ids) => {
                for upstream_id in ids {
                    entries.push(serde_json::json!({
                        "id": local_model_id(&provider.name, &upstream_id),
                        "object": "model",
                        "created": 0,
                        "owned_by": provider.name,
                    }));
                }
            }
            Err(error) => errors.push(format!("{}: {error}", provider.name)),
        }
    }

    if entries.is_empty() && !errors.is_empty() {
        return Err(format!(
            "无法从上游获取模型列表：{}",
            errors.join("；")
        ));
    }

    Ok(entries)
}

async fn fetch_upstream_model_ids(
    state: &GatewayState,
    provider: &ProviderConfig,
) -> Result<Vec<String>, String> {
    let url = models_list_url(provider);
    let mut request = state.http_client.get(&url);
    for (name, value) in provider.format.headers(&provider.api_key) {
        // Models list is GET; skip Content-Type from chat headers.
        if name.eq_ignore_ascii_case("content-type") {
            continue;
        }
        request = request.header(name, value);
    }

    let response = request
        .send()
        .await
        .map_err(|error| format!("连接失败: {error}"))?;
    let status = response.status();
    let body = response
        .bytes()
        .await
        .map_err(|error| format!("读取失败: {error}"))?;
    if !status.is_success() {
        let details = String::from_utf8_lossy(&body);
        return Err(format!("上游返回 {status}: {details}"));
    }

    parse_upstream_model_ids(provider.format, &body)
}

fn models_list_url(provider: &ProviderConfig) -> String {
    let base = provider.base_url.trim_end_matches('/');
    match provider.format {
        CodecFormat::Gemini => {
            // Accept either .../v1beta/models or a model-specific base.
            if base.ends_with("/models") {
                base.to_string()
            } else if let Some(prefix) = base.rsplit_once("/models/") {
                format!("{}/models", prefix.0)
            } else {
                format!("{base}/models")
            }
        }
        // OpenAI, Responses, Anthropic-compatible gateways all commonly expose this.
        _ => {
            if base.ends_with("/v1") {
                format!("{base}/models")
            } else if base.contains("/v1/") {
                // e.g. custom path already under /v1/...
                format!(
                    "{}/models",
                    base.split_once("/v1/")
                        .map(|(head, _)| format!("{head}/v1"))
                        .unwrap_or_else(|| base.to_string())
                )
            } else {
                format!("{base}/v1/models")
            }
        }
    }
}

fn parse_upstream_model_ids(format: CodecFormat, body: &[u8]) -> Result<Vec<String>, String> {
    let value: Value =
        serde_json::from_slice(body).map_err(|error| format!("模型列表 JSON 无效: {error}"))?;

    let mut ids = match format {
        CodecFormat::Gemini => parse_gemini_model_ids(&value),
        _ => parse_openai_style_model_ids(&value),
    };

    ids.retain(|id| !id.trim().is_empty());
    ids.sort();
    ids.dedup();
    if ids.is_empty() {
        return Err("上游模型列表为空".to_string());
    }
    Ok(ids)
}

fn parse_openai_style_model_ids(value: &Value) -> Vec<String> {
    value
        .get("data")
        .and_then(|data| data.as_array())
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("id").and_then(|id| id.as_str()).map(str::to_string))
        .collect()
}

fn parse_gemini_model_ids(value: &Value) -> Vec<String> {
    value
        .get("models")
        .and_then(|models| models.as_array())
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let name = item.get("name").and_then(|value| value.as_str())?;
            Some(
                name.strip_prefix("models/")
                    .unwrap_or(name)
                    .to_string(),
            )
        })
        .collect()
}

async fn proxy_openai(State(state): State<GatewayState>, body: Bytes) -> Response {
    proxy(state, body, CodecFormat::OpenAi, false, None).await
}

async fn proxy_responses(State(state): State<GatewayState>, body: Bytes) -> Response {
    proxy(state, body, CodecFormat::OpenAiResponses, false, None).await
}

async fn proxy_anthropic(State(state): State<GatewayState>, body: Bytes) -> Response {
    proxy(state, body, CodecFormat::Anthropic, false, None).await
}

async fn proxy_gemini(
    State(state): State<GatewayState>,
    Path(model_action): Path<String>,
    body: Bytes,
) -> Response {
    if let Some(model) = model_action.strip_suffix(":generateContent") {
        proxy(
            state,
            body,
            CodecFormat::Gemini,
            false,
            Some(model.to_string()),
        )
        .await
    } else if let Some(model) = model_action.strip_suffix(":streamGenerateContent") {
        proxy(
            state,
            body,
            CodecFormat::Gemini,
            true,
            Some(model.to_string()),
        )
        .await
    } else {
        error_response(StatusCode::NOT_FOUND, "未知的 Gemini 操作路径".to_string())
    }
}

async fn proxy(
    state: GatewayState,
    body: Bytes,
    source_format: CodecFormat,
    source_stream_endpoint: bool,
    path_model: Option<String>,
) -> Response {
    let codec = Codec::default();
    let mut request = match codec.decode_request(source_format, &body) {
        Ok(request) => request,
        Err(error) => {
            return error_response(StatusCode::BAD_REQUEST, format!("请求格式无效: {error}"))
        }
    };

    let local_model = path_model.as_deref().unwrap_or(request.model.as_ref());
    let (provider, upstream_model) = match resolve_route(&state, local_model) {
        Ok(resolved) => resolved,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, message),
    };

    request.stream |= source_stream_endpoint;
    request.model = Cow::Owned(upstream_model.clone());

    let target_body = match codec.encode_request(provider.format, &request) {
        Ok(payload) => payload,
        Err(error) => {
            return error_response(StatusCode::BAD_REQUEST, format!("无法转换请求: {error}"))
        }
    };
    let is_stream = request.stream;
    let endpoint = upstream_endpoint(&provider, &upstream_model, is_stream);

    let mut upstream_request = state.http_client.post(endpoint).body(target_body);
    for (name, value) in provider.format.headers(&provider.api_key) {
        upstream_request = upstream_request.header(name, value);
    }

    let upstream = match upstream_request.send().await {
        Ok(response) => response,
        Err(error) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("无法连接上游供应商: {error}"),
            )
        }
    };
    let upstream_status = upstream.status();
    if !upstream_status.is_success() {
        let details = upstream
            .text()
            .await
            .unwrap_or_else(|_| "上游返回了无法读取的错误响应".to_string());
        return error_response(
            StatusCode::BAD_GATEWAY,
            format!("上游供应商返回 {upstream_status}: {details}"),
        );
    }

    if is_stream {
        return stream_response(upstream, codec, provider.format, source_format);
    }

    let upstream_body = match upstream.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("无法读取上游响应: {error}"),
            )
        }
    };
    let response = match codec.decode_response(provider.format, &upstream_body) {
        Ok(response) => response,
        Err(error) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("无法解析上游响应: {error}"),
            )
        }
    };
    let output = match codec.encode_response(source_format, &response) {
        Ok(output) => output,
        Err(error) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("无法转换上游响应: {error}"),
            )
        }
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json; charset=utf-8")],
        output,
    )
        .into_response()
}

fn stream_response(
    upstream: reqwest::Response,
    codec: Codec,
    source_format: CodecFormat,
    target_format: CodecFormat,
) -> Response {
    let transcoder = match codec.sse_stream_transcoder(source_format, target_format) {
        Ok(transcoder) => transcoder,
        Err(error) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("无法建立流式转换: {error}"),
            )
        }
    };
    let input = Box::pin(upstream.bytes_stream());
    let stream = futures::stream::unfold(
        (input, transcoder, false),
        |(mut input, mut transcoder, finished)| async move {
            if finished {
                return None;
            }
            match input.next().await {
                Some(Ok(chunk)) => match transcoder.push(&chunk) {
                    Ok(output) => Some((
                        Ok::<Bytes, io::Error>(Bytes::from(output)),
                        (input, transcoder, false),
                    )),
                    Err(error) => Some((
                        Err(io::Error::other(format!("流式转换失败: {error}"))),
                        (input, transcoder, true),
                    )),
                },
                Some(Err(error)) => Some((
                    Err(io::Error::other(format!("上游流中断: {error}"))),
                    (input, transcoder, true),
                )),
                None => match transcoder.finish() {
                    Ok(output) if output.is_empty() => None,
                    Ok(output) => Some((Ok(Bytes::from(output)), (input, transcoder, true))),
                    Err(error) => Some((
                        Err(io::Error::other(format!("流式收尾失败: {error}"))),
                        (input, transcoder, true),
                    )),
                },
            }
        },
    );

    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-transform"),
    );
    response
}

fn enabled_providers(state: &GatewayState) -> Result<Vec<ProviderConfig>, String> {
    let providers = state
        .providers
        .read()
        .map_err(|error| format!("无法读取供应商配置: {error}"))?;
    Ok(providers
        .iter()
        .filter(|provider| provider.enabled)
        .cloned()
        .collect())
}

/// Match `{provider_name}-{upstream_model}` using the longest enabled provider
/// name prefix, then return the stripped upstream model id.
fn resolve_route(
    state: &GatewayState,
    local_model: &str,
) -> Result<(ProviderConfig, String), String> {
    let providers = enabled_providers(state)?;
    let mut best: Option<(ProviderConfig, String)> = None;

    for provider in providers {
        let prefix = format!("{}-", provider.name);
        if let Some(upstream) = local_model.strip_prefix(&prefix) {
            if upstream.is_empty() {
                continue;
            }
            let better = best
                .as_ref()
                .map(|(current, _)| provider.name.len() > current.name.len())
                .unwrap_or(true);
            if better {
                best = Some((provider, upstream.to_string()));
            }
        }
    }

    best.ok_or_else(|| {
        format!(
            "未找到本地模型 {local_model}；模型名称必须为 供应商名称-上游模型名，且供应商必须已启用"
        )
    })
}

fn local_model_id(provider_name: &str, upstream_model: &str) -> String {
    format!("{provider_name}-{upstream_model}")
}

fn upstream_endpoint(provider: &ProviderConfig, upstream_model: &str, stream: bool) -> String {
    let base_url = if provider.format == CodecFormat::Gemini {
        gemini_model_base(&provider.base_url, upstream_model)
    } else {
        provider.base_url.clone()
    };
    let endpoint = provider.format.endpoint(&base_url);
    if stream && provider.format == CodecFormat::Gemini {
        endpoint.replacen(":generateContent", ":streamGenerateContent?alt=sse", 1)
    } else {
        endpoint
    }
}

fn gemini_model_base(base_url: &str, upstream_model: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with(&format!("/models/{upstream_model}")) {
        base.to_string()
    } else if base.ends_with("/models") {
        format!("{base}/{upstream_model}")
    } else if let Some((prefix, _)) = base.rsplit_once("/models/") {
        format!("{prefix}/models/{upstream_model}")
    } else {
        format!("{base}/{upstream_model}")
    }
}

fn error_response(status: StatusCode, message: String) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json; charset=utf-8")],
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "openxlate_gateway_error"
            }
        })),
    )
        .into_response()
}

async fn add_cors_headers(request: axum::extract::Request, next: Next) -> Response {
    let allowed_origin = request
        .headers()
        .get(header::ORIGIN)
        .filter(|origin| is_allowed_origin(origin))
        .cloned();
    let mut response = next.run(request).await;
    if let Some(origin) = allowed_origin {
        response
            .headers_mut()
            .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
        response.headers_mut().insert(
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            HeaderValue::from_static("content-type, authorization"),
        );
        response.headers_mut().insert(
            header::ACCESS_CONTROL_ALLOW_METHODS,
            HeaderValue::from_static("GET, POST, OPTIONS"),
        );
        response
            .headers_mut()
            .insert(header::VARY, HeaderValue::from_static("Origin"));
    }
    response
}

async fn cors_preflight() -> StatusCode {
    StatusCode::NO_CONTENT
}

fn is_allowed_origin(origin: &HeaderValue) -> bool {
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    if matches!(origin, "tauri://localhost" | "http://tauri.localhost") {
        return true;
    }
    let Ok(url) = reqwest::Url::parse(origin) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    match url.host_str() {
        Some("localhost") => true,
        Some(host) => host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback()),
        None => false,
    }
}

fn set_gateway_status(state: &GatewayState, status: GatewayStatus) {
    if let Ok(mut current) = state.status.write() {
        *current = status;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn sample_state(providers: Vec<ProviderConfig>) -> GatewayState {
        GatewayState {
            providers: Arc::new(std::sync::RwLock::new(providers)),
            http_client: reqwest::Client::new(),
            status: Arc::new(std::sync::RwLock::new(GatewayStatus::starting())),
        }
    }

    #[test]
    fn resolve_route_strips_provider_prefix() {
        let state = sample_state(vec![
            ProviderConfig {
                id: "1".into(),
                name: "lamclod".into(),
                format: CodecFormat::OpenAi,
                base_url: "https://api.example.com".into(),
                api_key: "key".into(),
                enabled: true,
            },
            ProviderConfig {
                id: "2".into(),
                name: "other".into(),
                format: CodecFormat::OpenAi,
                base_url: "https://api.other.com".into(),
                api_key: "key".into(),
                enabled: true,
            },
        ]);

        let (provider, upstream) =
            resolve_route(&state, "lamclod-gpt-5.6-sol").expect("route should resolve");
        assert_eq!(provider.name, "lamclod");
        assert_eq!(upstream, "gpt-5.6-sol");
    }

    #[test]
    fn resolve_route_prefers_longest_provider_name() {
        let state = sample_state(vec![
            ProviderConfig {
                id: "1".into(),
                name: "ai".into(),
                format: CodecFormat::OpenAi,
                base_url: "https://api.example.com".into(),
                api_key: "key".into(),
                enabled: true,
            },
            ProviderConfig {
                id: "2".into(),
                name: "ai-pro".into(),
                format: CodecFormat::OpenAi,
                base_url: "https://api.pro.example.com".into(),
                api_key: "key".into(),
                enabled: true,
            },
        ]);

        // "ai-pro" contains a hyphen which validation forbids for new providers,
        // but longest-prefix matching still protects against ambiguous routing.
        let (provider, upstream) =
            resolve_route(&state, "ai-pro-model-x").expect("route should resolve");
        assert_eq!(provider.name, "ai-pro");
        assert_eq!(upstream, "model-x");
    }

    #[test]
    fn gemini_stream_endpoint_uses_request_model() {
        let provider = ProviderConfig {
            id: "gemini".into(),
            name: "Gemini".into(),
            format: CodecFormat::Gemini,
            base_url: "https://generativelanguage.googleapis.com/v1beta/models".into(),
            api_key: "key".into(),
            enabled: true,
        };
        assert_eq!(
            upstream_endpoint(&provider, "gemini-2.5-flash", true),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn openai_models_url_appends_v1_models() {
        let provider = ProviderConfig {
            id: "1".into(),
            name: "OpenAI".into(),
            format: CodecFormat::OpenAi,
            base_url: "https://api.openai.com".into(),
            api_key: "key".into(),
            enabled: true,
        };
        assert_eq!(models_list_url(&provider), "https://api.openai.com/v1/models");
    }

    #[test]
    fn parse_openai_and_gemini_model_lists() {
        let openai = br#"{"object":"list","data":[{"id":"gpt-4o"},{"id":"o1"}]}"#;
        assert_eq!(
            parse_upstream_model_ids(CodecFormat::OpenAi, openai).unwrap(),
            vec!["gpt-4o".to_string(), "o1".to_string()]
        );

        let gemini = br#"{"models":[{"name":"models/gemini-2.5-flash"},{"name":"models/gemini-2.5-pro"}]}"#;
        assert_eq!(
            parse_upstream_model_ids(CodecFormat::Gemini, gemini).unwrap(),
            vec![
                "gemini-2.5-flash".to_string(),
                "gemini-2.5-pro".to_string()
            ]
        );
    }

    #[test]
    fn cors_only_allows_local_application_origins() {
        assert!(is_allowed_origin(&HeaderValue::from_static(
            "http://127.0.0.1:3000"
        )));
        assert!(is_allowed_origin(&HeaderValue::from_static(
            "http://localhost:5173"
        )));
        assert!(is_allowed_origin(&HeaderValue::from_static(
            "tauri://localhost"
        )));
        assert!(!is_allowed_origin(&HeaderValue::from_static(
            "https://example.com"
        )));
        assert!(!is_allowed_origin(&HeaderValue::from_static(
            "http://127.0.0.1.example.com"
        )));
    }
}
