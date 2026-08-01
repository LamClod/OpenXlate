use std::borrow::Cow;

use tauri::State;

use super::state::{AppState, GatewayStatus, ProviderConfig, ProviderInput, Settings};
use crate::codec::ir::*;
use crate::codec::{Codec, CodecFormat};

#[derive(serde::Serialize)]
pub struct LanguageInfo {
    pub code: String,
    pub name: String,
}

fn select_format(provider: &str) -> CodecFormat {
    match provider {
        "anthropic" => CodecFormat::Anthropic,
        "google" | "gemini" => CodecFormat::Gemini,
        "openai-responses" => CodecFormat::OpenAiResponses,
        _ => CodecFormat::OpenAi, // openai, deepseek, 所有 OpenAI 兼容
    }
}

fn default_base_url(provider: &str) -> &'static str {
    match provider {
        "anthropic" => "https://api.anthropic.com",
        "deepseek" => "https://api.deepseek.com",
        "google" | "gemini" => {
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash"
        }
        _ => "https://api.openai.com", // openai, openai-responses 共用
    }
}

#[tauri::command]
pub async fn translate_text(
    state: State<'_, AppState>,
    text: String,
    source_lang: String,
    target_lang: String,
) -> Result<String, String> {
    let (api_key, api_base_url, model, provider) = {
        let settings = state.settings.read().map_err(|e| e.to_string())?;
        (
            settings.api_key.clone(),
            settings.api_base_url.clone(),
            settings.model.clone(),
            settings.provider.clone(),
        )
    };

    if api_key.is_empty() {
        return Err("请先在设置中配置 API Key".to_string());
    }

    let source_desc = if source_lang == "auto" {
        "auto-detected language"
    } else {
        &source_lang
    };

    let system_prompt = format!(
        "You are a professional translator. Translate the following text from {} to {}. \
         Only output the translated text, no explanations or extra content.",
        source_desc, target_lang
    );

    // ── 构建 IR ──
    let ir = IrRequest {
        model: Cow::Borrowed(&model),
        messages: vec![
            IrMessage {
                role: Role::System,
                content: IrContent::Text(Cow::Owned(system_prompt)),
                tool_call_id: None,
                tool_name: None,
                tool_calls: None,
                cache_control: None,
                refusal: None,
            },
            IrMessage {
                role: Role::User,
                content: IrContent::Text(Cow::Borrowed(&text)),
                tool_call_id: None,
                tool_name: None,
                tool_calls: None,
                cache_control: None,
                refusal: None,
            },
        ],
        temperature: Some(0.3),
        top_p: None,
        top_k: None,
        max_tokens: Some(4096),
        stop: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        n: None,
        logprobs: None,
        top_logprobs: None,
        stream: false,
        store: None,
        modalities: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        reasoning: None,
        response_format: None,
        previous_response_id: None,
        truncation: None,
        metadata: None,
        provider_metadata: None,
        metadata_mode: MetadataMode::Preserve,
    };

    // ── 编码 ──
    let format = select_format(&provider);
    let codec = Codec::default();

    let body_bytes = codec
        .encode_request(format, &ir)
        .map_err(|e| format!("编码失败: {e}"))?;

    let base_url = if api_base_url.is_empty() {
        default_base_url(&provider).to_string()
    } else {
        api_base_url.trim_end_matches('/').to_string()
    };

    let endpoint = format.endpoint(&base_url);
    let headers = format.headers(&api_key);

    // ── 发送 ──
    let mut req_builder = state.http_client().post(&endpoint);
    for (key, value) in &headers {
        req_builder = req_builder.header(*key, value);
    }

    let response = req_builder
        .body(body_bytes)
        .send()
        .await
        .map_err(|e| format!("请求失败: {e}"))?;

    let status = response.status();
    let response_bytes = response.bytes().await.map_err(|e| e.to_string())?;

    if !status.is_success() {
        return Err(format!(
            "API 错误 ({}): {}",
            status,
            String::from_utf8_lossy(&response_bytes)
        ));
    }

    // ── 解码 ──
    let ir_resp = codec
        .decode_response(format, &response_bytes)
        .map_err(|e| format!("解码失败: {e}"))?;

    ir_resp
        .choices
        .first()
        .and_then(|c| c.message.content.as_text())
        .map(|s| s.to_string())
        .ok_or_else(|| "无法解析翻译结果".to_string())
}

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> Result<Settings, String> {
    state
        .settings
        .read()
        .map(|s| s.clone())
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn update_settings(state: State<'_, AppState>, settings: Settings) -> Result<(), String> {
    let mut current = state.settings.write().map_err(|e| e.to_string())?;
    *current = settings;
    Ok(())
}

#[tauri::command]
pub fn list_providers(state: State<'_, AppState>) -> Result<Vec<ProviderConfig>, String> {
    state.providers()
}

#[tauri::command]
pub fn create_provider(
    state: State<'_, AppState>,
    input: ProviderInput,
) -> Result<ProviderConfig, String> {
    state.create_provider(input)
}

#[tauri::command]
pub fn update_provider(
    state: State<'_, AppState>,
    provider: ProviderConfig,
) -> Result<ProviderConfig, String> {
    state.update_provider(provider)
}

#[tauri::command]
pub fn delete_provider(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.delete_provider(&id)
}

#[tauri::command]
pub fn get_gateway_status(state: State<'_, AppState>) -> Result<GatewayStatus, String> {
    state.gateway_status()
}

#[tauri::command]
pub fn detect_language(_text: String) -> Result<String, String> {
    Ok("auto".to_string())
}

#[tauri::command]
pub fn get_supported_languages() -> Vec<LanguageInfo> {
    vec![
        LanguageInfo {
            code: "auto".into(),
            name: "自动检测".into(),
        },
        LanguageInfo {
            code: "zh".into(),
            name: "中文".into(),
        },
        LanguageInfo {
            code: "en".into(),
            name: "English".into(),
        },
        LanguageInfo {
            code: "ja".into(),
            name: "日本語".into(),
        },
        LanguageInfo {
            code: "ko".into(),
            name: "한국어".into(),
        },
        LanguageInfo {
            code: "fr".into(),
            name: "Français".into(),
        },
        LanguageInfo {
            code: "de".into(),
            name: "Deutsch".into(),
        },
        LanguageInfo {
            code: "es".into(),
            name: "Español".into(),
        },
        LanguageInfo {
            code: "ru".into(),
            name: "Русский".into(),
        },
        LanguageInfo {
            code: "pt".into(),
            name: "Português".into(),
        },
        LanguageInfo {
            code: "it".into(),
            name: "Italiano".into(),
        },
        LanguageInfo {
            code: "ar".into(),
            name: "العربية".into(),
        },
        LanguageInfo {
            code: "th".into(),
            name: "ไทย".into(),
        },
        LanguageInfo {
            code: "vi".into(),
            name: "Tiếng Việt".into(),
        },
    ]
}
