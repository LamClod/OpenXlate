use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use xlate_core::error::XlateError;
use xlate_core::event::{EventMeta, ModelEvent, Usage};
use xlate_core::plugin::{EventSink, OutboundPlugin};
use xlate_core::provider::ProviderConfig;
use xlate_core::types::NormalizedRequest;

pub struct MockProvider {
    call_count: Arc<AtomicU64>,
    response_chunks: Vec<String>,
    latency_ms: u64,
    fail_after: Option<usize>,
}

impl MockProvider {
    pub fn new() -> Self {
        Self {
            call_count: Arc::new(AtomicU64::new(0)),
            response_chunks: vec!["Hello".into(), ", ".into(), "world!".into()],
            latency_ms: 0,
            fail_after: None,
        }
    }

    pub fn with_chunks(mut self, chunks: Vec<String>) -> Self {
        self.response_chunks = chunks;
        self
    }

    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    pub fn with_fail_after(mut self, n: usize) -> Self {
        self.fail_after = Some(n);
        self
    }

    pub fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::Relaxed)
    }

    pub fn call_counter(&self) -> Arc<AtomicU64> {
        self.call_count.clone()
    }

    fn meta(model: &str) -> EventMeta {
        EventMeta {
            occurred_at_ms: xlate_core::now_ms(),
            provider: "mock".into(),
            model: model.to_string(),
            provider_item_id: String::new(),
            provider_status: String::new(),
            provider_summary: None,
            provider_call_id: String::new(),
        }
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl xlate_core::plugin::Plugin for MockProvider {
    fn manifest(&self) -> xlate_core::plugin::PluginManifest {
        xlate_core::plugin::PluginManifest {
            id: "mock".into(),
            name: "mock".into(),
            version: "0.1.0".into(),
            kind: xlate_core::plugin::PluginKind::Outbound,
            required_capabilities: vec![],
        }
    }
}

#[async_trait]
impl OutboundPlugin for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn supported_providers(&self) -> &[&str] {
        &["mock"]
    }

    async fn stream(
        &self,
        request: &NormalizedRequest,
        _config: &ProviderConfig,
        sink: &mut dyn EventSink,
    ) -> Result<(), XlateError> {
        let count = self.call_count.fetch_add(1, Ordering::Relaxed) as usize;

        if let Some(fail_after) = self.fail_after {
            if count >= fail_after {
                return Err(XlateError::Provider {
                    status: Some(500),
                    message: "mock failure".into(),
                });
            }
        }

        if self.latency_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
        }

        let model = &request.model;
        for chunk in &self.response_chunks {
            sink.send(ModelEvent::TextDelta {
                text: chunk.clone(),
                meta: Self::meta(model),
            })
            .await?;
        }

        let total_chars: usize = self.response_chunks.iter().map(|c| c.len()).sum();
        sink.send(ModelEvent::TurnFinished {
            finish_reason: "stop".into(),
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(total_chars as i64),
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
                total_tokens: Some(10 + total_chars as i64),
                estimated: false,
            },
            meta: Self::meta(model),
        })
        .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_default_has_three_chunks() {
        let mock = MockProvider::new();
        assert_eq!(mock.response_chunks.len(), 3);
        assert_eq!(mock.call_count(), 0);
    }

    #[test]
    fn mock_builder_pattern() {
        let mock = MockProvider::new()
            .with_chunks(vec!["a".into(), "b".into()])
            .with_latency(100)
            .with_fail_after(3);

        assert_eq!(mock.response_chunks.len(), 2);
        assert_eq!(mock.latency_ms, 100);
        assert_eq!(mock.fail_after, Some(3));
    }
}
