pub mod mock_provider;

pub use mock_provider::MockProvider;

use std::sync::Arc;

use xlate_core::config::{KernelConfig, RouteRuleConfig};
use xlate_core::error::XlateError;
use xlate_core::event::ModelEvent;
use xlate_core::kernel::{EventBus, Kernel, KernelBuilder};
use xlate_core::plugin::{EventSink, OutboundPlugin};
use xlate_core::provider::{ProviderConfig, RouteTarget};
use xlate_core::registry::ModelRegistry;
use xlate_core::store::Store;
use xlate_core::supervisor::Supervisor;

struct ChannelSink {
    tx: tokio::sync::mpsc::Sender<ModelEvent>,
}

#[async_trait::async_trait]
impl EventSink for ChannelSink {
    async fn send(&mut self, event: ModelEvent) -> Result<(), XlateError> {
        self.tx.send(event).await.map_err(|_| XlateError::Canceled)
    }
}

pub fn test_config(model: &str) -> KernelConfig {
    KernelConfig {
        routes: vec![RouteRuleConfig {
            group: "default".into(),
            model: model.to_string(),
            strategy: Default::default(),
            targets: vec![RouteTarget {
                id: "mock-target".into(),
                plugin: "mock".into(),
                config: ProviderConfig {
                    plugin: "mock".into(),
                    base_url: "http://mock.test".into(),
                    api_key: "test-key".into(),
                    upstream_model: None,
                    endpoint: None,
                    extra_params: None,
                    extra_headers: Default::default(),
                    anthropic_version: None,
                    max_tokens_override: None,
                },
                priority: 1,
                weight: 100,
                enabled: true,
            }],
            failover: Default::default(),
            patches: vec![],
        }],
        ..Default::default()
    }
}

pub fn build_test_kernel(config: KernelConfig, mock: MockProvider) -> Kernel {
    let outbound: Vec<Arc<dyn OutboundPlugin>> = vec![Arc::new(mock)];
    let store: Arc<dyn Store> = Arc::new(xlate_store::MemoryStore::new());
    let registry = Arc::new(ModelRegistry::new());
    let supervisor = Arc::new(Supervisor::new());
    let event_bus = Arc::new(EventBus::new(256));

    let result = xlate_hooks::standard_hooks(
        &config,
        Some(store.clone()),
        Some(registry.clone()),
        Some(event_bus.clone()),
        None,
        Some(supervisor.clone()),
    );

    KernelBuilder::new(config)
        .outbound_vec(outbound)
        .hooks(result.hooks)
        .latency_tracker(result.latency_tracker)
        .registry(registry)
        .supervisor(supervisor)
        .store(store)
        .event_bus(event_bus)
        .build()
}

pub async fn stream_collect(
    kernel: &Kernel,
    body: &serde_json::Value,
) -> (Vec<ModelEvent>, Option<XlateError>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ModelEvent>(256);
    let mut sink = ChannelSink { tx };

    let err = kernel.stream_raw(body, &mut sink).await.err();
    drop(sink);

    let mut events = Vec::new();
    while let Some(evt) = rx.recv().await {
        events.push(evt);
    }
    (events, err)
}

pub fn load_fixture(name: &str) -> serde_json::Value {
    let path = format!(
        "{}/src/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("failed to parse fixture {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_provider_streams_chunks() {
        let config = test_config("test-model");
        let mock = MockProvider::new();
        let counter = mock.call_counter();
        let kernel = build_test_kernel(config, mock);

        let fixture = load_fixture("minimal_request.json");
        let (events, err) = stream_collect(&kernel, &fixture).await;

        assert!(err.is_none(), "stream error: {err:?}");

        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ModelEvent::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(text, "Hello, world!");
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn custom_chunks_flow_through() {
        let config = test_config("test-model");
        let mock = MockProvider::new().with_chunks(vec!["foo".into(), "bar".into()]);
        let kernel = build_test_kernel(config, mock);

        let fixture = load_fixture("minimal_request.json");
        let (events, err) = stream_collect(&kernel, &fixture).await;

        assert!(err.is_none());

        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ModelEvent::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "foobar");

        let has_finish = events
            .iter()
            .any(|e| matches!(e, ModelEvent::TurnFinished { .. }));
        assert!(has_finish);
    }

    #[tokio::test]
    async fn kernel_stats_after_stream() {
        let config = test_config("test-model");
        let mock = MockProvider::new();
        let kernel = build_test_kernel(config, mock);

        let fixture = load_fixture("minimal_request.json");
        let (_, err) = stream_collect(&kernel, &fixture).await;
        assert!(err.is_none());

        let stats = kernel.stats();
        assert!(stats.total_streams >= 1);
    }

    #[tokio::test]
    async fn fixture_loading_works() {
        let openai = load_fixture("openai_chat_request.json");
        assert_eq!(openai["model"], "gpt-4o");
        assert!(openai["messages"].is_array());

        let anthropic = load_fixture("anthropic_messages_request.json");
        assert_eq!(anthropic["model"], "claude-sonnet-4-20250514");
    }

    #[tokio::test]
    async fn kernel_shutdown_prevents_new_streams() {
        let config = test_config("test-model");
        let mock = MockProvider::new();
        let kernel = build_test_kernel(config, mock);

        kernel.shutdown();

        let fixture = load_fixture("minimal_request.json");
        let (_, err) = stream_collect(&kernel, &fixture).await;
        assert!(err.is_some(), "expected error after shutdown");
    }
}
