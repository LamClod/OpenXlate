use std::sync::Arc;
use xlate_core::capability::PluginCapabilities;
use xlate_core::config::KernelConfig;
use xlate_core::registry::ModelRegistry;
use xlate_core::store::Store;
use xlate_core::supervisor::PluginStatus;

use crate::catalog::RemotePricingCatalog;
use crate::fetch::PricingFetcher;

pub struct PricingService {
    catalog: Arc<RemotePricingCatalog>,
    config: PricingServiceConfig,
    store: Option<Arc<dyn Store>>,
    registry: Option<Arc<ModelRegistry>>,
    cancel: tokio::sync::watch::Sender<bool>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
    handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

pub struct PricingServiceConfig {
    pub source_url: String,
    pub refresh_interval_s: u64,
    pub fallback_file: Option<String>,
}

impl PricingServiceConfig {
    pub fn from_kernel_config(config: &KernelConfig) -> Self {
        Self {
            source_url: config.billing.pricing.source.clone(),
            refresh_interval_s: config.billing.pricing.refresh_interval_s,
            fallback_file: config.billing.pricing.fallback_file.clone(),
        }
    }
}

impl PricingService {
    pub fn new(catalog: Arc<RemotePricingCatalog>, config: PricingServiceConfig) -> Self {
        let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
        Self {
            catalog,
            config,
            store: None,
            registry: None,
            cancel,
            cancel_rx,
            handle: tokio::sync::Mutex::new(None),
        }
    }

    pub fn with_store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_registry(mut self, registry: Arc<ModelRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    pub fn catalog(&self) -> &Arc<RemotePricingCatalog> {
        &self.catalog
    }
}

impl xlate_core::plugin::Plugin for PricingService {
    fn manifest(&self) -> xlate_core::plugin::PluginManifest {
        xlate_core::plugin::PluginManifest {
            id: "pricing-service".into(),
            name: "pricing-service".into(),
            version: "0.1.0".into(),
            kind: xlate_core::plugin::PluginKind::Service,
            required_capabilities: vec![],
        }
    }
}

#[async_trait::async_trait]
impl xlate_core::plugin::ServicePlugin for PricingService {
    fn name(&self) -> &str {
        "pricing-service"
    }

    async fn start(&self, _caps: PluginCapabilities) -> Result<(), xlate_core::plugin::PluginError> {
        let mut guard = self.handle.lock().await;
        if guard.is_some() {
            tracing::warn!("pricing service already started, ignoring duplicate start");
            return Ok(());
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_default();

        let mut fetcher = PricingFetcher::with_client(&self.config.source_url, self.catalog.clone(), client.clone());
        if let Some(ref file) = self.config.fallback_file {
            fetcher = fetcher.with_fallback_file(file);
        }
        if let Some(ref store) = self.store {
            fetcher = fetcher.with_store(store.clone());
        }

        if let Err(e) = fetcher.fetch_and_load().await {
            tracing::warn!(error = %e, "initial pricing fetch failed, will retry");
        }
        if let Some(ref registry) = self.registry {
            self.catalog.merge_into_registry(registry);
        }

        let interval = std::time::Duration::from_secs(self.config.refresh_interval_s);
        let mut cancel_rx = self.cancel_rx.clone();

        let catalog = self.catalog.clone();
        let registry = self.registry.clone();
        let url = self.config.source_url.clone();
        let fallback = self.config.fallback_file.clone();
        let store = self.store.clone();

        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await;
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let mut f = PricingFetcher::with_client(&url, catalog.clone(), client.clone());
                        if let Some(ref file) = fallback {
                            f = f.with_fallback_file(file);
                        }
                        if let Some(ref s) = store {
                            f = f.with_store(s.clone());
                        }
                        match f.fetch_and_load().await {
                            Ok(count) => {
                                tracing::debug!(count, "pricing catalog refreshed");
                                if let Some(ref reg) = registry {
                                    catalog.merge_into_registry(reg);
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "pricing refresh failed");
                            }
                        }
                    }
                    _ = cancel_rx.changed() => {
                        tracing::info!("pricing service stopping");
                        break;
                    }
                }
            }
        });

        *guard = Some(handle);
        Ok(())
    }

    async fn stop(&self) -> Result<(), xlate_core::plugin::PluginError> {
        let _ = self.cancel.send(true);
        if let Some(handle) = self.handle.lock().await.take() {
            let _ = handle.await;
        }
        Ok(())
    }

    async fn health_check(&self) -> PluginStatus {
        if self.catalog.model_count() > 0 {
            PluginStatus::Running
        } else {
            PluginStatus::Degraded {
                reason: "no pricing data loaded".into(),
            }
        }
    }
}
