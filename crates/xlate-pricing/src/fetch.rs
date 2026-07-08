use std::sync::Arc;
use xlate_core::error::XlateError;
use xlate_core::store::Store;

use crate::catalog::RemotePricingCatalog;

pub struct PricingFetcher {
    url: String,
    fallback_file: Option<String>,
    catalog: Arc<RemotePricingCatalog>,
    store: Option<Arc<dyn Store>>,
    client: reqwest::Client,
}

impl PricingFetcher {
    pub fn new(url: &str, catalog: Arc<RemotePricingCatalog>) -> Self {
        Self::with_client(url, catalog, reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_default())
    }

    pub fn with_client(url: &str, catalog: Arc<RemotePricingCatalog>, client: reqwest::Client) -> Self {
        Self {
            url: url.to_string(),
            fallback_file: None,
            catalog,
            store: None,
            client,
        }
    }

    pub fn with_fallback_file(mut self, path: &str) -> Self {
        self.fallback_file = Some(path.to_string());
        self
    }

    pub fn with_store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    pub async fn fetch_and_load(&self) -> Result<usize, XlateError> {
        match self.fetch_remote().await {
            Ok(count) => {
                tracing::info!(count, url = %self.url, "loaded pricing from remote");
                Ok(count)
            }
            Err(remote_err) => {
                tracing::warn!(error = %remote_err, "remote pricing fetch failed, trying fallbacks");
                if let Ok(count) = self.load_from_store().await {
                    if count > 0 {
                        tracing::info!(count, "loaded pricing from store cache");
                        return Ok(count);
                    }
                }
                if let Ok(count) = self.load_from_file() {
                    if count > 0 {
                        tracing::info!(count, "loaded pricing from fallback file");
                        return Ok(count);
                    }
                }
                Err(remote_err)
            }
        }
    }

    async fn fetch_remote(&self) -> Result<usize, XlateError> {
        let resp = self
            .client
            .get(&self.url)
            .send()
            .await
            .map_err(|e| XlateError::Transport(format!("pricing fetch: {e}")))?;

        if !resp.status().is_success() {
            return Err(XlateError::Provider {
                status: Some(resp.status().as_u16()),
                message: format!("pricing API returned {}", resp.status()),
            });
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| XlateError::Transport(format!("pricing parse: {e}")))?;

        let count = self.catalog.load_from_json(&data);

        if let Some(ref store) = self.store {
            let snapshot = xlate_core::store::PricingSnapshot {
                fetched_at: xlate_core::now_ms(),
                data: data.clone(),
            };
            if let Err(e) = store.save_pricing(snapshot).await {
                tracing::warn!(error = %e, "failed to cache pricing to store");
            }
        }

        Ok(count)
    }

    async fn load_from_store(&self) -> Result<usize, XlateError> {
        let store = self.store.as_ref().ok_or_else(|| {
            XlateError::Internal("no store configured".into())
        })?;
        let snapshot = store
            .load_pricing()
            .await
            .map_err(|e| XlateError::Internal(e.to_string()))?
            .ok_or_else(|| XlateError::Internal("no cached pricing".into()))?;
        Ok(self.catalog.load_from_json(&snapshot.data))
    }

    fn load_from_file(&self) -> Result<usize, XlateError> {
        let path = self.fallback_file.as_ref().ok_or_else(|| {
            XlateError::Internal("no fallback file configured".into())
        })?;
        let content = std::fs::read_to_string(path)
            .map_err(|e| XlateError::Internal(format!("read fallback file: {e}")))?;
        let data: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| XlateError::Internal(format!("parse fallback file: {e}")))?;
        Ok(self.catalog.load_from_json(&data))
    }
}
