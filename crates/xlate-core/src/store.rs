use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("store IO error: {0}")]
    Io(String),
    #[error("store query error: {0}")]
    Query(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageLog {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    pub request_id: String,
    pub timestamp: i64,
    pub model: String,
    pub requested_model: String,
    pub upstream_model: Option<String>,
    pub provider: String,
    pub target_id: String,
    pub source_format: String,
    pub stream: bool,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cache_read_tokens: Option<i64>,
    pub cache_write_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub usage_estimated: bool,
    pub input_cost: String,
    pub output_cost: String,
    pub cache_read_cost: String,
    pub cache_write_cost: String,
    pub reasoning_cost: String,
    pub total_cost: String,
    pub rate_multiplier: String,
    pub adjusted_cost: String,
    pub service_tier: Option<String>,
    pub track: Option<String>,
    pub duration_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub attempt_count: u32,
    pub success: bool,
    pub finish_reason: Option<String>,
    pub error_kind: Option<String>,
    pub error_message: Option<String>,
    pub http_status: Option<u16>,
    pub client_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTrace {
    pub stream_id: u64,
    pub timestamp: i64,
    pub model: String,
    pub provider: String,
    pub duration_ms: Option<f64>,
    pub success: bool,
    pub hooks_fired: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub timestamp: i64,
    pub alert_type: String,
    pub message: String,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub value: Option<f64>,
    pub threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealCacheEntry {
    pub model: String,
    pub provider: String,
    pub param: String,
    pub removed_at: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedStats {
    pub total_requests: u64,
    pub total_tokens_in: i64,
    pub total_tokens_out: i64,
    pub total_cost_usd: String,
    pub avg_duration_ms: f64,
    pub error_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatsPeriod {
    Hourly,
    Daily,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsAggregate {
    pub period_type: StatsPeriod,
    pub period_start: i64,
    pub stats: AggregatedStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingSnapshot {
    pub fetched_at: i64,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageQuery {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub from: Option<i64>,
    #[serde(default)]
    pub to: Option<i64>,
    #[serde(default = "default_query_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_query_limit() -> usize {
    100
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatsQuery {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub from: Option<i64>,
    #[serde(default)]
    pub to: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceQuery {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub from: Option<i64>,
    #[serde(default)]
    pub to: Option<i64>,
    #[serde(default = "default_query_limit")]
    pub limit: usize,
}

#[async_trait]
pub trait Store: Send + Sync {
    async fn record_usage(&self, log: UsageLog) -> Result<(), StoreError>;
    async fn record_usage_batch(&self, batch: Vec<UsageLog>) -> Result<(), StoreError> {
        for log in batch {
            self.record_usage(log).await?;
        }
        Ok(())
    }
    async fn query_usage(&self, query: UsageQuery) -> Result<Vec<UsageLog>, StoreError>;

    async fn get_stats(&self, query: StatsQuery) -> Result<AggregatedStats, StoreError> {
        let _ = query;
        Ok(AggregatedStats::default())
    }

    async fn record_trace(&self, _record: RequestTrace) -> Result<(), StoreError> {
        Ok(())
    }
    async fn query_traces(&self, _query: TraceQuery) -> Result<Vec<RequestTrace>, StoreError> {
        Ok(vec![])
    }

    async fn record_alert(&self, _alert: AlertEvent) -> Result<(), StoreError> {
        Ok(())
    }
    async fn query_alerts(
        &self,
        _limit: usize,
    ) -> Result<Vec<AlertEvent>, StoreError> {
        Ok(vec![])
    }

    async fn load_heal_cache(&self) -> Result<Vec<HealCacheEntry>, StoreError> {
        Ok(vec![])
    }
    async fn save_heal_entry(&self, _entry: HealCacheEntry) -> Result<(), StoreError> {
        Ok(())
    }

    async fn load_pricing(&self) -> Result<Option<PricingSnapshot>, StoreError> {
        Ok(None)
    }

    async fn save_pricing(&self, _snapshot: PricingSnapshot) -> Result<(), StoreError> {
        Ok(())
    }

    async fn save_stats_aggregate(&self, _agg: StatsAggregate) -> Result<(), StoreError> {
        Ok(())
    }

    async fn query_stats_aggregates(
        &self,
        _period: StatsPeriod,
        _since: i64,
        _limit: usize,
    ) -> Result<Vec<StatsAggregate>, StoreError> {
        Ok(vec![])
    }

    async fn get_stats_for_period(
        &self,
        _from: i64,
        _to: i64,
        _model: Option<&str>,
    ) -> Result<AggregatedStats, StoreError> {
        Ok(AggregatedStats::default())
    }
}

// ---------------------------------------------------------------------------
// UsageWriter: async batched write queue (§11 design doc)
// ---------------------------------------------------------------------------

pub struct UsageWriter {
    tx: std::sync::Mutex<Option<mpsc::Sender<UsageLog>>>,
    handle: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
    store: Arc<dyn Store>,
}

impl UsageWriter {
    pub fn new(store: Arc<dyn Store>, batch_size: usize, flush_interval: Duration) -> Self {
        let rt = tokio::runtime::Handle::try_current();
        if rt.is_err() {
            return Self {
                tx: std::sync::Mutex::new(None),
                handle: std::sync::Mutex::new(None),
                store,
            };
        }

        let (tx, mut rx) = mpsc::channel::<UsageLog>(16384);
        let flush_store = store.clone();
        let jh = tokio::spawn(async move {
            let mut buffer = Vec::with_capacity(batch_size);
            let mut ticker = tokio::time::interval(flush_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    msg = rx.recv() => {
                        match msg {
                            Some(log) => {
                                buffer.push(log);
                                if buffer.len() >= batch_size {
                                    let batch = std::mem::replace(
                                        &mut buffer,
                                        Vec::with_capacity(batch_size),
                                    );
                                    flush_batch(&flush_store, batch).await;
                                }
                            }
                            None => {
                                if !buffer.is_empty() {
                                    let batch = std::mem::take(&mut buffer);
                                    flush_batch(&flush_store, batch).await;
                                }
                                break;
                            }
                        }
                    }
                    _ = ticker.tick() => {
                        if !buffer.is_empty() {
                            let batch = std::mem::replace(
                                &mut buffer,
                                Vec::with_capacity(batch_size),
                            );
                            flush_batch(&flush_store, batch).await;
                        }
                    }
                }
            }
        });
        Self {
            tx: std::sync::Mutex::new(Some(tx)),
            handle: std::sync::Mutex::new(Some(jh)),
            store,
        }
    }

    pub fn try_send(&self, log: UsageLog) -> Result<(), StoreError> {
        let guard = self.tx.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(tx) => tx
                .try_send(log)
                .map_err(|_| StoreError::Io("usage writer channel full or closed".into())),
            None => {
                let store = self.store.clone();
                tokio::spawn(async move {
                    if let Err(e) = store.record_usage(log).await {
                        tracing::warn!(error = %e, "usage fallback write failed");
                    }
                });
                Ok(())
            }
        }
    }

    pub async fn flush(&self) {
        let tx = self
            .tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        drop(tx);
        let jh = self
            .handle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(h) = jh {
            let _ = h.await;
        }
    }
}

async fn flush_batch(store: &Arc<dyn Store>, batch: Vec<UsageLog>) {
    if let Err(e) = store.record_usage_batch(batch).await {
        tracing::warn!(error = %e, "usage writer: failed to flush batch");
    }
}
