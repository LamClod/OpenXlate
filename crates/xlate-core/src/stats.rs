use std::sync::Arc;

use crate::capability::PluginCapabilities;
use crate::error::XlateError;
use crate::store::{StatsPeriod, Store};
use crate::supervisor::PluginStatus;

pub struct StatsAggregator {
    store: Arc<dyn Store>,
    interval_s: u64,
    cancel: tokio::sync::watch::Sender<bool>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
    handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl StatsAggregator {
    pub fn new(store: Arc<dyn Store>, interval_s: u64) -> Self {
        let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
        Self {
            store,
            interval_s,
            cancel,
            cancel_rx,
            handle: tokio::sync::Mutex::new(None),
        }
    }

    async fn aggregate_period(store: &dyn Store, period: StatsPeriod) -> Result<(), XlateError> {
        let now_s = crate::now_ms() / 1000;
        let (period_secs, period_start) = match period {
            StatsPeriod::Hourly => (3600i64, (now_s / 3600) * 3600 - 3600),
            StatsPeriod::Daily => (86400i64, (now_s / 86400) * 86400 - 86400),
        };
        let period_end = period_start + period_secs;

        let from_ms = period_start * 1000;
        let to_ms = period_end * 1000;

        let stats = store
            .get_stats_for_period(from_ms, to_ms, None)
            .await
            .map_err(|e| XlateError::Internal(e.to_string()))?;

        if stats.total_requests == 0 {
            return Ok(());
        }

        let agg = crate::store::StatsAggregate {
            period_type: period,
            period_start,
            stats,
        };
        store
            .save_stats_aggregate(agg)
            .await
            .map_err(|e| XlateError::Internal(e.to_string()))?;

        Ok(())
    }
}

impl crate::plugin::Plugin for StatsAggregator {
    fn manifest(&self) -> crate::plugin::PluginManifest {
        crate::plugin::PluginManifest {
            id: "stats-aggregator".into(),
            name: "stats-aggregator".into(),
            version: "0.1.0".into(),
            kind: crate::plugin::PluginKind::Service,
            required_capabilities: vec![],
        }
    }
}

#[async_trait::async_trait]
impl crate::plugin::ServicePlugin for StatsAggregator {
    fn name(&self) -> &str {
        "stats-aggregator"
    }

    async fn start(&self, _caps: PluginCapabilities) -> Result<(), crate::plugin::PluginError> {
        let mut guard = self.handle.lock().await;
        if guard.is_some() {
            tracing::warn!("stats aggregator already started, ignoring duplicate start");
            return Ok(());
        }

        let interval = std::time::Duration::from_secs(self.interval_s);
        let mut cancel_rx = self.cancel_rx.clone();
        let store = self.store.clone();

        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await;
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        if let Err(e) = StatsAggregator::aggregate_period(&*store, StatsPeriod::Hourly).await {
                            tracing::warn!(error = %e, "hourly stats aggregation failed");
                        }
                        if let Err(e) = StatsAggregator::aggregate_period(&*store, StatsPeriod::Daily).await {
                            tracing::warn!(error = %e, "daily stats aggregation failed");
                        }
                    }
                    _ = cancel_rx.changed() => {
                        tracing::info!("stats aggregator stopping");
                        break;
                    }
                }
            }
        });

        *guard = Some(handle);
        Ok(())
    }

    async fn stop(&self) -> Result<(), crate::plugin::PluginError> {
        let _ = self.cancel.send(true);
        if let Some(handle) = self.handle.lock().await.take() {
            let _ = handle.await;
        }
        Ok(())
    }

    async fn health_check(&self) -> PluginStatus {
        PluginStatus::Running
    }
}
