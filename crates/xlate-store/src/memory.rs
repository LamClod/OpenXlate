use async_trait::async_trait;
use std::sync::Mutex;
use xlate_core::store::{
    AggregatedStats, AlertEvent, HealCacheEntry, PricingSnapshot, StatsAggregate, StatsPeriod, Store,
    StoreError, StatsQuery, TraceQuery, RequestTrace, UsageLog, UsageQuery,
};

pub struct MemoryStore {
    logs: Mutex<Vec<UsageLog>>,
    traces: Mutex<Vec<RequestTrace>>,
    alerts: Mutex<Vec<AlertEvent>>,
    heal_cache: Mutex<Vec<HealCacheEntry>>,
    stats_aggregates: Mutex<Vec<StatsAggregate>>,
    pricing: Mutex<Option<PricingSnapshot>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            logs: Mutex::new(Vec::new()),
            traces: Mutex::new(Vec::new()),
            alerts: Mutex::new(Vec::new()),
            heal_cache: Mutex::new(Vec::new()),
            stats_aggregates: Mutex::new(Vec::new()),
            pricing: Mutex::new(None),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

fn matches_usage_query(log: &UsageLog, query: &UsageQuery) -> bool {
    if let Some(ref m) = query.model {
        if &log.model != m {
            return false;
        }
    }
    if let Some(ref p) = query.provider {
        if &log.provider != p {
            return false;
        }
    }
    if let Some(from) = query.from {
        if log.timestamp < from {
            return false;
        }
    }
    if let Some(to) = query.to {
        if log.timestamp >= to {
            return false;
        }
    }
    true
}

fn matches_trace_query(trace: &RequestTrace, query: &TraceQuery) -> bool {
    if let Some(ref m) = query.model {
        if &trace.model != m {
            return false;
        }
    }
    if let Some(from) = query.from {
        if trace.timestamp < from {
            return false;
        }
    }
    if let Some(to) = query.to {
        if trace.timestamp >= to {
            return false;
        }
    }
    true
}

#[async_trait]
impl Store for MemoryStore {
    async fn record_usage(&self, log: UsageLog) -> Result<(), StoreError> {
        self.logs
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?
            .push(log);
        Ok(())
    }

    async fn query_usage(&self, query: UsageQuery) -> Result<Vec<UsageLog>, StoreError> {
        let logs = self
            .logs
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?;
        let filtered: Vec<_> = logs
            .iter()
            .rev()
            .filter(|l| matches_usage_query(l, &query))
            .skip(query.offset)
            .take(query.limit)
            .cloned()
            .collect();
        Ok(filtered)
    }

    async fn record_trace(&self, record: RequestTrace) -> Result<(), StoreError> {
        self.traces
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?
            .push(record);
        Ok(())
    }

    async fn query_traces(&self, query: TraceQuery) -> Result<Vec<RequestTrace>, StoreError> {
        let traces = self
            .traces
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(traces
            .iter()
            .rev()
            .filter(|t| matches_trace_query(t, &query))
            .take(query.limit)
            .cloned()
            .collect())
    }

    async fn record_alert(&self, alert: AlertEvent) -> Result<(), StoreError> {
        self.alerts
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?
            .push(alert);
        Ok(())
    }

    async fn query_alerts(&self, limit: usize) -> Result<Vec<AlertEvent>, StoreError> {
        let alerts = self
            .alerts
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(alerts.iter().rev().take(limit).cloned().collect())
    }

    async fn load_heal_cache(&self) -> Result<Vec<HealCacheEntry>, StoreError> {
        let cache = self
            .heal_cache
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(cache.clone())
    }

    async fn save_heal_entry(&self, entry: HealCacheEntry) -> Result<(), StoreError> {
        self.heal_cache
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?
            .push(entry);
        Ok(())
    }

    async fn load_pricing(&self) -> Result<Option<PricingSnapshot>, StoreError> {
        let p = self
            .pricing
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(p.clone())
    }

    async fn save_pricing(&self, snapshot: PricingSnapshot) -> Result<(), StoreError> {
        let mut p = self
            .pricing
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?;
        *p = Some(snapshot);
        Ok(())
    }

    async fn get_stats(&self, query: StatsQuery) -> Result<AggregatedStats, StoreError> {
        let logs = self
            .logs
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?;
        let mut stats = AggregatedStats::default();
        let mut total_cost: f64 = 0.0;
        let mut total_dur: f64 = 0.0;
        let mut dur_count: u64 = 0;

        for log in logs.iter() {
            if let Some(ref m) = query.model {
                if &log.model != m {
                    continue;
                }
            }
            if let Some(from) = query.from {
                if log.timestamp < from {
                    continue;
                }
            }
            if let Some(to) = query.to {
                if log.timestamp >= to {
                    continue;
                }
            }
            stats.total_requests += 1;
            stats.total_tokens_in += log.input_tokens.unwrap_or(0);
            stats.total_tokens_out += log.output_tokens.unwrap_or(0);
            if !log.success {
                stats.error_count += 1;
            }
            if let Ok(c) = log.total_cost.parse::<f64>() {
                total_cost += c;
            }
            if let Some(d) = log.duration_ms {
                total_dur += d;
                dur_count += 1;
            }
        }

        stats.total_cost_usd = format!("{total_cost:.6}");
        if dur_count > 0 {
            stats.avg_duration_ms = total_dur / dur_count as f64;
        }

        Ok(stats)
    }

    async fn save_stats_aggregate(&self, agg: StatsAggregate) -> Result<(), StoreError> {
        self.stats_aggregates
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?
            .push(agg);
        Ok(())
    }

    async fn query_stats_aggregates(
        &self,
        period: StatsPeriod,
        since: i64,
        limit: usize,
    ) -> Result<Vec<StatsAggregate>, StoreError> {
        let aggs = self
            .stats_aggregates
            .lock()
            .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(aggs
            .iter()
            .rev()
            .filter(|a| a.period_type == period && a.period_start >= since)
            .take(limit)
            .cloned()
            .collect())
    }

    async fn get_stats_for_period(
        &self,
        from: i64,
        to: i64,
        model: Option<&str>,
    ) -> Result<AggregatedStats, StoreError> {
        self.get_stats(StatsQuery {
            model: model.map(String::from),
            from: Some(from),
            to: Some(to),
        })
        .await
    }
}
