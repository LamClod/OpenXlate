use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Mutex;

use dashmap::DashMap;
use xlate_core::hook::{Hook, HookContext, HookVerdict};

const DEFAULT_WINDOW: usize = 1000;

struct CacheEntry {
    input_tokens: i64,
    cache_read_tokens: i64,
}

struct RollingStats {
    entries: VecDeque<CacheEntry>,
    window_size: usize,
}

impl RollingStats {
    fn new(window_size: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(window_size.min(256)),
            window_size,
        }
    }

    fn record(&mut self, input: i64, cached: i64) {
        if self.entries.len() >= self.window_size {
            self.entries.pop_front();
        }
        self.entries.push_back(CacheEntry {
            input_tokens: input,
            cache_read_tokens: cached,
        });
    }

    fn hit_rate(&self) -> f64 {
        let mut total_cache: i64 = 0;
        let mut total_input: i64 = 0;
        for e in &self.entries {
            total_cache += e.cache_read_tokens;
            total_input += e.input_tokens;
        }
        let denom = total_input + total_cache;
        if denom == 0 {
            0.0
        } else {
            total_cache as f64 / denom as f64
        }
    }

    fn request_count(&self) -> u64 {
        self.entries.len() as u64
    }
}

pub struct CacheStatsHook {
    per_model: DashMap<String, Mutex<RollingStats>>,
    per_target: DashMap<String, Mutex<RollingStats>>,
    window_size: usize,
}

impl CacheStatsHook {
    pub fn new() -> Self {
        Self::with_window(DEFAULT_WINDOW)
    }

    pub fn with_window(window_size: usize) -> Self {
        Self {
            per_model: DashMap::new(),
            per_target: DashMap::new(),
            window_size,
        }
    }

    fn hit_rate_from(map: &DashMap<String, Mutex<RollingStats>>) -> f64 {
        let mut total_cache: i64 = 0;
        let mut total_input: i64 = 0;
        for entry in map.iter() {
            if let Ok(stats) = entry.value().lock() {
                for e in &stats.entries {
                    total_cache += e.cache_read_tokens;
                    total_input += e.input_tokens;
                }
            }
        }
        let denom = total_input + total_cache;
        if denom == 0 {
            0.0
        } else {
            total_cache as f64 / denom as f64
        }
    }

    fn hit_rate_for(map: &DashMap<String, Mutex<RollingStats>>, key: &str) -> f64 {
        map.get(key)
            .and_then(|entry| entry.value().lock().ok().map(|s| s.hit_rate()))
            .unwrap_or(0.0)
    }

    pub fn hit_rate(&self) -> f64 {
        Self::hit_rate_from(&self.per_model)
    }

    pub fn hit_rate_for_model(&self, model: &str) -> f64 {
        Self::hit_rate_for(&self.per_model, model)
    }

    pub fn hit_rate_for_target(&self, target_id: &str) -> f64 {
        Self::hit_rate_for(&self.per_target, target_id)
    }

    pub fn total_requests(&self) -> u64 {
        self.per_model
            .iter()
            .filter_map(|e| e.value().lock().ok().map(|s| s.request_count()))
            .sum()
    }

    fn record(
        map: &DashMap<String, Mutex<RollingStats>>,
        key: String,
        input: Option<i64>,
        cached: Option<i64>,
        window_size: usize,
    ) {
        map.entry(key)
            .or_insert_with(|| Mutex::new(RollingStats::new(window_size)))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record(input.unwrap_or(0), cached.unwrap_or(0));
    }
}

impl Default for CacheStatsHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for CacheStatsHook {
    fn name(&self) -> &str {
        "cache-stats"
    }

    fn priority(&self) -> i32 {
        18
    }

    fn snapshot(&self) -> Option<serde_json::Value> {
        let mut per_model = serde_json::Map::new();
        for entry in self.per_model.iter() {
            if let Ok(stats) = entry.value().lock() {
                per_model.insert(
                    entry.key().clone(),
                    serde_json::json!({
                        "hit_rate": stats.hit_rate(),
                        "request_count": stats.request_count(),
                    }),
                );
            }
        }
        Some(serde_json::json!({
            "overall_hit_rate": self.hit_rate(),
            "total_requests": self.total_requests(),
            "window_size": self.window_size,
            "per_model": per_model,
        }))
    }

    async fn post_complete(&self, ctx: &mut HookContext) -> HookVerdict {
        let input = ctx.metrics.usage.input_tokens;
        let cached = ctx.metrics.usage.cache_read_tokens;

        Self::record(
            &self.per_model,
            ctx.request.model.clone(),
            input,
            cached,
            self.window_size,
        );

        let target_id = ctx
            .route
            .as_ref()
            .map(|r| r.target.id.clone())
            .unwrap_or_default();
        if !target_id.is_empty() {
            Self::record(&self.per_target, target_id, input, cached, self.window_size);
        }

        HookVerdict::Continue
    }
}
