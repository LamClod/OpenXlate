use async_trait::async_trait;
use rusqlite::Connection;
use std::sync::Mutex;
use xlate_core::store::{
    AggregatedStats, AlertEvent, HealCacheEntry, PricingSnapshot, StatsAggregate, StatsQuery,
    StatsPeriod, Store, StoreError, TraceQuery, RequestTrace, UsageLog, UsageQuery,
};

pub struct SqliteStore {
    conn: Mutex<Connection>,
}

impl SqliteStore {
    pub fn new(path: &str) -> Result<Self, StoreError> {
        let expanded = shellexpand(path);
        if let Some(parent) = std::path::Path::new(&expanded).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn =
            Connection::open(&expanded).map_err(|e| StoreError::Io(e.to_string()))?;
        conn.execute_batch(SCHEMA)
            .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

fn shellexpand(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs_home() {
            return format!("{}/{}", home, rest);
        }
    }
    path.to_string()
}

fn dirs_home() -> Option<String> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS usage_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    timestamp       INTEGER NOT NULL,
    model           TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    upstream_model  TEXT,
    provider        TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    source_format   TEXT NOT NULL,
    stream          INTEGER NOT NULL DEFAULT 1,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cache_read_tokens  INTEGER,
    cache_write_tokens INTEGER,
    reasoning_tokens   INTEGER,
    total_tokens    INTEGER,
    usage_estimated INTEGER NOT NULL DEFAULT 0,
    input_cost      TEXT NOT NULL DEFAULT '0',
    output_cost     TEXT NOT NULL DEFAULT '0',
    cache_read_cost TEXT NOT NULL DEFAULT '0',
    cache_write_cost TEXT NOT NULL DEFAULT '0',
    reasoning_cost  TEXT NOT NULL DEFAULT '0',
    total_cost      TEXT NOT NULL DEFAULT '0',
    rate_multiplier TEXT NOT NULL DEFAULT '1',
    adjusted_cost   TEXT NOT NULL DEFAULT '0',
    service_tier    TEXT,
    track           TEXT,
    duration_ms     REAL,
    ttft_ms         REAL,
    attempt_count   INTEGER NOT NULL DEFAULT 1,
    success         INTEGER NOT NULL DEFAULT 1,
    finish_reason   TEXT,
    error_kind      TEXT,
    error_message   TEXT,
    http_status     INTEGER,
    client_id       TEXT
);

CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_logs(model);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_logs(timestamp);

CREATE TABLE IF NOT EXISTS traces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id   INTEGER NOT NULL,
    timestamp   INTEGER NOT NULL,
    model       TEXT NOT NULL,
    provider    TEXT NOT NULL,
    duration_ms REAL,
    success     INTEGER NOT NULL DEFAULT 1,
    hooks_fired TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   INTEGER NOT NULL,
    alert_type  TEXT NOT NULL,
    message     TEXT NOT NULL,
    model       TEXT,
    provider    TEXT,
    value       REAL,
    threshold   REAL
);

CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(timestamp);

CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);

CREATE TABLE IF NOT EXISTS heal_cache (
    model       TEXT NOT NULL,
    provider    TEXT NOT NULL,
    param       TEXT NOT NULL,
    removed_at  INTEGER NOT NULL,
    PRIMARY KEY (model, provider, param)
);

CREATE TABLE IF NOT EXISTS pricing_cache (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    fetched_at  INTEGER NOT NULL,
    data        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stats_aggregates (
    period_type  TEXT NOT NULL,
    period_start INTEGER NOT NULL,
    total_requests  INTEGER NOT NULL DEFAULT 0,
    total_tokens_in INTEGER NOT NULL DEFAULT 0,
    total_tokens_out INTEGER NOT NULL DEFAULT 0,
    total_cost_usd  TEXT NOT NULL DEFAULT '0',
    avg_duration_ms REAL NOT NULL DEFAULT 0,
    error_count     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (period_type, period_start)
);
"#;

#[async_trait]
impl Store for SqliteStore {
    async fn record_usage(&self, log: UsageLog) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        conn.execute(
            "INSERT INTO usage_logs (
                request_id, timestamp, model, requested_model, upstream_model,
                provider, target_id, source_format, stream,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                reasoning_tokens, total_tokens, usage_estimated,
                input_cost, output_cost, cache_read_cost, cache_write_cost,
                reasoning_cost, total_cost, rate_multiplier, adjusted_cost,
                service_tier, track, duration_ms, ttft_ms, attempt_count,
                success, finish_reason, error_kind, error_message, http_status, client_id
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24,
                ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35
            )",
            rusqlite::params![
                log.request_id,
                log.timestamp,
                log.model,
                log.requested_model,
                log.upstream_model,
                log.provider,
                log.target_id,
                log.source_format,
                log.stream as i32,
                log.input_tokens,
                log.output_tokens,
                log.cache_read_tokens,
                log.cache_write_tokens,
                log.reasoning_tokens,
                log.total_tokens,
                log.usage_estimated as i32,
                log.input_cost,
                log.output_cost,
                log.cache_read_cost,
                log.cache_write_cost,
                log.reasoning_cost,
                log.total_cost,
                log.rate_multiplier,
                log.adjusted_cost,
                log.service_tier,
                log.track,
                log.duration_ms,
                log.ttft_ms,
                log.attempt_count,
                log.success as i32,
                log.finish_reason,
                log.error_kind,
                log.error_message,
                log.http_status,
                log.client_id,
            ],
        )
        .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(())
    }

    async fn record_usage_batch(&self, batch: Vec<UsageLog>) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        conn.execute_batch("BEGIN").map_err(|e| StoreError::Io(e.to_string()))?;
        for log in &batch {
            if let Err(e) = conn.execute(
                "INSERT INTO usage_logs (
                    request_id, timestamp, model, requested_model, upstream_model,
                    provider, target_id, source_format, stream,
                    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                    reasoning_tokens, total_tokens, usage_estimated,
                    input_cost, output_cost, cache_read_cost, cache_write_cost,
                    reasoning_cost, total_cost, rate_multiplier, adjusted_cost,
                    service_tier, track, duration_ms, ttft_ms, attempt_count,
                    success, finish_reason, error_kind, error_message, http_status, client_id
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                    ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                    ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24,
                    ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35
                )",
                rusqlite::params![
                    log.request_id, log.timestamp, log.model, log.requested_model,
                    log.upstream_model, log.provider, log.target_id, log.source_format,
                    log.stream as i32, log.input_tokens, log.output_tokens,
                    log.cache_read_tokens, log.cache_write_tokens, log.reasoning_tokens,
                    log.total_tokens, log.usage_estimated as i32,
                    log.input_cost, log.output_cost, log.cache_read_cost, log.cache_write_cost,
                    log.reasoning_cost, log.total_cost, log.rate_multiplier, log.adjusted_cost,
                    log.service_tier, log.track, log.duration_ms, log.ttft_ms,
                    log.attempt_count, log.success as i32, log.finish_reason,
                    log.error_kind, log.error_message, log.http_status, log.client_id,
                ],
            ) {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(StoreError::Io(e.to_string()));
            }
        }
        conn.execute_batch("COMMIT").map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(())
    }

    async fn query_usage(&self, query: UsageQuery) -> Result<Vec<UsageLog>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;

        let mut conditions = Vec::new();
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        if let Some(ref m) = query.model {
            conditions.push(format!("model = ?{idx}"));
            params.push(Box::new(m.clone()));
            idx += 1;
        }
        if let Some(ref p) = query.provider {
            conditions.push(format!("provider = ?{idx}"));
            params.push(Box::new(p.clone()));
            idx += 1;
        }
        if let Some(from) = query.from {
            conditions.push(format!("timestamp >= ?{idx}"));
            params.push(Box::new(from));
            idx += 1;
        }
        if let Some(to) = query.to {
            conditions.push(format!("timestamp < ?{idx}"));
            params.push(Box::new(to));
            idx += 1;
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT id, request_id, timestamp, model, requested_model, upstream_model,
             provider, target_id, source_format, stream,
             input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
             reasoning_tokens, total_tokens, usage_estimated,
             input_cost, output_cost, cache_read_cost, cache_write_cost,
             reasoning_cost, total_cost, rate_multiplier, adjusted_cost,
             service_tier, track, duration_ms, ttft_ms, attempt_count,
             success, finish_reason, error_kind, error_message, http_status, client_id
             FROM usage_logs{where_clause} ORDER BY id DESC LIMIT ?{idx} OFFSET ?{}",
            idx + 1
        );
        params.push(Box::new(query.limit as i64));
        params.push(Box::new(query.offset as i64));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql).map_err(|e| StoreError::Query(e.to_string()))?;
        let rows = stmt
            .query_map(params_refs.as_slice(), |row| {
                Ok(UsageLog {
                    id: row.get(0)?,
                    request_id: row.get(1)?,
                    timestamp: row.get(2)?,
                    model: row.get(3)?,
                    requested_model: row.get(4)?,
                    upstream_model: row.get(5)?,
                    provider: row.get(6)?,
                    target_id: row.get(7)?,
                    source_format: row.get(8)?,
                    stream: row.get::<_, i32>(9)? != 0,
                    input_tokens: row.get(10)?,
                    output_tokens: row.get(11)?,
                    cache_read_tokens: row.get(12)?,
                    cache_write_tokens: row.get(13)?,
                    reasoning_tokens: row.get(14)?,
                    total_tokens: row.get(15)?,
                    usage_estimated: row.get::<_, i32>(16)? != 0,
                    input_cost: row.get(17)?,
                    output_cost: row.get(18)?,
                    cache_read_cost: row.get(19)?,
                    cache_write_cost: row.get(20)?,
                    reasoning_cost: row.get(21)?,
                    total_cost: row.get(22)?,
                    rate_multiplier: row.get(23)?,
                    adjusted_cost: row.get(24)?,
                    service_tier: row.get(25)?,
                    track: row.get(26)?,
                    duration_ms: row.get(27)?,
                    ttft_ms: row.get(28)?,
                    attempt_count: row.get(29)?,
                    success: row.get::<_, i32>(30)? != 0,
                    finish_reason: row.get(31)?,
                    error_kind: row.get(32)?,
                    error_message: row.get(33)?,
                    http_status: row.get::<_, Option<i32>>(34)?.map(|v| v as u16),
                    client_id: row.get(35)?,
                })
            })
            .map_err(|e| StoreError::Query(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| StoreError::Query(e.to_string()))?);
        }
        Ok(result)
    }

    async fn record_trace(&self, record: RequestTrace) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        let hooks_json =
            serde_json::to_string(&record.hooks_fired).unwrap_or_else(|_| "[]".into());
        conn.execute(
            "INSERT INTO traces (stream_id, timestamp, model, provider, duration_ms, success, hooks_fired)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                record.stream_id as i64,
                record.timestamp,
                record.model,
                record.provider,
                record.duration_ms,
                record.success as i32,
                hooks_json,
            ],
        )
        .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(())
    }

    async fn query_traces(&self, query: TraceQuery) -> Result<Vec<RequestTrace>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;

        let mut conditions = Vec::new();
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        if let Some(ref m) = query.model {
            conditions.push(format!("model = ?{idx}"));
            params.push(Box::new(m.clone()));
            idx += 1;
        }
        if let Some(from) = query.from {
            conditions.push(format!("timestamp >= ?{idx}"));
            params.push(Box::new(from));
            idx += 1;
        }
        if let Some(to) = query.to {
            conditions.push(format!("timestamp < ?{idx}"));
            params.push(Box::new(to));
            idx += 1;
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT stream_id, timestamp, model, provider, duration_ms, success, hooks_fired
             FROM traces{where_clause} ORDER BY id DESC LIMIT ?{idx}"
        );
        params.push(Box::new(query.limit as i64));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql).map_err(|e| StoreError::Query(e.to_string()))?;
        let rows = stmt
            .query_map(params_refs.as_slice(), |row| {
                let hooks_str: String = row.get(6)?;
                let hooks: Vec<String> =
                    serde_json::from_str(&hooks_str).unwrap_or_default();
                Ok(RequestTrace {
                    stream_id: row.get::<_, i64>(0)? as u64,
                    timestamp: row.get(1)?,
                    model: row.get(2)?,
                    provider: row.get(3)?,
                    duration_ms: row.get(4)?,
                    success: row.get::<_, i32>(5)? != 0,
                    hooks_fired: hooks,
                })
            })
            .map_err(|e| StoreError::Query(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| StoreError::Query(e.to_string()))?);
        }
        Ok(result)
    }

    async fn record_alert(&self, alert: AlertEvent) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        conn.execute(
            "INSERT INTO alerts (timestamp, alert_type, message, model, provider, value, threshold)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                alert.timestamp,
                alert.alert_type,
                alert.message,
                alert.model,
                alert.provider,
                alert.value,
                alert.threshold,
            ],
        )
        .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(())
    }

    async fn query_alerts(&self, limit: usize) -> Result<Vec<AlertEvent>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT timestamp, alert_type, message, model, provider, value, threshold
                 FROM alerts ORDER BY id DESC LIMIT ?1",
            )
            .map_err(|e| StoreError::Query(e.to_string()))?;
        let rows = stmt
            .query_map(rusqlite::params![limit as i64], |row| {
                Ok(AlertEvent {
                    timestamp: row.get(0)?,
                    alert_type: row.get(1)?,
                    message: row.get(2)?,
                    model: row.get(3)?,
                    provider: row.get(4)?,
                    value: row.get(5)?,
                    threshold: row.get(6)?,
                })
            })
            .map_err(|e| StoreError::Query(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| StoreError::Query(e.to_string()))?);
        }
        Ok(result)
    }

    async fn load_heal_cache(&self) -> Result<Vec<HealCacheEntry>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT model, provider, param, removed_at FROM heal_cache")
            .map_err(|e| StoreError::Query(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(HealCacheEntry {
                    model: row.get(0)?,
                    provider: row.get(1)?,
                    param: row.get(2)?,
                    removed_at: row.get(3)?,
                })
            })
            .map_err(|e| StoreError::Query(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| StoreError::Query(e.to_string()))?);
        }
        Ok(result)
    }

    async fn save_heal_entry(&self, entry: HealCacheEntry) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO heal_cache (model, provider, param, removed_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![entry.model, entry.provider, entry.param, entry.removed_at],
        )
        .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(())
    }

    async fn get_stats(&self, query: StatsQuery) -> Result<AggregatedStats, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        let model = query.model.as_deref();
        let (sql, params): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(m) = model {
            (
                "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                 COALESCE(SUM(CAST(total_cost AS REAL)),0), COALESCE(AVG(duration_ms),0),
                 SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)
                 FROM usage_logs WHERE model = ?1",
                vec![Box::new(m.to_string()) as Box<dyn rusqlite::types::ToSql>],
            )
        } else {
            (
                "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                 COALESCE(SUM(CAST(total_cost AS REAL)),0), COALESCE(AVG(duration_ms),0),
                 SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)
                 FROM usage_logs",
                vec![],
            )
        };
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        conn.query_row(sql, params_refs.as_slice(), |row| {
            Ok(AggregatedStats {
                total_requests: row.get::<_, i64>(0)? as u64,
                total_tokens_in: row.get(1)?,
                total_tokens_out: row.get(2)?,
                total_cost_usd: format!("{:.6}", row.get::<_, f64>(3)?),
                avg_duration_ms: row.get(4)?,
                error_count: row.get::<_, i64>(5)? as u64,
            })
        })
        .map_err(|e| StoreError::Query(e.to_string()))
    }

    async fn load_pricing(&self) -> Result<Option<PricingSnapshot>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        let result = conn.query_row(
            "SELECT fetched_at, data FROM pricing_cache WHERE id = 1",
            [],
            |row| {
                let data_str: String = row.get(1)?;
                let data: serde_json::Value =
                    serde_json::from_str(&data_str).unwrap_or(serde_json::Value::Null);
                Ok(PricingSnapshot {
                    fetched_at: row.get(0)?,
                    data,
                })
            },
        );
        match result {
            Ok(snap) => Ok(Some(snap)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StoreError::Query(e.to_string())),
        }
    }

    async fn save_pricing(&self, snapshot: PricingSnapshot) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        let data_str = serde_json::to_string(&snapshot.data)
            .map_err(|e| StoreError::Io(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO pricing_cache (id, fetched_at, data) VALUES (1, ?1, ?2)",
            rusqlite::params![snapshot.fetched_at, data_str],
        )
        .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(())
    }

    async fn save_stats_aggregate(&self, agg: StatsAggregate) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        let period_str = match agg.period_type {
            StatsPeriod::Hourly => "hourly",
            StatsPeriod::Daily => "daily",
        };
        conn.execute(
            "INSERT OR REPLACE INTO stats_aggregates
             (period_type, period_start, total_requests, total_tokens_in, total_tokens_out,
              total_cost_usd, avg_duration_ms, error_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                period_str,
                agg.period_start,
                agg.stats.total_requests as i64,
                agg.stats.total_tokens_in,
                agg.stats.total_tokens_out,
                agg.stats.total_cost_usd,
                agg.stats.avg_duration_ms,
                agg.stats.error_count as i64,
            ],
        )
        .map_err(|e| StoreError::Io(e.to_string()))?;
        Ok(())
    }

    async fn query_stats_aggregates(
        &self,
        period: StatsPeriod,
        since: i64,
        limit: usize,
    ) -> Result<Vec<StatsAggregate>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        let period_str = match period {
            StatsPeriod::Hourly => "hourly",
            StatsPeriod::Daily => "daily",
        };
        let mut stmt = conn
            .prepare(
                "SELECT period_start, total_requests, total_tokens_in, total_tokens_out,
                 total_cost_usd, avg_duration_ms, error_count
                 FROM stats_aggregates
                 WHERE period_type = ?1 AND period_start >= ?2
                 ORDER BY period_start DESC LIMIT ?3",
            )
            .map_err(|e| StoreError::Query(e.to_string()))?;
        let rows = stmt
            .query_map(
                rusqlite::params![period_str, since, limit as i64],
                |row| {
                    Ok(StatsAggregate {
                        period_type: period,
                        period_start: row.get(0)?,
                        stats: AggregatedStats {
                            total_requests: row.get::<_, i64>(1)? as u64,
                            total_tokens_in: row.get(2)?,
                            total_tokens_out: row.get(3)?,
                            total_cost_usd: row.get(4)?,
                            avg_duration_ms: row.get(5)?,
                            error_count: row.get::<_, i64>(6)? as u64,
                        },
                    })
                },
            )
            .map_err(|e| StoreError::Query(e.to_string()))?;
        let mut result = Vec::new();
        for r in rows {
            result.push(r.map_err(|e| StoreError::Query(e.to_string()))?);
        }
        Ok(result)
    }

    async fn get_stats_for_period(
        &self,
        from: i64,
        to: i64,
        model: Option<&str>,
    ) -> Result<AggregatedStats, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Io(e.to_string()))?;
        let (sql, params): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(m) = model {
            (
                "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                 COALESCE(SUM(CAST(total_cost AS REAL)),0), COALESCE(AVG(duration_ms),0),
                 SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)
                 FROM usage_logs WHERE timestamp >= ?1 AND timestamp < ?2 AND model = ?3",
                vec![
                    Box::new(from) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(to) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(m.to_string()) as Box<dyn rusqlite::types::ToSql>,
                ],
            )
        } else {
            (
                "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                 COALESCE(SUM(CAST(total_cost AS REAL)),0), COALESCE(AVG(duration_ms),0),
                 SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)
                 FROM usage_logs WHERE timestamp >= ?1 AND timestamp < ?2",
                vec![
                    Box::new(from) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(to) as Box<dyn rusqlite::types::ToSql>,
                ],
            )
        };
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        conn.query_row(sql, params_refs.as_slice(), |row| {
            Ok(AggregatedStats {
                total_requests: row.get::<_, i64>(0)? as u64,
                total_tokens_in: row.get(1)?,
                total_tokens_out: row.get(2)?,
                total_cost_usd: format!("{:.6}", row.get::<_, f64>(3)?),
                avg_duration_ms: row.get(4)?,
                error_count: row.get::<_, i64>(5)? as u64,
            })
        })
        .map_err(|e| StoreError::Query(e.to_string()))
    }
}
