use dashmap::DashMap;
use std::time::{Duration, Instant};

use crate::capability::PluginCapabilities;
use crate::plugin::{PluginId, PluginKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginStatus {
    Starting,
    Running,
    Degraded { reason: String },
    Stopped,
    Crashed { error: String },
    Restarting { attempt: u32 },
}

pub struct PluginState {
    pub id: PluginId,
    pub kind: PluginKind,
    pub status: PluginStatus,
    pub last_heartbeat: Instant,
    pub restart_count: u32,
    pub capabilities: PluginCapabilities,
    pub last_crash: Option<Instant>,
    pub backoff: Duration,
}

pub struct Supervisor {
    plugins: DashMap<String, PluginState>,
    max_backoff: Duration,
    heartbeat_interval: Duration,
    max_missed_heartbeats: u32,
}

impl Supervisor {
    pub fn new() -> Self {
        Self {
            plugins: DashMap::new(),
            max_backoff: Duration::from_secs(60),
            heartbeat_interval: Duration::from_secs(30),
            max_missed_heartbeats: 3,
        }
    }

    pub fn register_plugin(&self, name: &str, kind: PluginKind) {
        self.plugins.insert(
            name.to_string(),
            PluginState {
                id: name.to_string(),
                kind,
                status: PluginStatus::Starting,
                last_heartbeat: Instant::now(),
                restart_count: 0,
                capabilities: PluginCapabilities::empty(),
                last_crash: None,
                backoff: Duration::from_secs(1),
            },
        );
    }

    pub fn set_capabilities(&self, name: &str, caps: PluginCapabilities) {
        if let Some(mut entry) = self.plugins.get_mut(name) {
            entry.capabilities = caps;
        }
    }

    pub fn record_heartbeat(&self, name: &str) {
        if let Some(mut entry) = self.plugins.get_mut(name) {
            entry.last_heartbeat = Instant::now();
        }
    }

    pub fn check_heartbeats(&self) {
        for mut entry in self.plugins.iter_mut() {
            let elapsed = entry.last_heartbeat.elapsed();
            let intervals = elapsed.as_secs_f64()
                / self.heartbeat_interval.as_secs_f64();
            let missed = intervals as u32;

            if missed > self.max_missed_heartbeats {
                let name = entry.key().clone();
                entry.restart_count += 1;
                entry.last_crash = Some(Instant::now());
                entry.status = PluginStatus::Crashed {
                    error: format!(
                        "missed {} heartbeats (threshold {})",
                        missed, self.max_missed_heartbeats
                    ),
                };
                entry.backoff = (entry.backoff * 2).min(self.max_backoff);
                tracing::error!(
                    plugin = %name,
                    missed_heartbeats = missed,
                    "plugin marked crashed due to missed heartbeats"
                );
            } else if missed >= 1 && entry.status == PluginStatus::Running {
                entry.status = PluginStatus::Degraded {
                    reason: format!("missed {} heartbeat(s)", missed),
                };
            }
        }
    }

    pub fn report_crash(&self, name: &str, reason: &str) -> u32 {
        let mut entry = self
            .plugins
            .entry(name.to_string())
            .or_insert_with(|| PluginState {
                id: name.to_string(),
                kind: PluginKind::Service,
                status: PluginStatus::Running,
                last_heartbeat: Instant::now(),
                restart_count: 0,
                capabilities: PluginCapabilities::empty(),
                last_crash: None,
                backoff: Duration::from_secs(1),
            });

        entry.restart_count += 1;
        entry.last_crash = Some(Instant::now());
        entry.status = PluginStatus::Crashed {
            error: reason.to_string(),
        };
        entry.backoff = (entry.backoff * 2).min(self.max_backoff);

        tracing::error!(
            plugin = name,
            restart_count = entry.restart_count,
            reason,
            "plugin crashed"
        );
        entry.restart_count
    }

    pub fn is_available(&self, name: &str) -> bool {
        match self.plugins.get(name) {
            None => true,
            Some(state) => match &state.status {
                PluginStatus::Running | PluginStatus::Degraded { .. } => true,
                PluginStatus::Crashed { .. } | PluginStatus::Restarting { .. } => {
                    if let Some(last) = state.last_crash {
                        last.elapsed() >= state.backoff
                    } else {
                        true
                    }
                }
                PluginStatus::Starting | PluginStatus::Stopped => false,
            },
        }
    }

    pub fn mark_running(&self, name: &str) {
        if let Some(mut entry) = self.plugins.get_mut(name) {
            entry.status = PluginStatus::Running;
        }
    }

    pub fn status(&self, name: &str) -> PluginStatus {
        self.plugins
            .get(name)
            .map(|s| s.status.clone())
            .unwrap_or(PluginStatus::Running)
    }

    pub fn plugins_needing_restart(&self) -> Vec<String> {
        self.plugins
            .iter()
            .filter_map(|entry| {
                if let PluginStatus::Crashed { .. } = &entry.status {
                    if let Some(last) = entry.last_crash {
                        if last.elapsed() >= entry.backoff {
                            return Some(entry.key().clone());
                        }
                    }
                }
                None
            })
            .collect()
    }

    pub fn mark_restarting(&self, name: &str) {
        if let Some(mut entry) = self.plugins.get_mut(name) {
            let attempt = entry.restart_count;
            entry.status = PluginStatus::Restarting { attempt };
        }
    }

    pub fn stats(&self) -> Vec<(String, String, u32, PluginKind)> {
        self.plugins
            .iter()
            .map(|e| {
                let status = match &e.status {
                    PluginStatus::Starting => "starting".to_string(),
                    PluginStatus::Running => "running".to_string(),
                    PluginStatus::Degraded { reason } => {
                        format!("degraded: {}", reason)
                    }
                    PluginStatus::Stopped => "stopped".to_string(),
                    PluginStatus::Crashed { error } => {
                        format!("crashed: {}", error)
                    }
                    PluginStatus::Restarting { attempt } => {
                        format!("restarting (attempt {})", attempt)
                    }
                };
                (e.key().clone(), status, e.restart_count, e.kind)
            })
            .collect()
    }
}

impl Default for Supervisor {
    fn default() -> Self {
        Self::new()
    }
}
