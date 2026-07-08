use crate::error::XlateError;
use crate::event::ModelEvent;
use crate::hook::StreamId;
use crate::inbound::RequestMetadata;
use crate::plugin::PluginManifest;
use crate::supervisor::PluginStatus;
use crate::types::NormalizedRequest;

pub enum KernelMessage {
    StreamRequest {
        id: StreamId,
        request: NormalizedRequest,
        metadata: RequestMetadata,
    },
    StreamEvent {
        id: StreamId,
        event: ModelEvent,
    },
    StreamError {
        id: StreamId,
        error: XlateError,
    },
    StreamCancel {
        id: StreamId,
    },

    PluginRegister {
        manifest: PluginManifest,
    },
    PluginHeartbeat {
        plugin_id: String,
        status: PluginStatus,
    },

    KernelEvent {
        event: KernelEventPayload,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KernelEventPayload {
    Alert {
        name: String,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        threshold: Option<f64>,
    },
    PluginCrashed {
        plugin: String,
        error: String,
        restart_count: u32,
    },
    PluginRecovered {
        plugin: String,
    },
    RateLimitExceeded {
        metric: String,
        current: u64,
        limit: u64,
    },
    ConfigReloaded {
        changed_sections: Vec<String>,
    },
    CircuitOpened {
        target_id: String,
        failure_count: u32,
    },
}
