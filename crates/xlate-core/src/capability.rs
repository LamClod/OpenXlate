use std::sync::Arc;

use crate::kernel::EventBus;
use crate::pricing::PricingCatalog;
use crate::registry::ModelRegistry;
use crate::router::Router;

#[derive(Debug, Clone, Copy)]
pub struct CapabilityRights {
    pub read: bool,
    pub write: bool,
    pub grant: bool,
}

impl CapabilityRights {
    pub fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            grant: false,
        }
    }

    pub fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            grant: false,
        }
    }

    pub fn full() -> Self {
        Self {
            read: true,
            write: true,
            grant: true,
        }
    }
}

pub struct Capability<T: ?Sized> {
    inner: Arc<T>,
    rights: CapabilityRights,
}

impl<T: ?Sized> Capability<T> {
    pub fn new(inner: Arc<T>, rights: CapabilityRights) -> Self {
        Self { inner, rights }
    }

    pub fn get(&self) -> &T {
        &self.inner
    }

    pub fn rights(&self) -> CapabilityRights {
        self.rights
    }

    pub fn can_read(&self) -> bool {
        self.rights.read
    }

    pub fn can_write(&self) -> bool {
        self.rights.write
    }
}

impl<T: ?Sized> Clone for Capability<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            rights: self.rights,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityType {
    Router,
    Store,
    ModelRegistry,
    PricingCatalog,
    EventBus,
    Config,
}

pub struct PluginCapabilities {
    pub store: Option<Capability<dyn crate::store::Store>>,
    pub config: Option<Capability<dyn std::any::Any + Send + Sync>>,
    pub router: Option<Capability<Router>>,
    pub model_registry: Option<Capability<ModelRegistry>>,
    pub event_bus: Option<Capability<EventBus>>,
    pub pricing: Option<Capability<dyn PricingCatalog>>,
}

impl PluginCapabilities {
    pub fn empty() -> Self {
        Self {
            store: None,
            config: None,
            router: None,
            model_registry: None,
            event_bus: None,
            pricing: None,
        }
    }
}
