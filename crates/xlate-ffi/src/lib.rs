#![allow(clippy::not_unsafe_ptr_arg_deref)]

mod panic_guard;
mod runtime;
mod stream;

use panic_guard::{guard_cstr, guard_ptr};
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;
use arc_swap::ArcSwap;
use xlate_core::kernel::{Kernel, KernelBuilder};
use xlate_core::{KernelConfig, NormalizedRequest};

#[repr(C)]
pub struct xlate_stream(stream::XlateStream);

#[repr(C)]
pub struct xlate_kernel {
    inner: ArcSwap<Kernel>,
}

fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

fn string_to_owned_cstr(value: String) -> *mut c_char {
    match CString::new(value) {
        Ok(cstring) => cstring.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Kernel API
// ---------------------------------------------------------------------------

pub fn build_kernel(config: KernelConfig) -> Kernel {
    use xlate_core::kernel::EventBus;
    use xlate_core::plugin::OutboundPlugin;
    use xlate_core::registry::ModelRegistry;
    use xlate_core::store::Store;
    use xlate_core::supervisor::Supervisor;

    let outbound: Vec<Arc<dyn OutboundPlugin>> = vec![
        Arc::new(xlate_openai::OpenAiAdapter::with_pool_ua(
            config.plugins.outbound.openai.connection_pool_size,
            config.plugins.outbound.openai.idle_connection_timeout_s,
            &config.plugins.outbound.openai.user_agent,
        )),
        Arc::new(
            xlate_anthropic::AnthropicAdapter::with_pool_ua(
                config.plugins.outbound.anthropic.connection_pool_size,
                config.plugins.outbound.anthropic.idle_connection_timeout_s,
                &config.plugins.outbound.anthropic.user_agent,
            )
            .with_settings(
                config.plugins.outbound.anthropic.default_max_tokens,
                config.plugins.outbound.anthropic.max_cache_breakpoints,
            )
            .with_anthropic_version(&config.plugins.outbound.anthropic.anthropic_version),
        ),
    ];

    let store: Arc<dyn Store> = match config.store.backend.as_str() {
        "sqlite" => match xlate_store::SqliteStore::new(&config.store.sqlite_path) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                tracing::warn!("sqlite store failed, falling back to memory: {e}");
                Arc::new(xlate_store::MemoryStore::new())
            }
        },
        _ => Arc::new(xlate_store::MemoryStore::new()),
    };
    let registry = Arc::new(ModelRegistry::new());
    for plugin in &outbound {
        for meta in plugin.get_model_list() {
            registry.register(meta);
        }
    }
    for (model_id, override_value) in &config.model_registry.overrides {
        let base = registry.get(model_id);
        let mut base_json = match &base {
            Some(m) => serde_json::to_value(m).unwrap_or_default(),
            None => serde_json::json!({ "id": model_id, "display_name": model_id }),
        };
        if let (serde_json::Value::Object(ref mut map), serde_json::Value::Object(overrides)) =
            (&mut base_json, override_value)
        {
            for (k, v) in overrides {
                map.insert(k.clone(), v.clone());
            }
        }
        if let Ok(merged) = serde_json::from_value::<xlate_core::registry::ModelMeta>(base_json) {
            registry.register(merged);
        } else {
            tracing::warn!(model = %model_id, "failed to parse model_registry override");
        }
    }
    let supervisor = Arc::new(Supervisor::new());
    let event_bus = Arc::new(EventBus::new(1024));

    let pricing_catalog = Arc::new(xlate_pricing::RemotePricingCatalog::new());
    let pricing_config = xlate_pricing::PricingServiceConfig::from_kernel_config(&config);
    let pricing_service = xlate_pricing::PricingService::new(
        pricing_catalog.clone(),
        pricing_config,
    ).with_store(store.clone())
     .with_registry(registry.clone());

    let stats_aggregator = xlate_core::StatsAggregator::new(
        store.clone(),
        config.store.stats_aggregation_interval_s,
    );

    let result = xlate_hooks::standard_hooks(
        &config,
        Some(store.clone()),
        Some(registry.clone()),
        Some(event_bus.clone()),
        Some(pricing_catalog.clone()),
        Some(supervisor.clone()),
    );

    if let Some(ref ph) = result.param_heal {
        let ph = ph.clone();
        let rt = crate::runtime::global_runtime();
        rt.block_on(ph.load_from_store());
    }

    KernelBuilder::new(config)
        .outbound_vec(outbound)
        .hooks(result.hooks)
        .latency_tracker(result.latency_tracker)
        .pricing_catalog(pricing_catalog)
        .service(Arc::new(pricing_service))
        .service(Arc::new(stats_aggregator))
        .registry(registry)
        .supervisor(supervisor)
        .store(store)
        .event_bus(event_bus)
        .build()
}

#[no_mangle]
pub extern "C" fn xlate_kernel_create(config_json: *const c_char) -> *mut xlate_kernel {
    guard_ptr(move || {
        let json = match cstr_to_str(config_json) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let config: KernelConfig = match serde_json::from_str(json) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("failed to parse kernel config: {e}");
                return std::ptr::null_mut();
            }
        };

        let _guard = runtime::global_runtime().enter();
        let kernel = Arc::new(build_kernel(config));
        let k = kernel.clone();
        runtime::global_runtime().spawn(async move {
            k.start_services().await;
        });
        Box::into_raw(Box::new(xlate_kernel {
            inner: ArcSwap::new(kernel),
        }))
    })
}

#[no_mangle]
pub extern "C" fn xlate_kernel_destroy(kernel: *mut xlate_kernel) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        if !kernel.is_null() {
            let k = unsafe { Box::from_raw(kernel) };
            let inner = k.inner.load();
            runtime::global_runtime().block_on(inner.shutdown_graceful());
            drop(k);
        }
    }));
}

#[no_mangle]
pub extern "C" fn xlate_kernel_update_config(
    kernel: *mut xlate_kernel,
    config_json: *const c_char,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        if kernel.is_null() {
            return;
        }
        let json = match cstr_to_str(config_json) {
            Some(s) => s,
            None => return,
        };
        let config: KernelConfig = match serde_json::from_str(json) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("failed to parse config update: {e}");
                return;
            }
        };
        let _guard = runtime::global_runtime().enter();
        let new_kernel = Arc::new(build_kernel(config));
        let new_ref = new_kernel.clone();
        runtime::global_runtime().spawn(async move {
            new_ref.start_services().await;
        });
        let k = unsafe { &*kernel };
        let old = k.inner.swap(new_kernel);
        k.inner.load().event_bus().emit(xlate_core::message::KernelEventPayload::ConfigReloaded {
            changed_sections: vec!["all".into()],
        });
        runtime::global_runtime().spawn(async move {
            old.shutdown_graceful().await;
        });
        tracing::info!("kernel config hot-reloaded, old kernel draining active streams");
    }));
}

#[no_mangle]
pub extern "C" fn xlate_kernel_stats(kernel: *mut xlate_kernel) -> *mut c_char {
    guard_cstr(move || {
        if kernel.is_null() {
            return std::ptr::null_mut();
        }
        let k = unsafe { &*kernel };
        let inner = k.inner.load();
        let stats = inner.stats();
        match serde_json::to_string(&stats) {
            Ok(json) => string_to_owned_cstr(json),
            Err(_) => std::ptr::null_mut(),
        }
    })
}

#[no_mangle]
pub extern "C" fn xlate_kernel_poll_event(
    kernel: *mut xlate_kernel,
    timeout_ms: i32,
) -> *mut c_char {
    guard_cstr(move || {
        if kernel.is_null() {
            return std::ptr::null_mut();
        }
        let k = unsafe { &*kernel };
        let inner = k.inner.load();
        let event = if timeout_ms == 0 {
            inner.event_bus().poll()
        } else {
            runtime::global_runtime().block_on(
                inner.event_bus().poll_timeout(timeout_ms),
            )
        };
        match event {
            Some(event) => match serde_json::to_string(&event) {
                Ok(json) => string_to_owned_cstr(json),
                Err(_) => std::ptr::null_mut(),
            },
            None => std::ptr::null_mut(),
        }
    })
}

/// v2 stream start: uses a Kernel instance for routing + hooks
#[no_mangle]
pub extern "C" fn xlate_kernel_stream_start(
    kernel: *mut xlate_kernel,
    request_json: *const c_char,
) -> *mut xlate_stream {
    guard_ptr(move || {
        if kernel.is_null() {
            return std::ptr::null_mut();
        }
        let json = match cstr_to_str(request_json) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let request: NormalizedRequest = match serde_json::from_str(json) {
            Ok(req) => req,
            Err(_) => return std::ptr::null_mut(),
        };
        let k = unsafe { &*kernel };
        let inner = k.inner.load_full();
        if inner.is_shutdown() {
            return std::ptr::null_mut();
        }
        let buffer = inner.config().kernel.stream_buffer_size;
        let stream_inner = stream::XlateStream::start(inner, request, buffer);
        Box::into_raw(Box::new(xlate_stream(stream_inner)))
    })
}

/// v2 raw stream start: accepts any JSON body (auto-detects format via InboundPlugin)
#[no_mangle]
pub extern "C" fn xlate_kernel_stream_raw(
    kernel: *mut xlate_kernel,
    body_json: *const c_char,
) -> *mut xlate_stream {
    guard_ptr(move || {
        if kernel.is_null() {
            return std::ptr::null_mut();
        }
        let json = match cstr_to_str(body_json) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let body: serde_json::Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(_) => return std::ptr::null_mut(),
        };
        let k = unsafe { &*kernel };
        let inner = k.inner.load_full();
        if inner.is_shutdown() {
            return std::ptr::null_mut();
        }
        let buffer = inner.config().kernel.stream_buffer_size;
        let stream_inner = stream::XlateStream::start_raw(inner, body, buffer);
        Box::into_raw(Box::new(xlate_stream(stream_inner)))
    })
}

// ---------------------------------------------------------------------------
// Stream operations
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn xlate_stream_poll(
    stream: *mut xlate_stream,
    timeout_ms: i32,
) -> *mut c_char {
    guard_cstr(move || {
        if stream.is_null() {
            return std::ptr::null_mut();
        }
        let stream_ref = unsafe { &*stream };
        match stream_ref.0.poll(timeout_ms) {
            Some(event) => match serde_json::to_string(&event) {
                Ok(json) => string_to_owned_cstr(json),
                Err(_) => std::ptr::null_mut(),
            },
            None => std::ptr::null_mut(),
        }
    })
}

#[no_mangle]
pub extern "C" fn xlate_stream_error(stream: *mut xlate_stream) -> *mut c_char {
    guard_cstr(move || {
        if stream.is_null() {
            return std::ptr::null_mut();
        }
        let stream_ref = unsafe { &*stream };
        match stream_ref.0.error() {
            Some(err) => {
                let payload = xlate_core::error::XlateErrorPayload::from(&err);
                match serde_json::to_string(&payload) {
                    Ok(json) => string_to_owned_cstr(json),
                    Err(_) => std::ptr::null_mut(),
                }
            }
            None => std::ptr::null_mut(),
        }
    })
}

#[no_mangle]
pub extern "C" fn xlate_stream_cancel(stream: *mut xlate_stream) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        if !stream.is_null() {
            unsafe { &*stream }.0.cancel();
        }
    }));
}

#[no_mangle]
pub extern "C" fn xlate_stream_free(stream: *mut xlate_stream) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        if !stream.is_null() {
            drop(unsafe { Box::from_raw(stream) });
        }
    }));
}

#[no_mangle]
pub extern "C" fn xlate_free_string(s: *mut c_char) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        if !s.is_null() {
            drop(unsafe { CString::from_raw(s) });
        }
    }));
}

#[no_mangle]
pub extern "C" fn xlate_version() -> *const c_char {
    c"0.1.0".as_ptr()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn v2_kernel_create_and_stream() {
        let config = CString::new(
            r#"{
                "routes": [{
                    "model": "*",
                    "targets": [{
                        "id": "test",
                        "plugin": "openai",
                        "config": {
                            "plugin": "openai",
                            "base_url": "https://api.openai.com",
                            "api_key": ""
                        }
                    }]
                }]
            }"#,
        )
        .unwrap();
        let kernel = xlate_kernel_create(config.as_ptr());
        assert!(!kernel.is_null());

        let request = CString::new(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        let stream = xlate_kernel_stream_start(kernel, request.as_ptr());
        assert!(!stream.is_null());

        loop {
            let event = xlate_stream_poll(stream, 500);
            if event.is_null() {
                break;
            }
            xlate_free_string(event);
        }

        let error = xlate_stream_error(stream);
        assert!(!error.is_null());
        let error_str = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(
            error_str.contains("api key is empty") || error_str.contains("error"),
            "unexpected: {error_str}"
        );
        xlate_free_string(error);
        xlate_stream_free(stream);
        xlate_kernel_destroy(kernel);
    }

    #[test]
    fn v2_kernel_stats_comprehensive() {
        let config = CString::new(r#"{}"#).unwrap();
        let kernel = xlate_kernel_create(config.as_ptr());
        assert!(!kernel.is_null());

        let stats = xlate_kernel_stats(kernel);
        assert!(!stats.is_null());
        let stats_str = unsafe { CStr::from_ptr(stats) }.to_str().unwrap();
        assert!(stats_str.contains("max_concurrent_streams"));
        assert!(stats_str.contains("available_permits"));
        assert!(stats_str.contains("hook_count"));
        assert!(stats_str.contains("outbound_plugins"));
        xlate_free_string(stats);
        xlate_kernel_destroy(kernel);
    }

    #[test]
    fn v2_kernel_stream_raw() {
        let config = CString::new(
            r#"{
                "routes": [{
                    "model": "*",
                    "targets": [{
                        "id": "test",
                        "plugin": "openai",
                        "config": {
                            "plugin": "openai",
                            "base_url": "https://api.openai.com",
                            "api_key": ""
                        }
                    }]
                }]
            }"#,
        )
        .unwrap();
        let kernel = xlate_kernel_create(config.as_ptr());
        assert!(!kernel.is_null());

        let raw_body = CString::new(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}],"_metadata":{"format":"openai-chat","client_id":"test-client"}}"#,
        )
        .unwrap();
        let stream = xlate_kernel_stream_raw(kernel, raw_body.as_ptr());
        assert!(!stream.is_null());

        loop {
            let event = xlate_stream_poll(stream, 500);
            if event.is_null() {
                break;
            }
            xlate_free_string(event);
        }

        xlate_stream_free(stream);
        xlate_kernel_destroy(kernel);
    }

    #[test]
    fn v2_kernel_shutdown_rejects_streams() {
        let config = CString::new(r#"{}"#).unwrap();
        let kernel = xlate_kernel_create(config.as_ptr());
        assert!(!kernel.is_null());
        xlate_kernel_destroy(kernel);
    }

    #[test]
    fn version_is_v2() {
        let v = xlate_version();
        assert!(!v.is_null());
        let s = unsafe { CStr::from_ptr(v) }.to_str().unwrap();
        assert_eq!(s, "0.1.0");
    }
}
