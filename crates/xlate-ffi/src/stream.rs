use std::sync::Arc;
use tokio::sync::mpsc;
use xlate_core::error::XlateError;
use xlate_core::kernel::Kernel;
use xlate_core::{EventSink, ModelEvent, NormalizedRequest};

pub struct XlateStream {
    events_rx: tokio::sync::Mutex<mpsc::Receiver<ModelEvent>>,
    cancel_tx: tokio::sync::watch::Sender<bool>,
    error_slot: Arc<std::sync::Mutex<Option<XlateError>>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

struct ChannelSink {
    tx: mpsc::Sender<ModelEvent>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
}

#[async_trait::async_trait]
impl EventSink for ChannelSink {
    async fn send(&mut self, event: ModelEvent) -> Result<(), XlateError> {
        if *self.cancel_rx.borrow() {
            return Err(XlateError::Canceled);
        }
        self.tx.send(event).await.map_err(|_| XlateError::Canceled)
    }
}

impl XlateStream {
    pub fn start(kernel: Arc<Kernel>, request: NormalizedRequest, buffer: usize) -> Self {
        let (events_tx, events_rx) = mpsc::channel::<ModelEvent>(buffer);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let error_slot = Arc::new(std::sync::Mutex::new(None));
        let error_slot_task = error_slot.clone();

        let task = crate::runtime::global_runtime().spawn(async move {
            let mut sink = ChannelSink {
                tx: events_tx,
                cancel_rx,
            };
            if let Err(err) = kernel.stream_normalized(request, &mut sink).await {
                if let Ok(mut slot) = error_slot_task.lock() {
                    *slot = Some(err);
                }
            }
        });

        Self {
            events_rx: tokio::sync::Mutex::new(events_rx),
            cancel_tx,
            error_slot,
            task: Some(task),
        }
    }

    pub fn start_raw(kernel: Arc<Kernel>, body: serde_json::Value, buffer: usize) -> Self {
        let (events_tx, events_rx) = mpsc::channel::<ModelEvent>(buffer);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let error_slot = Arc::new(std::sync::Mutex::new(None));
        let error_slot_task = error_slot.clone();

        let task = crate::runtime::global_runtime().spawn(async move {
            let mut sink = ChannelSink {
                tx: events_tx,
                cancel_rx,
            };
            if let Err(err) = kernel.stream_raw(&body, &mut sink).await {
                if let Ok(mut slot) = error_slot_task.lock() {
                    *slot = Some(err);
                }
            }
        });

        Self {
            events_rx: tokio::sync::Mutex::new(events_rx),
            cancel_tx,
            error_slot,
            task: Some(task),
        }
    }

    pub fn poll(&self, timeout_ms: i32) -> Option<ModelEvent> {
        crate::runtime::global_runtime().block_on(async {
            let mut rx = self.events_rx.lock().await;
            match timeout_ms {
                0 => rx.try_recv().ok(),
                ms if ms < 0 => rx.recv().await,
                ms => {
                    tokio::time::timeout(
                        std::time::Duration::from_millis(ms as u64),
                        rx.recv(),
                    )
                    .await
                    .ok()
                    .flatten()
                }
            }
        })
    }

    pub fn error(&self) -> Option<XlateError> {
        self.error_slot.lock().ok().and_then(|mut slot| slot.take())
    }

    pub fn cancel(&self) {
        let _ = self.cancel_tx.send(true);
    }
}

impl Drop for XlateStream {
    fn drop(&mut self) {
        let _ = self.cancel_tx.send(true);
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
