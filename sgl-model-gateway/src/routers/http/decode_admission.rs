// SPDX-License-Identifier: Apache-2.0

//! Per decode-DP token credits for fail-fast PD admission.
//!
//! The reservation deliberately covers the complete decode response lifetime.
//! P-side prefix hits reduce compute, but do not reduce the physical KV that a
//! D worker without a matching radix-cache entry must allocate.
//!
//! `Controller` is the legacy process-local fallback. `RemoteController` uses
//! a D-issued lease, so concurrent Router replicas share Decode's capacity fact.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use metrics::{counter, gauge};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::Worker;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rejection {
    pub requested_tokens: usize,
    pub available_tokens: usize,
    pub current_tokens: usize,
    pub max_tokens: usize,
}

#[derive(Debug)]
pub struct Controller {
    workers: Mutex<HashMap<String, usize>>,
    max_tokens: usize,
}

impl Controller {
    pub fn new(max_tokens: usize) -> Arc<Self> {
        Arc::new(Self {
            workers: Mutex::new(HashMap::new()),
            max_tokens,
        })
    }

    pub fn enabled(&self) -> bool {
        self.max_tokens > 0
    }

    pub fn try_acquire(
        self: &Arc<Self>,
        worker_url: &str,
        tokens: usize,
    ) -> Result<Guard, Rejection> {
        let mut workers = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        let current = workers.entry(worker_url.to_owned()).or_default();
        let next = current.saturating_add(tokens);
        if next > self.max_tokens {
            return Err(Rejection {
                requested_tokens: tokens,
                available_tokens: self.max_tokens.saturating_sub(*current),
                current_tokens: *current,
                max_tokens: self.max_tokens,
            });
        }
        *current = next;
        drop(workers);
        publish(worker_url, next);
        counter!(
            "smg_pd_decode_admission_total",
            "worker" => worker_url.to_owned(),
            "result" => "accepted"
        )
        .increment(1);
        Ok(Guard {
            controller: Arc::clone(self),
            worker_url: worker_url.to_owned(),
            tokens,
        })
    }

    pub fn record_rejection(&self, worker_url: &str) {
        counter!(
            "smg_pd_decode_admission_total",
            "worker" => worker_url.to_owned(),
            "result" => "rejected"
        )
        .increment(1);
    }

    pub fn record_reroute(&self, worker_url: &str) {
        counter!(
            "smg_pd_decode_admission_total",
            "worker" => worker_url.to_owned(),
            "result" => "rerouted"
        )
        .increment(1);
    }

    #[cfg(test)]
    fn snapshot(&self, worker_url: &str) -> usize {
        *self
            .workers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(worker_url)
            .unwrap_or(&0)
    }
}

fn publish(worker_url: &str, tokens: usize) {
    gauge!(
        "smg_pd_decode_admission_inflight_tokens",
        "worker" => worker_url.to_owned()
    )
    .set(tokens as f64);
}

#[derive(Debug)]
#[must_use = "dropping this guard releases the decode token reservation"]
pub struct Guard {
    controller: Arc<Controller>,
    worker_url: String,
    tokens: usize,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let mut workers = self
            .controller
            .workers
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(current) = workers.get_mut(&self.worker_url) else {
            return;
        };
        *current = current.saturating_sub(self.tokens);
        let remaining = *current;
        drop(workers);
        publish(&self.worker_url, remaining);
    }
}

#[derive(Debug)]
pub enum RemoteError {
    Rejected(Rejection),
    Unavailable(String),
}

#[derive(Debug, Default)]
pub struct RemoteController;

#[derive(Serialize)]
struct ReservationRequest<'a> {
    operation: &'static str,
    reservation_id: &'a str,
    dp_rank: usize,
    tokens: usize,
}

#[derive(Deserialize)]
struct ReservationResponse {
    accepted: bool,
    reserved_tokens: usize,
    admittable_tokens: usize,
}

impl RemoteController {
    pub async fn try_acquire(
        client: &Client,
        worker: &dyn Worker,
        tokens: usize,
    ) -> Result<RemoteGuard, RemoteError> {
        let dp_rank = worker.dp_rank().unwrap_or(0);
        let endpoint = format!("{}/internal/decode_token_reservation", worker.base_url());
        let reservation_id = Uuid::new_v4().to_string();
        let response = post_reservation(
            client,
            &endpoint,
            ReservationRequest {
                operation: "reserve",
                reservation_id: &reservation_id,
                dp_rank,
                tokens,
            },
        )
        .await
        .map_err(|error| {
            counter!(
                "smg_pd_decode_admission_total",
                "worker" => worker.url().to_owned(),
                "result" => "remote_unavailable"
            )
            .increment(1);
            RemoteError::Unavailable(error)
        })?;

        if !response.accepted {
            return Err(RemoteError::Rejected(Rejection {
                requested_tokens: tokens,
                available_tokens: response.admittable_tokens,
                current_tokens: 0,
                max_tokens: 0,
            }));
        }
        counter!(
            "smg_pd_decode_admission_total",
            "worker" => worker.url().to_owned(),
            "result" => "accepted"
        )
        .increment(1);
        debug_assert_eq!(response.reserved_tokens, tokens);
        Ok(RemoteGuard {
            client: client.clone(),
            endpoint,
            reservation_id,
            dp_rank,
        })
    }
}

async fn post_reservation(
    client: &Client,
    endpoint: &str,
    request: ReservationRequest<'_>,
) -> Result<ReservationResponse, String> {
    let response = client
        .post(endpoint)
        .timeout(Duration::from_secs(2))
        .json(&request)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!(
            "reservation endpoint returned {}",
            response.status()
        ));
    }
    response.json().await.map_err(|error| error.to_string())
}

#[derive(Debug)]
pub struct RemoteGuard {
    client: Client,
    endpoint: String,
    reservation_id: String,
    dp_rank: usize,
}

impl RemoteGuard {
    pub fn reservation_id(&self) -> &str {
        &self.reservation_id
    }
}

impl Drop for RemoteGuard {
    fn drop(&mut self) {
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let reservation_id = self.reservation_id.clone();
        let dp_rank = self.dp_rank;
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = post_reservation(
                    &client,
                    &endpoint,
                    ReservationRequest {
                        operation: "release",
                        reservation_id: &reservation_id,
                        dp_rank,
                        tokens: 0,
                    },
                )
                .await;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn credits_are_atomic_per_decode_rank_and_release_on_drop() {
        let controller = Controller::new(1000);
        let guard = controller.try_acquire("d0@0", 700).unwrap();
        let rejected = controller.try_acquire("d0@0", 301).unwrap_err();
        assert_eq!(rejected.current_tokens, 700);
        assert!(controller.try_acquire("d0@1", 1000).is_ok());
        drop(guard);
        assert_eq!(controller.snapshot("d0@0"), 0);
    }

    #[tokio::test]
    async fn remote_controller_reserves_against_decode_endpoint() {
        use axum::{routing::post, Json, Router};
        use serde_json::{json, Value};

        async fn reserve(Json(request): Json<Value>) -> Json<Value> {
            assert_eq!(request["operation"], "reserve");
            assert_eq!(request["dp_rank"], 0);
            assert_eq!(request["tokens"], 700);
            Json(json!({
                "accepted": true,
                "reserved_tokens": 700,
                "admittable_tokens": 300,
            }))
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route("/internal/decode_token_reservation", post(reserve));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let worker = BasicWorkerBuilder::new(format!("http://{address}"))
            .worker_type(WorkerType::Decode)
            .build();

        let guard = RemoteController::try_acquire(&Client::new(), &worker, 700)
            .await
            .unwrap();
        assert!(!guard.reservation_id().is_empty());
        drop(guard);
        server.abort();
    }
}
