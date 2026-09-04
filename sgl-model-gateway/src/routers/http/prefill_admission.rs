// SPDX-License-Identifier: Apache-2.0

//! Fail-fast admission control for cold prefill work in PD mode.
//!
//! Reservations are independent per prefill worker. There is intentionally no
//! local waiting queue: callers receive 429 before either P or D is dispatched,
//! allowing an upstream gateway to try another serving backend.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use metrics::{counter, gauge};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rejection {
    pub requested_cold_tokens: usize,
    pub current_cold_tokens: usize,
    pub max_cold_tokens: usize,
    pub current_cold_requests: usize,
    pub max_cold_requests: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct WorkerState {
    cold_tokens: usize,
    cold_requests: usize,
}

#[derive(Debug)]
pub struct Controller {
    workers: Mutex<HashMap<String, WorkerState>>,
    max_cold_tokens: usize,
    max_cold_requests: usize,
    cold_request_threshold_tokens: usize,
}

impl Controller {
    pub fn new(
        max_cold_tokens: usize,
        max_cold_requests: usize,
        cold_request_threshold_tokens: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            workers: Mutex::new(HashMap::new()),
            max_cold_tokens,
            max_cold_requests,
            cold_request_threshold_tokens,
        })
    }

    pub fn enabled(&self) -> bool {
        self.max_cold_tokens > 0 || self.max_cold_requests > 0
    }

    pub fn try_acquire(
        self: &Arc<Self>,
        worker_url: &str,
        cold_tokens: usize,
    ) -> Result<Guard, Rejection> {
        let is_long = cold_tokens > 0 && cold_tokens >= self.cold_request_threshold_tokens;
        let mut workers = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        let state = workers.entry(worker_url.to_owned()).or_default();
        let next_tokens = state.cold_tokens.saturating_add(cold_tokens);
        let next_requests = state.cold_requests.saturating_add(usize::from(is_long));

        if (self.max_cold_tokens > 0 && next_tokens > self.max_cold_tokens)
            || (self.max_cold_requests > 0 && next_requests > self.max_cold_requests)
        {
            return Err(Rejection {
                requested_cold_tokens: cold_tokens,
                current_cold_tokens: state.cold_tokens,
                max_cold_tokens: self.max_cold_tokens,
                current_cold_requests: state.cold_requests,
                max_cold_requests: self.max_cold_requests,
            });
        }

        state.cold_tokens = next_tokens;
        state.cold_requests = next_requests;
        drop(workers);
        publish(worker_url, next_tokens, next_requests);
        counter!(
            "smg_pd_prefill_admission_total",
            "worker" => worker_url.to_owned(),
            "result" => "accepted"
        )
        .increment(1);

        Ok(Guard {
            controller: Arc::clone(self),
            worker_url: worker_url.to_owned(),
            cold_tokens,
            is_long,
        })
    }

    /// Record only a request-level rejection. Failed internal candidate probes
    /// are not externally visible 429s and must not inflate rejection metrics.
    pub fn record_rejection(&self, worker_url: &str) {
        counter!(
            "smg_pd_prefill_admission_total",
            "worker" => worker_url.to_owned(),
            "result" => "rejected"
        )
        .increment(1);
    }

    pub fn record_reroute(&self, worker_url: &str) {
        counter!(
            "smg_pd_prefill_admission_total",
            "worker" => worker_url.to_owned(),
            "result" => "rerouted"
        )
        .increment(1);
    }

    #[cfg(test)]
    fn snapshot(&self, worker_url: &str) -> WorkerState {
        self.workers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(worker_url)
            .copied()
            .unwrap_or_default()
    }
}

fn publish(worker_url: &str, cold_tokens: usize, cold_requests: usize) {
    gauge!(
        "smg_pd_prefill_admission_inflight_cold_tokens",
        "worker" => worker_url.to_owned()
    )
    .set(cold_tokens as f64);
    gauge!(
        "smg_pd_prefill_admission_inflight_cold_requests",
        "worker" => worker_url.to_owned()
    )
    .set(cold_requests as f64);
}

#[derive(Debug)]
#[must_use = "dropping this guard releases the prefill reservation"]
pub struct Guard {
    controller: Arc<Controller>,
    worker_url: String,
    cold_tokens: usize,
    is_long: bool,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let mut workers = self
            .controller
            .workers
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(state) = workers.get_mut(&self.worker_url) else {
            return;
        };
        state.cold_tokens = state.cold_tokens.saturating_sub(self.cold_tokens);
        if self.is_long {
            state.cold_requests = state.cold_requests.saturating_sub(1);
        }
        let state = *state;
        drop(workers);
        publish(&self.worker_url, state.cold_tokens, state.cold_requests);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_atomically_and_releases_on_drop() {
        let controller = Controller::new(1000, 1, 100);
        let guard = controller.try_acquire("p0", 700).unwrap();
        let rejected = controller.try_acquire("p0", 301).unwrap_err();
        assert_eq!(rejected.current_cold_tokens, 700);
        assert_eq!(controller.snapshot("p0").cold_requests, 1);
        drop(guard);
        assert_eq!(controller.snapshot("p0"), WorkerState::default());
        assert!(controller.try_acquire("p0", 1000).is_ok());
    }

    #[test]
    fn budgets_are_per_prefill_worker() {
        let controller = Controller::new(100, 0, 0);
        let _guard = controller.try_acquire("p0", 100).unwrap();
        assert!(controller.try_acquire("p0", 1).is_err());
        assert!(controller.try_acquire("p1", 100).is_ok());
    }

    #[test]
    fn zero_cold_token_cache_hit_does_not_consume_long_request_slot() {
        let controller = Controller::new(0, 1, 0);
        let _cold = controller.try_acquire("p0", 100).unwrap();
        let hit = controller.try_acquire("p0", 0).unwrap();
        assert_eq!(controller.snapshot("p0").cold_requests, 1);
        drop(hit);
        assert_eq!(controller.snapshot("p0").cold_requests, 1);
    }
}
