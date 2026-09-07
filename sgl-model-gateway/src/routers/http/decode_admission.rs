// SPDX-License-Identifier: Apache-2.0

//! Per decode-DP token credits for fail-fast PD admission.
//!
//! The reservation deliberately covers the complete decode response lifetime.
//! P-side prefix hits reduce compute, but do not reduce the physical KV that a
//! D worker without a matching radix-cache entry must allocate.
//!
//! Credits are process-local. Production must run one active PD router for a
//! worker set and drain/gate traffic before restarting it; an exact multi-router
//! implementation requires a D-issued lease rather than local estimation.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use metrics::{counter, gauge};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rejection {
    pub requested_tokens: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
