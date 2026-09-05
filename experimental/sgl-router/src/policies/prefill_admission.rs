// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Drop-safe, per-prefill-worker admission for PD-disaggregated requests.
//!
//! This is deliberately a fail-fast gate rather than a queue: callers that
//! cannot reserve capacity receive HTTP 429 and can route the request to a
//! different serving backend. A successful guard is moved into the detached
//! prefill task and releases its reservation only when that task finishes,
//! which includes the P-to-D KV transfer lifetime.

use crate::config::ActiveLoadConfig;
use crate::discovery::WorkerId;
use crate::server::metrics::{MetricsRegistry, PrefillAdmissionLoadKind, PrefillAdmissionOutcome};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillAdmissionSnapshot {
    pub cold_tokens: usize,
    pub cold_requests: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillAdmissionRejected {
    pub requested_cold_tokens: usize,
    pub current_cold_tokens: usize,
    pub max_cold_tokens: Option<usize>,
    pub current_cold_requests: usize,
    pub max_cold_requests: Option<usize>,
}

#[derive(Debug, Default)]
struct WorkerAdmission {
    cold_tokens: usize,
    cold_requests: usize,
}

#[derive(Debug)]
pub struct PrefillAdmissionRegistry {
    workers: DashMap<WorkerId, Arc<Mutex<WorkerAdmission>>>,
    max_cold_tokens: Option<usize>,
    max_cold_requests: Option<usize>,
    cold_request_threshold_tokens: usize,
    metrics: OnceLock<Arc<MetricsRegistry>>,
}

impl PrefillAdmissionRegistry {
    pub fn from_config(config: &ActiveLoadConfig) -> Arc<Self> {
        Arc::new(Self {
            workers: DashMap::new(),
            max_cold_tokens: config.prefill_admission_max_inflight_cold_tokens,
            max_cold_requests: config.prefill_admission_max_inflight_cold_requests,
            cold_request_threshold_tokens: config.prefill_admission_cold_request_threshold_tokens,
            metrics: OnceLock::new(),
        })
    }

    pub fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
    }

    pub fn enabled(&self) -> bool {
        self.max_cold_tokens.is_some() || self.max_cold_requests.is_some()
    }

    /// Atomically reserve capacity on one prefill worker. There is no wait
    /// path: rejection is returned synchronously so ingress can emit 429
    /// before either the P or D request is dispatched.
    pub fn try_acquire(
        self: &Arc<Self>,
        worker: WorkerId,
        worker_url: &str,
        cold_tokens: usize,
    ) -> Result<PrefillAdmissionGuard, PrefillAdmissionRejected> {
        let state = self
            .workers
            .entry(worker.clone())
            .or_insert_with(|| Arc::new(Mutex::new(WorkerAdmission::default())))
            .clone();
        let is_cold_request = cold_tokens >= self.cold_request_threshold_tokens;
        let mut locked = state.lock();
        let next_tokens = locked.cold_tokens.saturating_add(cold_tokens);
        let next_requests = locked
            .cold_requests
            .saturating_add(usize::from(is_cold_request));
        let tokens_exceeded = self.max_cold_tokens.is_some_and(|max| next_tokens > max);
        let requests_exceeded = self
            .max_cold_requests
            .is_some_and(|max| next_requests > max);
        if tokens_exceeded || requests_exceeded {
            if let Some(metrics) = self.metrics.get() {
                metrics.record_prefill_admission(worker_url, PrefillAdmissionOutcome::Rejected);
            }
            return Err(PrefillAdmissionRejected {
                requested_cold_tokens: cold_tokens,
                current_cold_tokens: locked.cold_tokens,
                max_cold_tokens: self.max_cold_tokens,
                current_cold_requests: locked.cold_requests,
                max_cold_requests: self.max_cold_requests,
            });
        }
        locked.cold_tokens = next_tokens;
        locked.cold_requests = next_requests;
        drop(locked);
        self.publish_load(worker_url, next_tokens, next_requests);
        if let Some(metrics) = self.metrics.get() {
            metrics.record_prefill_admission(worker_url, PrefillAdmissionOutcome::Accepted);
        }
        Ok(PrefillAdmissionGuard {
            registry: Arc::clone(self),
            worker,
            worker_url: worker_url.to_owned(),
            cold_tokens,
            is_cold_request,
        })
    }

    fn publish_load(&self, worker_url: &str, cold_tokens: usize, cold_requests: usize) {
        if let Some(metrics) = self.metrics.get() {
            metrics.set_prefill_admission_load(
                worker_url,
                PrefillAdmissionLoadKind::ColdTokens,
                cold_tokens as i64,
            );
            metrics.set_prefill_admission_load(
                worker_url,
                PrefillAdmissionLoadKind::ColdRequests,
                cold_requests as i64,
            );
        }
    }

    pub fn snapshot(&self, worker: &WorkerId) -> PrefillAdmissionSnapshot {
        let Some(state) = self.workers.get(worker).map(|v| Arc::clone(v.value())) else {
            return PrefillAdmissionSnapshot {
                cold_tokens: 0,
                cold_requests: 0,
            };
        };
        let locked = state.lock();
        let snapshot = PrefillAdmissionSnapshot {
            cold_tokens: locked.cold_tokens,
            cold_requests: locked.cold_requests,
        };
        drop(locked);
        snapshot
    }
}

#[derive(Debug)]
#[must_use = "dropping the guard releases the prefill admission reservation"]
pub struct PrefillAdmissionGuard {
    registry: Arc<PrefillAdmissionRegistry>,
    worker: WorkerId,
    worker_url: String,
    cold_tokens: usize,
    is_cold_request: bool,
}

impl Drop for PrefillAdmissionGuard {
    fn drop(&mut self) {
        let Some(state) = self
            .registry
            .workers
            .get(&self.worker)
            .map(|v| Arc::clone(v.value()))
        else {
            return;
        };
        let mut locked = state.lock();
        locked.cold_tokens = locked.cold_tokens.saturating_sub(self.cold_tokens);
        if self.is_cold_request {
            locked.cold_requests = locked.cold_requests.saturating_sub(1);
        }
        let cold_tokens = locked.cold_tokens;
        let cold_requests = locked.cold_requests;
        drop(locked);
        self.registry
            .publish_load(&self.worker_url, cold_tokens, cold_requests);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(token_cap: Option<usize>, request_cap: Option<usize>) -> ActiveLoadConfig {
        ActiveLoadConfig {
            prefill_admission_max_inflight_cold_tokens: token_cap,
            prefill_admission_max_inflight_cold_requests: request_cap,
            prefill_admission_cold_request_threshold_tokens: 100,
            ..ActiveLoadConfig::default()
        }
    }

    #[test]
    fn token_budget_is_atomic_and_guard_releases_it() {
        let registry = PrefillAdmissionRegistry::from_config(&config(Some(1000), None));
        let worker = WorkerId("p0".into());
        let first = registry
            .try_acquire(worker.clone(), "http://p0", 700)
            .unwrap();
        let rejected = registry
            .try_acquire(worker.clone(), "http://p0", 301)
            .unwrap_err();
        assert_eq!(rejected.current_cold_tokens, 700);
        assert_eq!(registry.snapshot(&worker).cold_tokens, 700);
        drop(first);
        assert_eq!(registry.snapshot(&worker).cold_tokens, 0);
        let _second = registry.try_acquire(worker, "http://p0", 1000).unwrap();
    }

    #[test]
    fn request_cap_counts_only_long_cold_requests() {
        let registry = PrefillAdmissionRegistry::from_config(&config(None, Some(1)));
        let worker = WorkerId("p0".into());
        let small = registry
            .try_acquire(worker.clone(), "http://p0", 99)
            .unwrap();
        let long = registry
            .try_acquire(worker.clone(), "http://p0", 100)
            .unwrap();
        assert_eq!(registry.snapshot(&worker).cold_requests, 1);
        assert!(registry
            .try_acquire(worker.clone(), "http://p0", 101)
            .is_err());
        drop(long);
        assert!(registry
            .try_acquire(worker.clone(), "http://p0", 101)
            .is_ok());
        drop(small);
    }

    #[test]
    fn budgets_are_independent_per_worker() {
        let registry = PrefillAdmissionRegistry::from_config(&config(Some(100), None));
        let _p0 = registry
            .try_acquire(WorkerId("p0".into()), "http://p0", 100)
            .unwrap();
        assert!(registry
            .try_acquire(WorkerId("p0".into()), "http://p0", 1)
            .is_err());
        assert!(registry
            .try_acquire(WorkerId("p1".into()), "http://p1", 100)
            .is_ok());
    }
}
