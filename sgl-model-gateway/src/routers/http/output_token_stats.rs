// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Mutex};

use metrics::gauge;
use serde_json::Value;

const WINDOW_SIZE: usize = 100;
const MIN_SAMPLES: usize = 20;
const OUTPUT_RESERVE_CEILING: usize = 4096;

#[derive(Debug)]
struct Window {
    values: Box<[u32; WINDOW_SIZE]>,
    count: usize,
    cursor: usize,
    p90: usize,
}

impl Default for Window {
    fn default() -> Self {
        Self {
            values: Box::new([0; WINDOW_SIZE]),
            count: 0,
            cursor: 0,
            p90: 0,
        }
    }
}

impl Window {
    fn record(&mut self, tokens: usize) {
        self.values[self.cursor] = tokens.min(u32::MAX as usize) as u32;
        self.cursor = (self.cursor + 1) % WINDOW_SIZE;
        self.count = (self.count + 1).min(WINDOW_SIZE);

        if self.count >= MIN_SAMPLES {
            let mut sorted = self.values[..self.count].to_vec();
            sorted.sort_unstable();
            self.p90 = sorted[(9 * sorted.len() - 1) / 10] as usize;
        }
    }

    fn reserve(&self, fallback: usize, request_max: usize) -> usize {
        let estimate = if self.count >= MIN_SAMPLES {
            self.p90.max(fallback)
        } else {
            fallback
        };
        estimate.min(OUTPUT_RESERVE_CEILING).min(request_max)
    }
}

#[derive(Debug, Default)]
pub struct OutputTokenStats {
    by_model: Mutex<HashMap<String, Window>>,
}

impl OutputTokenStats {
    pub fn record(&self, model: &str, tokens: usize) {
        let (p90, count) = {
            let mut windows = self
                .by_model
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let window = windows.entry(model.to_owned()).or_default();
            window.record(tokens);
            (window.p90, window.count)
        };
        gauge!("smg_pd_decode_output_p90_tokens", "model" => model.to_owned()).set(p90 as f64);
        gauge!("smg_pd_decode_output_window_samples", "model" => model.to_owned())
            .set(count as f64);
    }

    pub fn reserve(&self, model: &str, fallback: usize, request_max: usize) -> usize {
        self.by_model
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(model)
            .map(|window| window.reserve(fallback, request_max))
            .unwrap_or_else(|| fallback.min(OUTPUT_RESERVE_CEILING).min(request_max))
    }
}

#[derive(Debug, Default)]
pub struct OutputObservation {
    exact_tokens: Option<usize>,
    text: String,
    pending: Vec<u8>,
    completed: bool,
    failed: bool,
}

impl OutputObservation {
    pub fn observe_json(&mut self, value: &Value) {
        self.failed |= value.get("error").is_some()
            || value.get("object").and_then(Value::as_str) == Some("error");
        if let Some(tokens) = exact_completion_tokens(value) {
            self.exact_tokens = Some(tokens);
        }
        collect_generated_text(value, &mut self.text);
        self.completed |= value
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                choices.iter().any(|choice| {
                    choice
                        .get("finish_reason")
                        .is_some_and(|reason| !reason.is_null())
                })
            })
            || value
                .pointer("/meta_info/finish_reason")
                .is_some_and(|reason| !reason.is_null());
    }

    pub fn observe_sse_chunk(&mut self, chunk: &[u8]) {
        self.pending.extend_from_slice(chunk);
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line = self.pending.drain(..=end).collect::<Vec<_>>();
            self.observe_sse_line(&line);
        }
    }

    fn observe_sse_line(&mut self, line: &[u8]) {
        let line = line.strip_suffix(b"\n").unwrap_or(line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(data) = line.strip_prefix(b"data:") else {
            return;
        };
        let data = data.strip_prefix(b" ").unwrap_or(data);
        if data == b"[DONE]" {
            self.completed = true;
            return;
        }
        if let Ok(value) = serde_json::from_slice::<Value>(data) {
            self.observe_json(&value);
        }
    }

    pub fn finish(mut self, count_text: impl FnOnce(&str) -> usize) -> usize {
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.observe_sse_line(&pending);
        }
        self.exact_tokens.unwrap_or_else(|| count_text(&self.text))
    }

    pub fn is_completed(&self) -> bool {
        self.completed && !self.failed
    }

    pub fn is_failed(&self) -> bool {
        self.failed
    }
}

fn exact_completion_tokens(value: &Value) -> Option<usize> {
    value
        .pointer("/usage/completion_tokens")
        .or_else(|| value.pointer("/usage/output_tokens"))
        .or_else(|| value.pointer("/meta_info/completion_tokens"))
        .and_then(Value::as_u64)
        .map(|tokens| tokens as usize)
}

fn collect_generated_text(value: &Value, output: &mut String) {
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        output.push_str(text);
    }
    let Some(choices) = value.get("choices").and_then(Value::as_array) else {
        return;
    };
    for choice in choices {
        if let Some(text) = choice.get("text").and_then(Value::as_str) {
            output.push_str(text);
        }
        for message_key in ["delta", "message"] {
            let Some(message) = choice.get(message_key) else {
                continue;
            };
            for key in ["reasoning_content", "reasoning", "content"] {
                if let Some(text) = message.get(key).and_then(Value::as_str) {
                    output.push_str(text);
                }
            }
            if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    if let Some(function) = tool_call.get("function") {
                        for key in ["name", "arguments"] {
                            if let Some(text) = function.get(key).and_then(Value::as_str) {
                                output.push_str(text);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p90_uses_last_100_samples_and_keeps_floor_and_ceiling() {
        let stats = OutputTokenStats::default();
        for value in 1..=19 {
            stats.record("m", value);
        }
        assert_eq!(stats.reserve("m", 640, 10_000), 640);
        for value in 1000..=1100 {
            stats.record("m", value);
        }
        assert_eq!(stats.reserve("m", 640, 10_000), 1090);
        for value in 10_000..10_100 {
            stats.record("m", value);
        }
        assert_eq!(stats.reserve("m", 640, 10_000), 4096);
        assert_eq!(stats.reserve("m", 640, 1024), 1024);
    }

    #[test]
    fn parses_usage_or_streamed_generated_text_without_modifying_chunks() {
        let mut observation = OutputObservation::default();
        observation.observe_sse_chunk(
            b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"ab\"}}]}\n\n",
        );
        observation.observe_sse_chunk(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"cd\"}}],\"usage\":{\"completion_tokens\":7}}\n\n",
        );
        assert_eq!(observation.finish(|text| text.len()), 7);

        let mut fallback = OutputObservation::default();
        fallback.observe_sse_chunk(b"data: {\"choices\":[{\"delta\":{\"content\":\"ab");
        fallback.observe_sse_chunk(b"cd\"}}]}\n\n");
        assert!(!fallback.is_completed());
        fallback.observe_sse_chunk(b"data: [DONE]\n\n");
        assert!(fallback.is_completed());
        assert_eq!(fallback.finish(|text| text.len()), 4);

        let mut failed = OutputObservation::default();
        failed.observe_sse_chunk(b"data: {\"error\":{\"message\":\"boom\"}}\n\n");
        failed.observe_sse_chunk(b"data: [DONE]\n\n");
        assert!(!failed.is_completed());
    }
}
