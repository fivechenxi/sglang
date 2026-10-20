"""Arithmetic/provenance checks only; these are NOT new GPU acceptance tests."""
import json
from fractions import Fraction
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results/review_evidence_20261020.json").read_text())


class ReviewEvidenceTests(unittest.TestCase):
    def test_payload_model_includes_residual_restore(self):
        cp, tp, s = 4, 2, 4096 * 6144 * 2
        original = 2 * Fraction(tp - 1, tp) * s + (cp - 1) * s
        candidate = (
            Fraction(tp - 1, tp) * s
            + Fraction(cp * tp - 1, tp) * s
            + Fraction(tp - 1, tp) * s
        )
        self.assertEqual(original, 192 * 1024**2)
        self.assertEqual(candidate, 216 * 1024**2)
        self.assertEqual(candidate / original - 1, Fraction(1, 8))

    def test_profile_excludes_tiny_probe_but_preserves_it(self):
        traces = DATA["input_sp_profile"]
        self.assertEqual([len(t["steps"]) for t in traces], [4, 4])
        large = [[s for s in t["steps"] if s["tokens"] >= 8000] for t in traces]
        self.assertEqual([len(s) for s in large], [4, 3])
        self.assertEqual(traces[1]["steps"][-1]["tokens"], 1)
        for trace in traces:
            self.assertTrue(trace["instrumented_not_performance"])
            for step in trace["steps"]:
                self.assertAlmostEqual(
                    step["gpu_union_ms"],
                    step["comm_only_ms"]
                    + step["compute_only_ms"]
                    + step["comm_compute_overlap_ms"],
                )
        self.assertAlmostEqual(
            statistics.mean(s["gpu_union_ms"] for s in large[0]), 1440.7339744262695
        )
        self.assertAlmostEqual(
            statistics.mean(s["gpu_union_ms"] for s in large[1]), 1364.046189453125
        )

    def test_r2_gain_is_short_proxy_and_overloaded(self):
        rows = [r for r in DATA["input_sp_open_loop"] if "r2-" in r["run_id"]]
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r["n_ok"] for r in rows), 512)
        rates = {}
        for mode in ("baseline", "candidate"):
            rates[mode] = statistics.mean(
                r["central_first_delta_req_s"] for r in rows if r["mode"] == mode
            )
        gain = 100 * (rates["candidate"] / rates["baseline"] - 1)
        self.assertAlmostEqual(gain, 5.33877, places=4)
        for row in rows:
            self.assertFalse(row["sustainable_claim_allowed"])
            self.assertGreater(row["central_ttft_growth_seconds_per_arrival_second"], 0)

    def test_layout_window_accounting_and_queue_growth(self):
        arms = DATA["layout_comparison"]["arms"]
        self.assertEqual(len(arms), 2)
        for arm in arms:
            window = arm["window"]
            self.assertEqual(window["n_ok"], 96)
            self.assertEqual(window["arrived"], 60)
            self.assertEqual(window["errors"], 0)
            self.assertAlmostEqual(window["completion_req_s"], window["completed"] / 60)
            self.assertGreater(*reversed(window["backlog_begin_end"]))
        self.assertAlmostEqual(100 * (51 / 46 - 1), 10.8695652173913)
        self.assertTrue(DATA["layout_comparison"]["confounds"])

    def test_p_capacity_is_not_multiplied_by_ranks(self):
        capacity = DATA["layer_split_capacity"]["r27_r28_pool_tokens_each"]
        self.assertEqual(capacity, 110464)
        self.assertLess(73728 + 4 * 8192, capacity)
        self.assertGreater(73728 + 5 * 8192, capacity)
        self.assertIsNone(DATA["layer_split_capacity"]["fixed_budget_capacity_ratio"])

    def test_pack_missing_measurements_are_not_zero_or_claimed_pass(self):
        pack = DATA["bounded_pack"]
        self.assertGreater(pack["r27_failure_payload_bytes"], pack["buffer_bytes"])
        self.assertFalse(pack["isolated_performance_ab"])
        self.assertFalse(pack["double_buffer_allowed"])
        self.assertIsNone(pack["qualification_r65_r66_trigger_fraction"])
        self.assertIsNone(pack["per_source_registered_memory_measured"])
        self.assertEqual(pack["r28_byte_mismatches"], 0)

    def test_source_hashes_and_no_raw_requests(self):
        for digest in DATA["sources_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        for audit in DATA["layout_comparison"]["request_audit"]:
            self.assertEqual(audit["n"], 96)
            self.assertGreater(audit["input_tokens_min"], 81920)
            self.assertNotIn("requests", audit)
            self.assertNotIn("text", audit)


if __name__ == "__main__":
    unittest.main()
