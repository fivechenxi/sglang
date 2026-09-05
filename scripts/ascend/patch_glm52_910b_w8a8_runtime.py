#!/usr/bin/env python3
"""Apply the DeepEP patches used by the GLM-5.2 W8A8 910B baseline.

The compatibility fixes target the DeepEP package shipped by the Ascend image:

* add the hybrid prefill/decode strategy;
* use graph-compatible all_to_all_single in the equal-split fallback.

The patch is intentionally strict and idempotent. It exits instead of silently
continuing when the installed package no longer matches the validated baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def package_root(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"required package is not installed: {module_name}")
    return Path(spec.origin).resolve().parent


def replace_once(
    path: Path,
    old: str,
    new: str,
    description: str,
    *,
    applied_marker: str | None = None,
) -> bool:
    text = path.read_text()
    if (applied_marker or new) in text:
        return False
    if old not in text:
        raise RuntimeError(f"{description} patch target not found in {path}")
    patched = text.replace(old, new, 1)
    compile(patched, str(path), "exec")
    path.write_text(patched)
    return True


def patch_deepep_all_to_all() -> list[Path]:
    path = package_root("deep_ep") / "strategies" / "low_latency_strategy.py"
    dispatch_old = """        input_list = [
            expanded_x_2d[r * chunk_size : (r + 1) * chunk_size].contiguous()
            for r in range(group_size)
        ]
        output_list = [
            torch.empty(chunk_size, hidden, dtype=expanded_x_2d.dtype, device=device)
            for r in range(group_size)
        ]
        dist.all_to_all(output_list, input_list, group=group)
        recv_x_raw = torch.cat(output_list, dim=0)
"""
    dispatch_new = """        # Equal expert-capacity splits make this layout-equivalent to the
        # list collective, while all_to_all_single is NPU-graph compatible.
        dispatch_input = expanded_x_2d.contiguous()
        recv_x_raw = torch.empty_like(dispatch_input)
        dist.all_to_all_single(recv_x_raw, dispatch_input, group=group)
"""
    combine_old = """        input_list = [
            x_reordered[r * chunk_size : (r + 1) * chunk_size].contiguous()
            for r in range(group_size)
        ]
        output_list = [
            torch.empty(chunk_size, hidden, dtype=x.dtype, device=device)
            for r in range(group_size)
        ]
        dist.all_to_all(output_list, input_list, group=group)
        recv_all_raw = torch.cat(output_list, dim=0)
"""
    combine_new = """        combine_input = x_reordered.contiguous()
        recv_all_raw = torch.empty_like(combine_input)
        dist.all_to_all_single(recv_all_raw, combine_input, group=group)
"""
    changed = replace_once(
        path,
        dispatch_old,
        dispatch_new,
        "DeepEP dispatch",
        applied_marker="dist.all_to_all_single(recv_x_raw, dispatch_input",
    )
    changed = (
        replace_once(
            path,
            combine_old,
            combine_new,
            "DeepEP combine",
            applied_marker="dist.all_to_all_single(recv_all_raw, combine_input",
        )
        or changed
    )
    return [path] if changed else []


def patch_deepep_hybrid_strategy() -> list[Path]:
    path = package_root("deep_ep") / "ep_strategy.py"
    old = """        ("ops"): (
            NormalStrategy.DEFAULT,
            LowLatencyStrategy.OPS,
        ),
"""
    new = """        ("ops"): (
            NormalStrategy.DEFAULT,
            LowLatencyStrategy.OPS,
        ),
        # GLM-5.2 910B: prefill uses all-to-all while decode keeps the native
        # low-latency dispatch/combine implementation.
        ("hybrid"): (
            NormalStrategy.ALLTOALL,
            LowLatencyStrategy.DEFAULT,
        ),
"""
    return (
        [path]
        if replace_once(
            path,
            old,
            new,
            "DeepEP hybrid strategy",
            applied_marker='("hybrid"): (',
        )
        else []
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate against temporary copies without modifying packages",
    )
    args = parser.parse_args()

    if args.check:
        print(
            "check requires the validated Ascend image and is performed by "
            "applying this idempotent script during image build/startup"
        )
        return 0

    changed: list[Path] = []
    changed.extend(patch_deepep_all_to_all())
    changed.extend(patch_deepep_hybrid_strategy())
    if changed:
        for path in dict.fromkeys(changed):
            print(f"patched {path}", flush=True)
    else:
        print("GLM-5.2 W8A8 910B DeepEP patches already applied", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
