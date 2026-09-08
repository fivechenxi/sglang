#!/usr/bin/env python3
"""Statically enforce io_struct's BaseReq naming convention."""

import ast
from pathlib import Path
from typing import Optional


SOURCE = Path(__file__).parents[2] / "python/sglang/srt/managers/io_struct.py"
VALID_SUFFIXES = ("Req", "Input", "Output")


def base_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def main() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    request_types = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(base_name(base) in {"BaseReq", "BaseBatchReq"} for base in node.bases)
    ]
    invalid = [name for name in request_types if not name.endswith(VALID_SUFFIXES)]
    if invalid:
        raise SystemExit(f"Invalid BaseReq/BaseBatchReq names: {', '.join(invalid)}")
    if "PrefillAdmissionAckReq" not in request_types:
        raise SystemExit("PrefillAdmissionAckReq is missing from io_struct.py")
    print(f"Validated {len(request_types)} request protocol types")


if __name__ == "__main__":
    main()
