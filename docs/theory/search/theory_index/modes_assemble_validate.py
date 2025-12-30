"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Modes: assemble and validate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from .index_data import IndexData
from .segment import Segment


def mode_assemble(
    idx: IndexData,
    lines: Optional[List[str]],
    tag: Optional[str],
    category: Optional[str],
    phrase: Optional[str],
    out_path: Optional[str],
    fmt: str,
) -> int:
    if lines is None:
        print("ERROR: --theory is required for assemble mode.", file=sys.stderr)
        return 1
    if not out_path:
        print("ERROR: --output-path is required for assemble mode.", file=sys.stderr)
        return 1

    # Reuse search filters to decide which segments to include
    # (We use a minimal in-memory filter to keep behavior consistent.)
    used: List[Segment] = []
    cache: Dict[str, str] = {}
    for seg in idx.segments:
        if tag and tag.lower() not in seg.id.lower():
            continue
        if category and category.lower() not in seg.category.lower():
            continue
        # reuse phrase matching from mode_search: call directly in json mode is overkill
        # so we do a simple local match: search in keywords+summary+text
        if phrase:
            if seg.id not in cache:
                parts = [str(kw) for kw in (seg.keywords or [])] + [str(seg.summary)]
                for s, e in seg.ranges:
                    s0 = max(0, s - 1)
                    e0 = min(len(lines), e)
                    parts.append("".join(lines[s0:e0]))
                cache[seg.id] = " ".join(parts).lower()
            if phrase.lower() not in cache[seg.id]:
                continue
        used.append(seg)

    used = sorted(used, key=lambda s: s.start_line)
    chunks: List[str] = []
    for seg in used:
        for a, b in seg.ranges:
            a0 = max(0, a - 1)
            b0 = min(len(lines), b)
            chunks.append("".join(lines[a0:b0]) + "\n")

    Path(out_path).write_text("".join(chunks), encoding="utf-8")
    if fmt == "json":
        import json

        payload = {
            "output_path": out_path,
            "segments": [s.id for s in used],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Assembled {len(used)} segments → {out_path}")
    return 0


def mode_validate(idx: IndexData, lines: Optional[List[str]], fmt: str) -> int:
    if lines is None:
        print("ERROR: --theory is required for validate mode.", file=sys.stderr)
        return 1

    n_lines = len(lines)
    errors: List[Dict[str, object]] = []
    warnings: List[Dict[str, object]] = []

    # Validate ranges are within bounds and non-empty
    for seg in idx.segments:
        if not seg.ranges:
            errors.append({"segment": seg.id, "type": "missing_ranges"})
            continue
        for a, b in seg.ranges:
            if a <= 0 or b <= 0:
                errors.append({"segment": seg.id, "type": "bad_range", "range": [a, b]})
                continue
            if a > b:
                errors.append(
                    {"segment": seg.id, "type": "inverted_range", "range": [a, b]}
                )
                continue
            if a > n_lines or b > n_lines:
                errors.append(
                    {
                        "segment": seg.id,
                        "type": "out_of_bounds",
                        "range": [a, b],
                        "n_lines": n_lines,
                    }
                )

        if seg.length > 5000:
            warnings.append(
                {"segment": seg.id, "type": "long_segment", "length": seg.length}
            )

    if fmt == "json":
        import json

        print(
            json.dumps(
                {"errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2
            )
        )
    else:
        for w in warnings:
            print("[W]", w)
        for e in errors:
            print("[E]", e, file=sys.stderr)
        print(f"Validation finished: {len(errors)} errors, {len(warnings)} warnings.")

    return 0 if not errors else 2
