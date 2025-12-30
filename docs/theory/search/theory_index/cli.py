"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CLI entrypoint for the theory index/search tools.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .index_io import load_index, load_theory_lines
from .modes_assemble_validate import mode_assemble, mode_validate
from .modes_basic import mode_help, mode_search, mode_stats, mode_tree
from .presets import apply_preset
from .sqlite_build import mode_build_sqlite, mode_build_sqlite_chain
from .sqlite_search import mode_sqlite_search_chain


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to ALL_index.yaml")
    parser.add_argument(
        "--theory",
        help="Path to All.md or to a manifest listing chained part files",
    )
    parser.add_argument(
        "--mode",
        default="search",
        help=(
            "Mode: search|assemble|stats|validate|tree|sqlite_build|sqlite_build_chain|"
            "sqlite_search|help"
        ),
    )
    parser.add_argument("--tag")
    parser.add_argument("--category")
    parser.add_argument("--phrase")
    parser.add_argument(
        "--db-path", help="Path to SQLite db/dir/manifest for sqlite_* modes"
    )
    parser.add_argument(
        "--max-db-bytes",
        type=int,
        default=15_000_000,
        help="Max size of a single SQLite shard for sqlite_build_chain (bytes).",
    )
    parser.add_argument(
        "--scope",
        choices=["segments", "formulas"],
        default="segments",
        help="SQLite search scope (segments or formulas)",
    )
    parser.add_argument("--preset", choices=["earth", "sun", "particles"])
    parser.add_argument("--output-path", help="Output path for assemble mode")
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["text", "json"],
        default="text",
        help="Output format (text or json)",
    )

    args = parser.parse_args(argv)
    fmt = args.fmt or "text"

    if args.mode == "help":
        return mode_help()

    idx = load_index(args.index)
    lines = load_theory_lines(args.theory) if args.theory else None
    tag, cat, phr = apply_preset(args.preset, args.tag, args.category, args.phrase)

    if args.mode == "search":
        return mode_search(idx, lines, tag, cat, phr, fmt)
    if args.mode == "assemble":
        return mode_assemble(idx, lines, tag, cat, phr, args.output_path, fmt)
    if args.mode == "stats":
        return mode_stats(idx, fmt)
    if args.mode == "sqlite_build":
        return mode_build_sqlite(idx, lines, args.db_path or "")
    if args.mode == "sqlite_build_chain":
        return mode_build_sqlite_chain(
            idx, lines, args.db_path or "", args.max_db_bytes
        )
    if args.mode == "sqlite_search":
        return mode_sqlite_search_chain(
            args.db_path or "", phr, args.scope, cat, tag, fmt
        )
    if args.mode == "validate":
        return mode_validate(idx, lines, fmt)
    if args.mode == "tree":
        return mode_tree(idx, fmt)

    import sys

    print("Unknown mode:", args.mode, file=sys.stderr)
    return 1
