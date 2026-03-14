"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CLI entrypoint for the theory index/search tools.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .index_io import load_index, load_theory_lines
from .modes_assemble_validate import mode_assemble, mode_validate
from .modes_basic import mode_help, mode_search, mode_stats, mode_tree
from .presets import apply_preset
from .sqlite_build import mode_build_sqlite, mode_build_sqlite_chain
from .sqlite_search import mode_sqlite_search_chain
from .sqlite_validate_export import (
    mode_sqlite_export_segment,
    mode_sqlite_validate_chain,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        required=False,  # Optional for sqlite_search mode
        help="Path to ALL_index.yaml (required for most modes, optional for sqlite_search)",
    )
    parser.add_argument(
        "--theory",
        help="Path to All.md or to a manifest listing chained part files",
    )
    parser.add_argument(
        "--mode",
        default="search",
        help=(
            "Mode: search|assemble|stats|validate|tree|sqlite_build|sqlite_build_chain|"
            "sqlite_search|sqlite_validate|sqlite_export_segment|help"
        ),
    )
    parser.add_argument("--tag")
    parser.add_argument("--category")
    parser.add_argument("--phrase")
    parser.add_argument(
        "--phrases",
        help=(
            "Comma-separated phrases for sqlite_search (OR). "
            "If provided, overrides --phrase."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="For sqlite_search: print only id/category/summary lines (no snippets).",
    )
    parser.add_argument(
        "--dedupe-by-id",
        action="store_true",
        help="For sqlite_search: deduplicate results by segment id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="For sqlite_search: limit number of results.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="For sqlite_search: skip N results (for pagination).",
    )
    parser.add_argument(
        "--query",
        help=(
            "For sqlite_search: logical query with AND/OR/NOT operators. "
            "Example: '(A OR B) AND C NOT D'. Overrides --phrase and --phrases."
        ),
    )
    parser.add_argument(
        "--highlight",
        action="store_true",
        help="For sqlite_search: highlight found phrases in results.",
    )
    parser.add_argument(
        "--sort",
        choices=["relevance", "id", "none"],
        default="none",
        help="For sqlite_search: sort results (relevance, id, or none).",
    )
    parser.add_argument(
        "--regex",
        action="store_true",
        help="For sqlite_search: treat phrase as regular expression.",
    )
    parser.add_argument(
        "--proximity",
        type=int,
        help="For sqlite_search: find phrases within N words of each other.",
    )
    parser.add_argument(
        "--context",
        type=int,
        help="For sqlite_search: show N lines before and after matches.",
    )
    parser.add_argument(
        "--group-by",
        choices=["category", "db", "id", "none"],
        default="none",
        help="For sqlite_search: group results by field.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        help="For sqlite_search: filter by minimum text length.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="For sqlite_search: filter by maximum text length.",
    )
    parser.add_argument(
        "--export-html",
        help="For sqlite_search: export results to HTML file.",
    )
    parser.add_argument(
        "--db-path", help="Path to SQLite db/dir/manifest for sqlite_* modes"
    )
    parser.add_argument(
        "--db-path-glob",
        help=(
            "Glob pattern for SQLite files (e.g., '*.chain.part*.sqlite'). "
            "If provided, overrides --db-path."
        ),
    )
    parser.add_argument(
        "--db-dir",
        default="/mnt/data",
        help=(
            "Directory to scan for SQLite files (default: /mnt/data). "
            "If provided, automatically finds all *.sqlite files in this directory. "
            "For sqlite_search: if neither --db-path nor --db-path-glob is provided, "
            "uses --db-dir (defaults to /mnt/data)."
        ),
    )
    parser.add_argument(
        "--segment-id",
        help="Segment id for sqlite_export_segment (e.g., 7d-105).",
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
        "--segment-ids",
        help=(
            "For assemble mode: comma-separated list of segment IDs to compile. "
            "Example: '7d-02,7d-09,7d-84'. Overrides --tag, --category, --phrase."
        ),
    )
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

    # For sqlite_search, index is optional (not used in search logic)
    # For other modes, index is required
    if args.mode != "sqlite_search":
        if not args.index:
            print(
                "ERROR: --index is required for mode '{}'".format(args.mode),
                file=sys.stderr,
            )
            return 1
        idx = load_index(args.index)
    else:
        idx = None  # Not used for sqlite_search

    lines = load_theory_lines(args.theory) if args.theory else None
    tag, cat, phr = apply_preset(args.preset, args.tag, args.category, args.phrase)

    if args.mode == "search":
        return mode_search(idx, lines, tag, cat, phr, fmt)
    if args.mode == "assemble":
        segment_ids_list = None
        if args.segment_ids:
            segment_ids_list = [sid.strip() for sid in args.segment_ids.split(",") if sid.strip()]
        return mode_assemble(idx, lines, tag, cat, phr, args.output_path, fmt, segment_ids_list)
    if args.mode == "stats":
        return mode_stats(idx, fmt)
    if args.mode == "sqlite_build":
        return mode_build_sqlite(idx, lines, args.db_path or "")
    if args.mode == "sqlite_build_chain":
        return mode_build_sqlite_chain(
            idx, lines, args.db_path or "", args.max_db_bytes
        )
    if args.mode == "sqlite_search":
        query_str = args.query
        phrases = None
        if query_str:
            # Use logical query
            phrase_list = []
        else:
            if args.phrases:
                phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]
            phrase_list = phrases if phrases else ([phr] if phr else [])
        
        # Determine database path: priority: db_path_glob > db_path > db_dir (default: /mnt/data)
        db_path = args.db_path or ""
        db_path_glob = args.db_path_glob
        db_dir = args.db_dir or "/mnt/data"  # Default to /mnt/data
        
        # If db_dir specified and no explicit db_path/db_path_glob, use db_dir
        if db_dir and not db_path and not db_path_glob:
            db_path = db_dir
        
        return mode_sqlite_search_chain(
            db_path,
            phrase_list,
            args.scope,
            cat,
            tag,
            fmt,
            summary_only=bool(args.summary_only),
            dedupe_by_id=bool(args.dedupe_by_id),
            limit=args.limit,
            offset=args.offset,
            query_str=query_str,
            highlight=bool(args.highlight),
            sort_by=args.sort,
            use_regex=bool(args.regex),
            proximity=args.proximity,
            context_lines=args.context,
            group_by=args.group_by or "none",
            min_length=args.min_length,
            max_length=args.max_length,
            export_html=args.export_html,
            db_path_glob=db_path_glob,
        )
    if args.mode == "sqlite_validate":
        return mode_sqlite_validate_chain(idx, args.db_path or "", fmt)
    if args.mode == "sqlite_export_segment":
        return mode_sqlite_export_segment(
            db_path=args.db_path or "",
            seg_id=str(args.segment_id or ""),
            out_path=str(args.output_path or ""),
        )
    if args.mode == "validate":
        return mode_validate(idx, lines, fmt)
    if args.mode == "tree":
        return mode_tree(idx, fmt)

    import sys

    print("Unknown mode:", args.mode, file=sys.stderr)
    return 1
