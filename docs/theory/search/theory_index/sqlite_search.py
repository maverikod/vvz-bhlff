"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

SQLite search across a single DB or a chain of DB shards.

`--db-path` can point to:
- a single `.sqlite` file
- a directory containing `.sqlite` shards
- a manifest file listing `.sqlite` shards (one per line)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .index_io import resolve_db_paths


def _has_table(cur: sqlite3.Cursor, name: str) -> bool:
    try:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return cur.fetchone() is not None
    except Exception:
        return False


def _sqlite_search_one(
    db_path: Path,
    phrase: str,
    scope: str,
    category: Optional[str],
    tag: Optional[str],
) -> List[Dict[str, str]]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    has_segments_fts = _has_table(cur, "segments_fts")
    has_formulas_fts = _has_table(cur, "formulas_fts")

    q = (phrase or "").strip()
    if not q:
        conn.close()
        return []

    results: List[Dict[str, str]] = []

    if scope == "formulas":
        if has_formulas_fts:
            rows = cur.execute(
                "SELECT segment_id, text FROM formulas_fts "
                "WHERE formulas_fts MATCH ? LIMIT 200",
                (q,),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT segment_id, text FROM formulas WHERE text LIKE ? LIMIT 200",
                (f"%{q}%",),
            ).fetchall()
        for seg_id, text in rows:
            results.append(
                {"id": str(seg_id), "formula": str(text), "db": db_path.name}
            )
        conn.close()
        return results

    # scope == segments
    if has_segments_fts:
        rows = cur.execute(
            "SELECT id, category, summary, text FROM segments_fts "
            "WHERE segments_fts MATCH ?",
            (q,),
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT id, category, summary, text FROM segments "
            "WHERE text LIKE ? OR summary LIKE ?",
            (f"%{q}%", f"%{q}%"),
        ).fetchall()

    for seg_id, cat, summary, text in rows:
        sid = str(seg_id)
        if tag and tag.lower() not in sid.lower():
            continue
        if category and category.lower() not in (str(cat) or "").lower():
            continue
        results.append(
            {
                "id": sid,
                "category": str(cat or ""),
                "summary": str(summary or ""),
                "snippet": str(text or "")[:2000],
                "db": db_path.name,
            }
        )

    conn.close()
    return results


def mode_sqlite_search_chain(
    db_path: str,
    phrase: Optional[str],
    scope: str,
    category: Optional[str],
    tag: Optional[str],
    fmt: str,
) -> int:
    if not db_path:
        print("ERROR: --db-path is required for sqlite_search mode.", file=sys.stderr)
        return 1
    phr = (phrase or "").strip()
    if not phr:
        print("ERROR: --phrase is required for sqlite_search mode.", file=sys.stderr)
        return 1

    dbs = resolve_db_paths(db_path)
    if not dbs:
        print(f"ERROR: no sqlite db files resolved from: {db_path}", file=sys.stderr)
        return 1

    all_results: List[Dict[str, str]] = []
    for db in dbs:
        all_results.extend(_sqlite_search_one(db, phr, scope, category, tag))

    if fmt == "json":
        import json

        print(json.dumps(all_results, ensure_ascii=False, indent=2))
        return 0

    if not all_results:
        print("No matches in SQLite search.", file=sys.stderr)
        return 0

    try:
        for r in all_results:
            if scope == "segments":
                print(
                    f"[{r['id']}] ({r['db']}) "
                    f"{r.get('category', '')} :: {r.get('summary', '')}"
                )
                print((r.get("snippet") or "").rstrip())
                print("----")
            else:
                print(f"[{r['id']}] ({r['db']}) :: {r.get('formula', '')}")
    except BrokenPipeError:
        return 0
    return 0
