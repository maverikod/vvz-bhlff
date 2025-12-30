"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Regex search functionality for SQLite database.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


def _has_table(cur: sqlite3.Cursor, name: str) -> bool:
    """Check if table exists."""
    try:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return cur.fetchone() is not None
    except Exception:
        return False


def regex_search_one(
    db_path: Path,
    pattern: str,
    scope: str,
    category: Optional[str],
    tag: Optional[str],
) -> List[Dict[str, str]]:
    """
    Search using regular expression.

    Args:
        db_path: Path to SQLite database
        pattern: Regular expression pattern
        scope: Search scope (segments or formulas)
        category: Optional category filter
        tag: Optional tag filter

    Returns:
        List of matching results
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE | re.UNICODE)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}") from e

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    has_segments_fts = _has_table(cur, "segments_fts")
    has_formulas_fts = _has_table(cur, "formulas_fts")

    results: List[Dict[str, str]] = []

    if scope == "formulas":
        # Get all formulas and filter by regex
        rows = cur.execute(
            "SELECT segment_id, text FROM formulas LIMIT 1000"
        ).fetchall()
        for seg_id, text in rows:
            if regex.search(text or ""):
                results.append(
                    {"id": str(seg_id), "formula": str(text), "db": db_path.name}
                )
    else:
        # Get all segments and filter by regex
        rows = cur.execute(
            "SELECT id, category, summary, text FROM segments"
        ).fetchall()
        for seg_id, cat, summary, text in rows:
            sid = str(seg_id)
            if tag and tag.lower() not in sid.lower():
                continue
            if category and category.lower() not in (str(cat) or "").lower():
                continue
            # Check if regex matches text or summary
            full_text = f"{summary or ''} {text or ''}"
            if regex.search(full_text):
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

