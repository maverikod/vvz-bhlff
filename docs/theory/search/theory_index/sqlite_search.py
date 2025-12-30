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
from .proximity_search import filter_by_proximity
from .query_executor import execute_query
from .query_parser import extract_terms, parse_query
from .regex_search import regex_search_one
from .result_formatter import format_results
from .result_ranking import rank_results


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

    def _like_escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    if scope == "formulas":
        q_like = f"%{_like_escape(q)}%"
        rows = []
        used_fts = False
        if has_formulas_fts and ("\\" not in q):
            try:
                rows = cur.execute(
                    "SELECT segment_id, text FROM formulas_fts "
                    "WHERE formulas_fts MATCH ? LIMIT 200",
                    (q,),
                ).fetchall()
                used_fts = True
            except sqlite3.OperationalError:
                rows = []
                used_fts = False

        if (not used_fts) or (not rows):
            rows = cur.execute(
                "SELECT segment_id, text FROM formulas "
                "WHERE text LIKE ? ESCAPE '\\' LIMIT 200",
                (q_like,),
            ).fetchall()
        for seg_id, text in rows:
            results.append(
                {"id": str(seg_id), "formula": str(text), "db": db_path.name}
            )
        conn.close()
        return results

    # scope == segments
    q_like = f"%{_like_escape(q)}%"
    rows = []
    used_fts = False
    if has_segments_fts and ("\\" not in q):
        try:
            rows = cur.execute(
                "SELECT id, category, summary, text FROM segments_fts "
                "WHERE segments_fts MATCH ?",
                (q,),
            ).fetchall()
            used_fts = True
        except sqlite3.OperationalError:
            rows = []
            used_fts = False

    if (not used_fts) or (not rows):
        rows = cur.execute(
            "SELECT id, category, summary, text FROM segments "
            "WHERE text LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\'",
            (q_like, q_like),
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
    phrases: List[str],
    scope: str,
    category: Optional[str],
    tag: Optional[str],
    fmt: str,
    summary_only: bool = False,
    dedupe_by_id: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
    query_str: Optional[str] = None,
    highlight: bool = False,
    sort_by: str = "none",
    use_regex: bool = False,
    proximity: Optional[int] = None,
    context_lines: Optional[int] = None,
) -> int:
    if not db_path:
        print("ERROR: --db-path is required for sqlite_search mode.", file=sys.stderr)
        return 1

    dbs = resolve_db_paths(db_path)
    if not dbs:
        print(f"ERROR: no sqlite db files resolved from: {db_path}", file=sys.stderr)
        return 1

    all_results: List[Dict[str, str]] = []

    # Handle regex search
    if use_regex and phrases:
        try:
            for db in dbs:
                for phr in phrases:
                    all_results.extend(
                        regex_search_one(db, phr, scope, category, tag)
                    )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    # Handle logical query
    elif query_str:
        try:
            query_node = parse_query(query_str)
            terms = extract_terms(query_node)
            if not terms:
                print(
                    "ERROR: no terms found in query.",
                    file=sys.stderr,
                )
                return 1
            # Search for each term
            term_results: Dict[str, List[Dict[str, str]]] = {}
            for term in terms:
                term_results[term] = []
                for db in dbs:
                    term_results[term].extend(
                        _sqlite_search_one(db, term, scope, category, tag)
                    )
            # Execute logical query
            all_results = execute_query(query_node, term_results)
        except Exception as e:
            print(f"ERROR: query parsing failed: {e}", file=sys.stderr)
            return 1
    else:
        # Original behavior: OR of phrases
        norm_phrases = [p.strip() for p in (phrases or []) if p and p.strip()]
        if not norm_phrases:
            print(
                "ERROR: --phrase, --phrases, or --query is required for sqlite_search mode.",
                file=sys.stderr,
            )
            return 1
        for db in dbs:
            for phr in norm_phrases:
                all_results.extend(_sqlite_search_one(db, phr, scope, category, tag))

    if dedupe_by_id and scope == "segments":
        seen: Dict[str, Dict[str, str]] = {}
        for r in all_results:
            sid = r.get("id") or ""
            if not sid:
                continue
            if sid not in seen:
                seen[sid] = r
        all_results = sorted(
            seen.values(), key=lambda r: (r.get("id", ""), r.get("db", ""))
        )

    # Apply proximity filter
    if proximity is not None and proximity > 0:
        proximity_phrases: List[str] = []
        if query_str:
            try:
                query_node = parse_query(query_str)
                proximity_phrases = extract_terms(query_node)
            except Exception:
                pass
        if not proximity_phrases:
            proximity_phrases = phrases or []
        if len(proximity_phrases) >= 2:
            all_results = filter_by_proximity(
                all_results, proximity_phrases, proximity, scope
            )

    # Apply ranking/sorting
    if sort_by == "relevance":
        # Extract phrases for ranking
        rank_phrases: List[str] = []
        if query_str:
            try:
                query_node = parse_query(query_str)
                rank_phrases = extract_terms(query_node)
            except Exception:
                pass
        if not rank_phrases:
            rank_phrases = phrases or []
        if rank_phrases:
            all_results = rank_results(all_results, rank_phrases, scope)
    elif sort_by == "id":
        all_results = sorted(
            all_results, key=lambda r: (r.get("id", ""), r.get("db", ""))
        )

    # Apply pagination
    if offset > 0:
        all_results = all_results[offset:]
    if limit is not None and limit > 0:
        all_results = all_results[:limit]

    if fmt == "json":
        import json

        print(json.dumps(all_results, ensure_ascii=False, indent=2))
        return 0

    if not all_results:
        print("No matches in SQLite search.", file=sys.stderr)
        return 0

    # Extract phrases for highlighting
    highlight_phrases_list: List[str] = []
    if highlight:
        if query_str:
            try:
                query_node = parse_query(query_str)
                highlight_phrases_list = extract_terms(query_node)
            except Exception:
                pass
        else:
            highlight_phrases_list = phrases or []

    # Format and print results
    context_lines_val = context_lines if context_lines is not None else 0
    if highlight and highlight_phrases_list:
        format_results(
            all_results,
            highlight_phrases_list,
            scope,
            summary_only,
            highlight,
            context_lines_val,
        )
    else:
        try:
            for r in all_results:
                if scope == "segments":
                    print(
                        f"[{r['id']}] ({r['db']}) {r.get('category', '')} :: "
                        f"{r.get('summary', '')}"
                    )
                    if not summary_only:
                        print((r.get("snippet") or "").rstrip())
                        print("----")
                else:
                    print(f"[{r['id']}] ({r['db']}) :: {r.get('formula', '')}")
        except BrokenPipeError:
            return 0
    return 0
