"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Index + theory text loading utilities.

Supports:
- Loading `ALL_index.yaml` with a lightweight pickle cache.
- Loading theory text from:
  - a single aggregated markdown file, or
  - a manifest file containing a list of part filenames (one per line),
    where part files are concatenated in order.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from .index_data import IndexData
from .segment import Segment


def build_index(raw: Dict[str, Any]) -> IndexData:
    segs: List[Segment] = []
    for d in raw.get("segments", []):
        if isinstance(d, dict):
            try:
                segs.append(Segment.from_dict(d))
            except Exception:
                continue
    return IndexData(segs, raw)


def load_index(path: str, use_cache: bool = True) -> IndexData:
    p = Path(path).resolve()
    cache_path = str(p) + ".pkl"

    mtime = os.path.getmtime(p)
    size = os.path.getsize(p)

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("mtime") == mtime and payload.get("size") == size:
                return build_index(payload["data"])
        except Exception:
            pass

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    if use_cache:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"mtime": mtime, "size": size, "data": raw}, f)
        except Exception:
            pass

    return build_index(raw)


def _is_sqlite_file(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            sig = f.read(16)
        return sig.startswith(b"SQLite format 3")
    except Exception:
        return False


def _load_manifest_paths(manifest_path: Path) -> List[Path]:
    base_dir = manifest_path.parent
    lines = [
        ln.strip() for ln in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    files: List[Path] = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        files.append((base_dir / ln).resolve())
    return files


def load_theory_lines(path: str) -> List[str]:
    """
    Load theory lines from:
    - a single markdown file, or
    - a manifest (list of part files).

    Manifest is detected heuristically:
    - file is not SQLite;
    - file extension is not `.md` OR first non-empty line ends with `.md`.
    """

    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    if _is_sqlite_file(p):
        raise ValueError(f"--theory must be markdown or manifest, got SQLite: {p}")

    text = p.read_text(encoding="utf-8")
    first_non_empty: Optional[str] = None
    for ln in text.splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            first_non_empty = s
            break

    looks_like_manifest = False
    if p.suffix.lower() != ".md":
        looks_like_manifest = True
    if first_non_empty and first_non_empty.lower().endswith(".md"):
        looks_like_manifest = True

    if not looks_like_manifest:
        return text.splitlines(True)

    parts = _load_manifest_paths(p)
    out: List[str] = []
    for part in parts:
        out.extend(part.read_text(encoding="utf-8").splitlines(True))
    return out


def resolve_db_paths(db_path: str) -> List[Path]:
    """
    Resolve `--db-path` into a list of sqlite files.

    Supported:
    - a single `.sqlite` file;
    - a directory (all `*.sqlite` inside, sorted);
    - a manifest text file listing `.sqlite` filenames (one per line).
    """

    p = Path(db_path).resolve()
    if p.is_dir():
        return sorted(p.glob("*.sqlite"))
    if p.is_file() and _is_sqlite_file(p):
        return [p]
    if p.is_file():
        return _load_manifest_paths(p)
    return []


def segment_to_dict(seg: Segment) -> Dict[str, Any]:
    return {
        "id": seg.id,
        "category": seg.category,
        "keywords": seg.keywords,
        "summary": seg.summary,
        "start_line": seg.start_line,
        "end_line": seg.end_line,
        "ranges": [{"start_line": a, "end_line": b} for (a, b) in seg.ranges],
    }
