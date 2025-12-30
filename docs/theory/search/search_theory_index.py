#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Theory index/search tool (stable CLI entrypoint).

This file intentionally stays small and stable. The implementation is located
in `docs/theory/search/theory_index/`.

Usage:
    python3 docs/theory/search/search_theory_index.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    """
    Ensure `docs/theory/search/` is on sys.path so `theory_index` package
    can be imported when running this script directly.
    """

    base = Path(__file__).resolve().parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))


def main() -> int:
    _bootstrap_import_path()
    from theory_index.cli import main as impl_main

    return impl_main()


if __name__ == "__main__":
    raise SystemExit(main())
