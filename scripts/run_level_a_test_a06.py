"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Standalone wrapper for Level A quench detection.

This script delegates to the `bhlff quench-detect` CLI command, allowing the
same functionality to be executed directly via Python without installing the
console entry point.

Example:
    python scripts/run_level_a_test_a06.py --verbose
"""

from __future__ import annotations

import sys

from bhlff.cli.quench_detect import main as quench_detect_main


def main(argv: list[str] | None = None) -> int:
    """
    Delegate execution to CLI quench detection command.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        int: Exit code returned by CLI command.
    """
    return quench_detect_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
