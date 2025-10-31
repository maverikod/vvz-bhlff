#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Main entry point for code mapping in BHLFF project.

This module provides the main interface for the code mapper,
combining all functionality into a unified interface.
"""

import argparse
import logging
from pathlib import Path

from code_mapper_core import CodeMapperCore
from code_mapper_reports import CodeMapperReports
from code_mapper_methods import CodeMapperMethods

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeMapper:
    """Main code mapper class that combines all functionality."""

    def __init__(
        self,
        root_dir: str = ".",
        output_dir: str = "code_analysis",
        max_lines: int = 400,
    ):
        """Initialize code mapper."""
        self.core = CodeMapperCore(root_dir, output_dir, max_lines)
        self.reports = None
        self.methods = None

    def scan_directory(self, directory: str = ".") -> None:
        """Scan directory for Python files."""
        self.core.scan_directory(directory)

        # Initialize reports and methods after scanning
        self.reports = CodeMapperReports(self.core.code_map, self.core.issues)
        self.methods = CodeMapperMethods(self.core.code_map)

    def generate_all_reports(self) -> None:
        """Generate all reports."""
        if not self.reports or not self.methods:
            raise RuntimeError("Must scan directory first")

        # Generate and save all reports
        self.reports.save_report(str(self.core.output_dir))
        self.reports.save_issues_report(str(self.core.output_dir))
        self.reports.save_yaml_code_map(str(self.core.output_dir))
        self.reports.save_yaml_issues_report(str(self.core.output_dir))

        self.methods.save_method_index(str(self.core.output_dir))
        self.methods.save_yaml_method_index(str(self.core.output_dir))

    def print_summary(self) -> None:
        """Print analysis summary."""
        total_files = len(self.core.code_map["files"])
        total_classes = len(self.core.code_map["classes"])
        total_functions = len(self.core.code_map["functions"])
        total_issues = sum(len(issues) for issues in self.core.issues.values())

        print(f"\nАнализ завершен!")
        print(f"Созданы YAML отчеты в каталоге: {self.core.output_dir}")
        print(f"- code_map.yaml - карта кода")
        print(f"- code_issues.yaml - проблемы в коде")
        print(f"- method_index.yaml - индекс методов")
        print(f"Лимит строк на файл: {self.core.max_lines}")
        print(f"Всего найдено проблем: {total_issues}")

        # Files exceeding limit
        files_too_large = self.core.issues["files_too_large"]
        if files_too_large:
            print(f"Файлов, превышающих лимит: {len(files_too_large)}")
            for file_info in files_too_large[:10]:  # Show first 10
                print(
                    f"  - {file_info['file']}: {file_info['lines']} строк (превышение на {file_info['exceeds_limit']})"
                )
            if len(files_too_large) > 10:
                print(f"  ... и еще {len(files_too_large) - 10} файлов")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BHLFF Code Mapper")
    parser.add_argument("--root-dir", default=".", help="Root directory to scan")
    parser.add_argument(
        "--output-dir", default="code_analysis", help="Output directory for reports"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=400,
        help="Maximum lines per file (default: 400)",
    )

    args = parser.parse_args()

    # Create code mapper
    mapper = CodeMapper(args.root_dir, args.output_dir, args.max_lines)

    # Scan directory
    mapper.scan_directory(args.root_dir)

    # Generate all reports
    mapper.generate_all_reports()

    # Print summary
    mapper.print_summary()


if __name__ == "__main__":
    main()
