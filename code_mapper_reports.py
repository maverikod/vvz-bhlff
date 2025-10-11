#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Report generation functionality for code mapping in BHLFF project.

This module provides report generation capabilities for the code mapper,
including text and YAML report formats.
"""

import json
import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CodeMapperReports:
    """Report generation functionality for code mapper."""

    def __init__(self, code_map: Dict[str, Any], issues: Dict[str, Any]):
        """Initialize report generator."""
        self.code_map = code_map
        self.issues = issues

    def generate_report(self) -> str:
        """Generate comprehensive text report."""
        report = []
        report.append("=" * 80)
        report.append("BHLFF CODE MAP REPORT")
        report.append("=" * 80)
        report.append("")

        # Files summary
        report.append("FILES SUMMARY:")
        report.append("-" * 40)
        report.append(f"Total files analyzed: {len(self.code_map['files'])}")
        report.append("")

        # Classes summary
        report.append("CLASSES SUMMARY:")
        report.append("-" * 40)
        report.append(f"Total classes: {len(self.code_map['classes'])}")
        report.append("")

        # Functions summary
        report.append("FUNCTIONS SUMMARY:")
        report.append("-" * 40)
        report.append(f"Total functions: {len(self.code_map['functions'])}")
        report.append("")

        # Issues summary
        report.append("ISSUES SUMMARY:")
        report.append("-" * 40)
        for issue_type, issues_list in self.issues.items():
            report.append(f"{issue_type}: {len(issues_list)}")
        report.append("")

        # Detailed issues
        for issue_type, issues_list in self.issues.items():
            if issues_list:
                report.append(f"{issue_type.upper()}:")
                report.append("-" * 40)
                for issue in issues_list[:10]:  # Show first 10 issues
                    if isinstance(issue, dict):
                        if 'file' in issue:
                            report.append(f"  File: {issue['file']}")
                        if 'line' in issue:
                            report.append(f"  Line: {issue['line']}")
                        if 'class' in issue and issue['class']:
                            report.append(f"  Class: {issue['class']}")
                        if 'method' in issue:
                            report.append(f"  Method: {issue['method']}")
                        report.append("")
                if len(issues_list) > 10:
                    report.append(f"  ... and {len(issues_list) - 10} more")
                report.append("")

        return "\n".join(report)

    def generate_issues_report(self) -> str:
        """Generate issues-focused report."""
        report = []
        report.append("=" * 80)
        report.append("BHLFF CODE ISSUES REPORT")
        report.append("=" * 80)
        report.append("")

        total_issues = sum(len(issues) for issues in self.issues.values())
        report.append(f"Total issues found: {total_issues}")
        report.append("")

        for issue_type, issues_list in self.issues.items():
            if issues_list:
                report.append(f"{issue_type.upper()}: {len(issues_list)} issues")
                report.append("-" * 40)
                for issue in issues_list:
                    if isinstance(issue, dict):
                        if 'file' in issue:
                            report.append(f"  {issue['file']}")
                        if 'line' in issue:
                            report.append(f"    Line {issue['line']}")
                        if 'class' in issue and issue['class']:
                            report.append(f"    Class: {issue['class']}")
                        if 'method' in issue:
                            report.append(f"    Method: {issue['method']}")
                        report.append("")
                report.append("")

        return "\n".join(report)

    def save_report(self, output_dir: str) -> None:
        """Save text report to file."""
        report_content = self.generate_report()
        with open(f"{output_dir}/code_map_report.txt", "w", encoding="utf-8") as f:
            f.write(report_content)

    def generate_yaml_code_map(self) -> str:
        """Generate YAML code map."""
        yaml_data = {
            "code_map": {
                "files": self.code_map["files"],
                "classes": self.code_map["classes"],
                "functions": self.code_map["functions"],
                "imports": self.code_map["imports"],
                "dependencies": self.code_map["dependencies"],
            },
            "metadata": {
                "total_files": len(self.code_map["files"]),
                "total_classes": len(self.code_map["classes"]),
                "total_functions": len(self.code_map["functions"]),
            }
        }
        return yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True, sort_keys=True)

    def save_yaml_code_map(self, output_dir: str) -> None:
        """Save YAML code map to file."""
        yaml_content = self.generate_yaml_code_map()
        with open(f"{output_dir}/code_map.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)

    def save_issues_report(self, output_dir: str) -> None:
        """Save issues report to file."""
        issues_content = self.generate_issues_report()
        with open(f"{output_dir}/issues_report.txt", "w", encoding="utf-8") as f:
            f.write(issues_content)

    def generate_yaml_issues_report(self) -> str:
        """Generate YAML issues report."""
        yaml_data = {
            "issues": self.issues,
            "summary": {
                issue_type: len(issues_list) 
                for issue_type, issues_list in self.issues.items()
            }
        }
        return yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True, sort_keys=True)

    def save_yaml_issues_report(self, output_dir: str) -> None:
        """Save YAML issues report to file."""
        yaml_content = self.generate_yaml_issues_report()
        with open(f"{output_dir}/code_issues.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)
