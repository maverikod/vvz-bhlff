#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Method indexing functionality for code mapping in BHLFF project.

This module provides method indexing capabilities for the code mapper,
including method signature extraction and indexing.
"""

import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CodeMapperMethods:
    """Method indexing functionality for code mapper."""

    def __init__(self, code_map: Dict[str, Any]):
        """Initialize method indexer."""
        self.code_map = code_map

    def generate_method_index(self) -> str:
        """Generate method index report."""
        report = []
        report.append("=" * 80)
        report.append("BHLFF METHOD INDEX")
        report.append("=" * 80)
        report.append("")

        # Group methods by class
        class_methods = {}
        standalone_functions = []

        for func_key, func_info in self.code_map["functions"].items():
            if "::" in func_key:
                file_path, method_name = func_key.split("::", 1)
                # This is a method, find its class
                for class_key, class_info in self.code_map["classes"].items():
                    if class_key.startswith(file_path + "::"):
                        class_name = class_key.split("::", 1)[1]
                        if class_name not in class_methods:
                            class_methods[class_name] = []
                        class_methods[class_name].append(
                            {
                                "name": method_name,
                                "file": func_info["file"],
                                "line": func_info["line"],
                                "args": func_info["args"],
                                "returns": func_info["returns"],
                            }
                        )
                        break
            else:
                # Standalone function
                standalone_functions.append(
                    {
                        "name": func_info["name"],
                        "file": func_info["file"],
                        "line": func_info["line"],
                        "args": func_info["args"],
                        "returns": func_info["returns"],
                    }
                )

        # Report class methods
        if class_methods:
            report.append("CLASS METHODS:")
            report.append("-" * 40)
            for class_name, methods in class_methods.items():
                report.append(f"Class: {class_name}")
                for method in methods:
                    args_str = ", ".join(method["args"]) if method["args"] else "()"
                    returns_str = (
                        f" -> {method['returns']}" if method["returns"] else ""
                    )
                    report.append(f"  {method['name']}({args_str}){returns_str}")
                    report.append(f"    File: {method['file']}, Line: {method['line']}")
                report.append("")

        # Report standalone functions
        if standalone_functions:
            report.append("STANDALONE FUNCTIONS:")
            report.append("-" * 40)
            for func in standalone_functions:
                args_str = ", ".join(func["args"]) if func["args"] else "()"
                returns_str = f" -> {func['returns']}" if func["returns"] else ""
                report.append(f"{func['name']}({args_str}){returns_str}")
                report.append(f"  File: {func['file']}, Line: {func['line']}")
                report.append("")

        return "\n".join(report)

    def save_method_index(self, output_dir: str) -> None:
        """Save method index to file."""
        method_content = self.generate_method_index()
        with open(f"{output_dir}/method_index.txt", "w", encoding="utf-8") as f:
            f.write(method_content)

    def generate_yaml_method_index(self) -> str:
        """Generate YAML method index."""
        # Group methods by class
        class_methods = {}
        standalone_functions = []

        for func_key, func_info in self.code_map["functions"].items():
            if "::" in func_key:
                file_path, method_name = func_key.split("::", 1)
                # This is a method, find its class
                for class_key, class_info in self.code_map["classes"].items():
                    if class_key.startswith(file_path + "::"):
                        class_name = class_key.split("::", 1)[1]
                        if class_name not in class_methods:
                            class_methods[class_name] = []
                        class_methods[class_name].append(
                            {
                                "name": method_name,
                                "file": func_info["file"],
                                "line": func_info["line"],
                                "args": func_info["args"],
                                "returns": func_info["returns"],
                            }
                        )
                        break
            else:
                # Standalone function
                standalone_functions.append(
                    {
                        "name": func_info["name"],
                        "file": func_info["file"],
                        "line": func_info["line"],
                        "args": func_info["args"],
                        "returns": func_info["returns"],
                    }
                )

        yaml_data = {
            "method_index": {
                "class_methods": class_methods,
                "standalone_functions": standalone_functions,
                "summary": {
                    "total_class_methods": sum(
                        len(methods) for methods in class_methods.values()
                    ),
                    "total_standalone_functions": len(standalone_functions),
                    "total_classes": len(class_methods),
                },
            }
        }
        return yaml.dump(
            yaml_data, default_flow_style=False, allow_unicode=True, sort_keys=True
        )

    def save_yaml_method_index(self, output_dir: str) -> None:
        """Save YAML method index to file."""
        yaml_content = self.generate_yaml_method_index()
        with open(f"{output_dir}/method_index.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)
