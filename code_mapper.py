#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Code mapper for BHLFF project.

This script analyzes the codebase and generates a comprehensive
code map with method signatures, class hierarchies, and dependencies.
"""

import os
import ast
import json
from typing import Dict, List, Any, Set
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeMapper:
    """Code mapper for analyzing Python codebase."""

    def __init__(self, root_dir: str = "."):
        """Initialize code mapper."""
        self.root_dir = Path(root_dir)
        self.code_map = {
            "files": {},
            "classes": {},
            "functions": {},
            "imports": {},
            "dependencies": {},
        }

    def scan_directory(self, directory: str = ".") -> None:
        """Scan directory for Python files."""
        logger.info(f"Сканирование директории: {directory}")

        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    logger.info(f"Анализ файла: {file_path}")
                    self.analyze_file(file_path)

    def analyze_file(self, file_path: str) -> None:
        """Analyze single Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)

            # Extract file information
            file_info = {
                "path": file_path,
                "classes": [],
                "functions": [],
                "imports": [],
                "lines": len(content.splitlines()),
            }

            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self.extract_class_info(node)
                    file_info["classes"].append(class_info)
                    self.code_map["classes"][f"{file_path}::{node.name}"] = class_info

                elif isinstance(node, ast.FunctionDef):
                    func_info = self.extract_function_info(node)
                    file_info["functions"].append(func_info)
                    self.code_map["functions"][f"{file_path}::{node.name}"] = func_info

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self.extract_import_info(node)
                    file_info["imports"].append(import_info)
                    self.code_map["imports"][
                        f"{file_path}::{import_info['name']}"
                    ] = import_info

            self.code_map["files"][file_path] = file_info

        except Exception as e:
            logger.error(f"Ошибка при анализе файла {file_path}: {e}")

    def extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class information from AST node."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(
                    {
                        "name": item.name,
                        "args": [arg.arg for arg in item.args.args],
                        "returns": (
                            getattr(item.returns, "id", None) if item.returns else None
                        ),
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                    }
                )

        return {
            "name": node.name,
            "bases": [
                base.id if isinstance(base, ast.Name) else str(base)
                for base in node.bases
            ],
            "methods": methods,
            "docstring": ast.get_docstring(node),
            "decorators": [
                d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list
            ],
        }

    def extract_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function information from AST node."""
        return {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "returns": getattr(node.returns, "id", None) if node.returns else None,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "docstring": ast.get_docstring(node),
            "decorators": [
                d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list
            ],
        }

    def extract_import_info(self, node: ast.Import) -> Dict[str, Any]:
        """Extract import information from AST node."""
        if isinstance(node, ast.Import):
            return {
                "type": "import",
                "name": node.names[0].name,
                "alias": node.names[0].asname,
            }
        elif isinstance(node, ast.ImportFrom):
            return {
                "type": "from_import",
                "module": node.module,
                "name": node.names[0].name,
                "alias": node.names[0].asname,
            }

    def generate_report(self) -> str:
        """Generate code map report."""
        report = []
        report.append("# Code Map Report")
        report.append(f"Generated for: {self.root_dir}")
        report.append("")

        # File statistics
        total_files = len(self.code_map["files"])
        total_classes = len(self.code_map["classes"])
        total_functions = len(self.code_map["functions"])
        total_imports = len(self.code_map["imports"])

        report.append("## Statistics")
        report.append(f"- Total files: {total_files}")
        report.append(f"- Total classes: {total_classes}")
        report.append(f"- Total functions: {total_functions}")
        report.append(f"- Total imports: {total_imports}")
        report.append("")

        # Files overview
        report.append("## Files Overview")
        for file_path, file_info in self.code_map["files"].items():
            report.append(f"### {file_path}")
            report.append(f"- Lines: {file_info['lines']}")
            report.append(f"- Classes: {len(file_info['classes'])}")
            report.append(f"- Functions: {len(file_info['functions'])}")
            report.append(f"- Imports: {len(file_info['imports'])}")
            report.append("")

        return "\n".join(report)

    def save_report(self) -> None:
        """Save code map report to file."""
        report = self.generate_report()
        with open("code_map.md", "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Отчет сохранен в файл: code_map.md")


def main():
    """Main function."""
    mapper = CodeMapper()
    mapper.scan_directory(".")
    mapper.save_report()
    print("Анализ завершен!")


if __name__ == "__main__":
    main()

