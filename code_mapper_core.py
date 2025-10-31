#!/usr/bin/env python3
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core functionality for code mapping in BHLFF project.

This module provides the core CodeMapper class with basic functionality
for analyzing Python codebase structure.
"""

import os
import ast
from typing import Dict, List, Any, Set
from pathlib import Path
import logging

# Setup logging
logger = logging.getLogger(__name__)


class CodeMapperCore:
    """Core code mapper functionality for analyzing Python codebase."""

    def __init__(
        self,
        root_dir: str = ".",
        output_dir: str = "code_analysis",
        max_lines: int = 400,
    ):
        """Initialize code mapper core."""
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.max_lines = max_lines

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)

        self.code_map = {
            "files": {},
            "classes": {},
            "functions": {},
            "imports": {},
            "dependencies": {},
        }
        # Issues tracking
        self.issues = {
            "methods_with_pass": [],
            "not_implemented_in_non_abstract": [],
            "methods_without_docstrings": [],
            "files_without_docstrings": [],
            "classes_without_docstrings": [],
            "imports_in_middle": [],
            "files_too_large": [],
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

            # Check file docstring
            file_docstring = ast.get_docstring(tree)
            if not file_docstring:
                self.issues["files_without_docstrings"].append(
                    {"file": file_path, "line": 1}
                )

            # Check file size limit
            if file_info["lines"] > self.max_lines:
                self.issues["files_too_large"].append(
                    {
                        "file": file_path,
                        "lines": file_info["lines"],
                        "exceeds_limit": file_info["lines"] - self.max_lines,
                        "limit": self.max_lines,
                    }
                )

            # Track import lines for checking imports in middle
            import_lines = []

            # Process AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self.extract_class_info(node, file_path, import_lines)
                    file_info["classes"].append(class_info)
                    self.code_map["classes"][f"{file_path}::{node.name}"] = class_info

                elif isinstance(node, ast.FunctionDef):
                    function_info = self.extract_function_info(
                        node, file_path, import_lines
                    )
                    file_info["functions"].append(function_info)
                    self.code_map["functions"][
                        f"{file_path}::{node.name}"
                    ] = function_info

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self.extract_import_info(node)
                    file_info["imports"].append(import_info)
                    import_lines.append(node.lineno)

            # Check for imports in middle of file
            if import_lines:
                max_import_line = max(import_lines)
                if (
                    max_import_line > 20
                ):  # Consider imports after line 20 as "in middle"
                    self.issues["imports_in_middle"].append(
                        {"file": file_path, "line": max_import_line}
                    )

            # Store file information
            self.code_map["files"][file_path] = file_info

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")

    def extract_class_info(
        self, node: ast.ClassDef, file_path: str, import_lines: List[int]
    ) -> Dict[str, Any]:
        """Extract class information from AST node."""
        class_info = {
            "name": node.name,
            "file": file_path,
            "line": node.lineno,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "bases": [
                base.id if isinstance(base, ast.Name) else str(base)
                for base in node.bases
            ],
        }

        # Check if class has docstring
        if not class_info["docstring"]:
            self.issues["classes_without_docstrings"].append(
                {"class": node.name, "file": file_path, "line": node.lineno}
            )

        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self.extract_function_info(item, file_path, import_lines)
                class_info["methods"].append(method_info)

                # Check method issues
                self._check_method_issues(item, file_path, node.name)

        return class_info

    def extract_function_info(
        self, node: ast.FunctionDef, file_path: str, import_lines: List[int]
    ) -> Dict[str, Any]:
        """Extract function information from AST node."""
        function_info = {
            "name": node.name,
            "file": file_path,
            "line": node.lineno,
            "docstring": ast.get_docstring(node),
            "args": [arg.arg for arg in node.args.args],
            "returns": ast.unparse(node.returns) if node.returns else None,
        }

        # Check if function has docstring
        if not function_info["docstring"]:
            self.issues["methods_without_docstrings"].append(
                {
                    "class": None,
                    "file": file_path,
                    "line": node.lineno,
                    "method": node.name,
                }
            )

        return function_info

    def _check_method_issues(
        self, node: ast.FunctionDef, file_path: str, class_name: str = None
    ) -> None:
        """Check for common method issues."""
        # Check for methods with only pass
        if self._has_only_pass(node):
            self.issues["methods_with_pass"].append(
                {
                    "class": class_name,
                    "file": file_path,
                    "line": node.lineno,
                    "method": node.name,
                }
            )

        # Check for NotImplemented in non-abstract methods
        if self._has_not_implemented(node) and not self._is_abstract_method(node):
            self.issues["not_implemented_in_non_abstract"].append(
                {
                    "class": class_name,
                    "file": file_path,
                    "line": node.lineno,
                    "method": node.name,
                }
            )

        # Check for missing docstrings
        if not ast.get_docstring(node):
            self.issues["methods_without_docstrings"].append(
                {
                    "class": class_name,
                    "file": file_path,
                    "line": node.lineno,
                    "method": node.name,
                }
            )

    def _has_only_pass(self, node: ast.FunctionDef) -> bool:
        """Check if function has only pass statement."""
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            return True
        return False

    def _has_not_implemented(self, node: ast.FunctionDef) -> bool:
        """Check if function contains NotImplemented."""
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Raise)
                and isinstance(stmt.exc, ast.Name)
                and stmt.exc.id == "NotImplementedError"
            ):
                return True
        return False

    def _is_abstract_method(self, node: ast.FunctionDef) -> bool:
        """Check if function is abstract method."""
        return any(
            isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
            for decorator in node.decorator_list
        )

    def extract_import_info(self, node: ast.Import) -> Dict[str, Any]:
        """Extract import information from AST node."""
        return {
            "line": node.lineno,
            "names": [alias.name for alias in node.names],
            "type": "import",
        }
