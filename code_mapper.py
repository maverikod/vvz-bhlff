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

    def __init__(self, root_dir: str = ".", output_dir: str = "code_analysis"):
        """Initialize code mapper."""
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        
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
                self.issues["files_without_docstrings"].append({
                    "file": file_path,
                    "line": 1
                })

            # Track import positions and check for imports in middle of file
            import_lines = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_lines.append(node.lineno)
            
            # Check for imports in middle of file (after line 20)
            for import_line in import_lines:
                if import_line > 20:  # Imports after line 20 are considered "in middle"
                    self.issues["imports_in_middle"].append({
                        "file": file_path,
                        "line": import_line
                    })

            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self.extract_class_info(node, file_path, import_lines)
                    file_info["classes"].append(class_info)
                    self.code_map["classes"][f"{file_path}::{node.name}"] = class_info

                elif isinstance(node, ast.FunctionDef):
                    func_info = self.extract_function_info(node, file_path, import_lines)
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

    def extract_class_info(self, node: ast.ClassDef, file_path: str, import_lines: List[int]) -> Dict[str, Any]:
        """Extract class information from AST node."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = {
                    "name": item.name,
                    "args": [arg.arg for arg in item.args.args],
                    "returns": (
                        getattr(item.returns, "id", None) if item.returns else None
                    ),
                    "is_async": isinstance(item, ast.AsyncFunctionDef),
                }
                methods.append(method_info)
                
                # Check for issues in methods
                self._check_method_issues(item, file_path, node.name)

        # Check class docstring
        class_docstring = ast.get_docstring(node)
        if not class_docstring:
            self.issues["classes_without_docstrings"].append({
                "file": file_path,
                "class": node.name,
                "line": node.lineno
            })

        return {
            "name": node.name,
            "bases": [
                base.id if isinstance(base, ast.Name) else str(base)
                for base in node.bases
            ],
            "methods": methods,
            "docstring": class_docstring,
            "decorators": [
                d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list
            ],
        }

    def extract_function_info(self, node: ast.FunctionDef, file_path: str, import_lines: List[int]) -> Dict[str, Any]:
        """Extract function information from AST node."""
        # Check for issues in functions
        self._check_method_issues(node, file_path, None)
        
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

    def _check_method_issues(self, node: ast.FunctionDef, file_path: str, class_name: str = None) -> None:
        """Check for common issues in methods and functions."""
        # Check for pass statements
        if self._has_only_pass(node):
            self.issues["methods_with_pass"].append({
                "file": file_path,
                "class": class_name,
                "method": node.name,
                "line": node.lineno
            })
        
        # Check for NotImplemented in non-abstract methods
        if self._has_not_implemented(node) and not self._is_abstract_method(node):
            self.issues["not_implemented_in_non_abstract"].append({
                "file": file_path,
                "class": class_name,
                "method": node.name,
                "line": node.lineno
            })
        
        # Check for missing docstrings
        if not ast.get_docstring(node):
            self.issues["methods_without_docstrings"].append({
                "file": file_path,
                "class": class_name,
                "method": node.name,
                "line": node.lineno
            })

    def _has_only_pass(self, node: ast.FunctionDef) -> bool:
        """Check if function has only pass statement."""
        if len(node.body) == 1:
            if isinstance(node.body[0], ast.Pass):
                return True
        return False

    def _has_not_implemented(self, node: ast.FunctionDef) -> bool:
        """Check if function contains NotImplemented."""
        for stmt in node.body:
            if isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Name) and stmt.exc.id == "NotImplementedError":
                    return True
            elif isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Name) and stmt.value.id == "NotImplemented":
                    return True
        return False

    def _is_abstract_method(self, node: ast.FunctionDef) -> bool:
        """Check if method is abstract."""
        return "abstractmethod" in [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]

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

    def generate_issues_report(self) -> str:
        """Generate issues report."""
        report = []
        report.append("# Code Issues Report")
        report.append(f"Generated for: {self.root_dir}")
        report.append("")

        # Methods with pass
        if self.issues["methods_with_pass"]:
            report.append("## 1. Методы с pass вместо тела")
            report.append(f"Найдено: {len(self.issues['methods_with_pass'])} случаев")
            report.append("")
            for issue in self.issues["methods_with_pass"]:
                if issue["class"]:
                    report.append(f"- {issue['file']}:{issue['line']} - {issue['class']}.{issue['method']}")
                else:
                    report.append(f"- {issue['file']}:{issue['line']} - {issue['method']}")
            report.append("")

        # NotImplemented in non-abstract methods
        if self.issues["not_implemented_in_non_abstract"]:
            report.append("## 2. NotImplemented в неабстрактных методах")
            report.append(f"Найдено: {len(self.issues['not_implemented_in_non_abstract'])} случаев")
            report.append("")
            for issue in self.issues["not_implemented_in_non_abstract"]:
                if issue["class"]:
                    report.append(f"- {issue['file']}:{issue['line']} - {issue['class']}.{issue['method']}")
                else:
                    report.append(f"- {issue['file']}:{issue['line']} - {issue['method']}")
            report.append("")

        # Methods without docstrings
        if self.issues["methods_without_docstrings"]:
            report.append("## 3. Методы без докстрингов")
            report.append(f"Найдено: {len(self.issues['methods_without_docstrings'])} случаев")
            report.append("")
            for issue in self.issues["methods_without_docstrings"]:
                if issue["class"]:
                    report.append(f"- {issue['file']}:{issue['line']} - {issue['class']}.{issue['method']}")
                else:
                    report.append(f"- {issue['file']}:{issue['line']} - {issue['method']}")
            report.append("")

        # Files without docstrings
        if self.issues["files_without_docstrings"]:
            report.append("## 4. Файлы без докстрингов")
            report.append(f"Найдено: {len(self.issues['files_without_docstrings'])} случаев")
            report.append("")
            for issue in self.issues["files_without_docstrings"]:
                report.append(f"- {issue['file']}:{issue['line']}")
            report.append("")

        # Classes without docstrings
        if self.issues["classes_without_docstrings"]:
            report.append("## 5. Классы без докстрингов")
            report.append(f"Найдено: {len(self.issues['classes_without_docstrings'])} случаев")
            report.append("")
            for issue in self.issues["classes_without_docstrings"]:
                report.append(f"- {issue['file']}:{issue['line']} - {issue['class']}")
            report.append("")

        # Imports in middle of file
        if self.issues["imports_in_middle"]:
            report.append("## 6. Импорты в середине файла")
            report.append(f"Найдено: {len(self.issues['imports_in_middle'])} случаев")
            report.append("")
            for issue in self.issues["imports_in_middle"]:
                report.append(f"- {issue['file']}:{issue['line']}")
            report.append("")

        # Summary
        total_issues = sum(len(issues) for issues in self.issues.values())
        report.append("## Сводка")
        report.append(f"Всего проблем: {total_issues}")
        report.append(f"- Методы с pass: {len(self.issues['methods_with_pass'])}")
        report.append(f"- NotImplemented в неабстрактных: {len(self.issues['not_implemented_in_non_abstract'])}")
        report.append(f"- Методы без докстрингов: {len(self.issues['methods_without_docstrings'])}")
        report.append(f"- Файлы без докстрингов: {len(self.issues['files_without_docstrings'])}")
        report.append(f"- Классы без докстрингов: {len(self.issues['classes_without_docstrings'])}")
        report.append(f"- Импорты в середине файла: {len(self.issues['imports_in_middle'])}")

        return "\n".join(report)

    def save_report(self) -> None:
        """Save code map report to file."""
        report = self.generate_report()
        output_file = self.output_dir / "code_map.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Отчет сохранен в файл: {output_file}")

    def generate_yaml_code_map(self) -> str:
        """Generate YAML code map."""
        import yaml
        
        yaml_data = {
            "summary": {
                "total_files": len(self.code_map["files"]),
                "total_classes": len(self.code_map["classes"]),
                "total_functions": len(self.code_map["functions"]),
                "total_imports": len(self.code_map["imports"])
            },
            "files": {},
            "classes": {},
            "functions": {},
            "imports": {}
        }
        
        # Files
        for file_key, file_info in self.code_map["files"].items():
            yaml_data["files"][file_key] = {
                "path": file_info.get("path", ""),
                "size": file_info.get("size", 0),
                "lines": file_info.get("lines", 0),
                "classes_count": file_info.get("classes_count", 0),
                "functions_count": file_info.get("functions_count", 0),
                "imports_count": file_info.get("imports_count", 0),
                "docstring": file_info.get("docstring", "")
            }
        
        # Classes
        for class_key, class_info in self.code_map["classes"].items():
            yaml_data["classes"][class_key] = {
                "name": class_info.get("name", ""),
                "file": class_info.get("file", ""),
                "line": class_info.get("line", 0),
                "methods": [method.get("name", "") for method in class_info.get("methods", [])],
                "docstring": class_info.get("docstring", ""),
                "bases": class_info.get("bases", [])
            }
        
        # Functions
        for func_key, func_info in self.code_map["functions"].items():
            yaml_data["functions"][func_key] = {
                "name": func_info.get("name", ""),
                "file": func_info.get("file", ""),
                "line": func_info.get("line", 0),
                "args": func_info.get("args", []),
                "returns": func_info.get("returns", ""),
                "docstring": func_info.get("docstring", ""),
                "decorators": func_info.get("decorators", []),
                "is_async": func_info.get("is_async", False)
            }
        
        # Imports
        for import_key, import_info in self.code_map["imports"].items():
            yaml_data["imports"][import_key] = {
                "module": import_info.get("module", ""),
                "file": import_info.get("file", ""),
                "line": import_info.get("line", 0),
                "type": import_info.get("type", "")
            }
        
        return yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True, sort_keys=True)

    def save_yaml_code_map(self) -> None:
        """Save YAML code map to file."""
        yaml_content = self.generate_yaml_code_map()
        output_file = self.output_dir / "code_map.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        logger.info(f"YAML карта кода сохранена в файл: {output_file}")

    def save_issues_report(self) -> None:
        """Save issues report to file."""
        report = self.generate_issues_report()
        output_file = self.output_dir / "code_issues.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Отчет о проблемах сохранен в файл: {output_file}")

    def generate_yaml_issues_report(self) -> str:
        """Generate YAML issues report."""
        import yaml
        
        yaml_data = {
            "summary": {
                "total_issues": sum(len(issues) for issues in self.issues.values()),
                "methods_with_pass": len(self.issues["methods_with_pass"]),
                "not_implemented_in_non_abstract": len(self.issues["not_implemented_in_non_abstract"]),
                "methods_without_docstrings": len(self.issues["methods_without_docstrings"]),
                "files_without_docstrings": len(self.issues["files_without_docstrings"]),
                "classes_without_docstrings": len(self.issues["classes_without_docstrings"]),
                "imports_in_middle": len(self.issues["imports_in_middle"])
            },
            "issues": {
                "methods_with_pass": self.issues["methods_with_pass"],
                "not_implemented_in_non_abstract": self.issues["not_implemented_in_non_abstract"],
                "methods_without_docstrings": self.issues["methods_without_docstrings"],
                "files_without_docstrings": self.issues["files_without_docstrings"],
                "classes_without_docstrings": self.issues["classes_without_docstrings"],
                "imports_in_middle": self.issues["imports_in_middle"]
            }
        }
        
        return yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True, sort_keys=True)

    def save_yaml_issues_report(self) -> None:
        """Save YAML issues report to file."""
        yaml_content = self.generate_yaml_issues_report()
        output_file = self.output_dir / "code_issues.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        logger.info(f"YAML отчет о проблемах сохранен в файл: {output_file}")

    def generate_method_index(self) -> str:
        """Generate method index report."""
        report = []
        report.append("# Method Index Report")
        report.append(f"Generated for: {self.root_dir}")
        report.append("")

        # Group methods by file
        methods_by_file = {}
        for method_key, method_info in self.code_map["functions"].items():
            file_path = method_key.split("::")[0]
            if file_path not in methods_by_file:
                methods_by_file[file_path] = []
            methods_by_file[file_path].append(method_info)

        # Generate index
        for file_path, methods in sorted(methods_by_file.items()):
            report.append(f"## {file_path}")
            report.append(f"Methods: {len(methods)}")
            report.append("")
            
            for method in methods:
                # Method signature
                args_str = ", ".join(method["args"]) if method["args"] else ""
                signature = f"{method['name']}({args_str})"
                if method["returns"]:
                    signature += f" -> {method['returns']}"
                
                report.append(f"### {signature}")
                
                # Method info
                if method["docstring"]:
                    # Truncate long docstrings
                    docstring = method["docstring"]
                    if len(docstring) > 200:
                        docstring = docstring[:200] + "..."
                    report.append(f"**Описание:** {docstring}")
                else:
                    report.append("**Описание:** Нет докстринга")
                
                if method["decorators"]:
                    report.append(f"**Декораторы:** {', '.join(method['decorators'])}")
                
                if method["is_async"]:
                    report.append("**Тип:** async")
                
                report.append("")

        # Summary
        total_methods = len(self.code_map["functions"])
        report.append("## Сводка")
        report.append(f"Всего методов: {total_methods}")
        report.append(f"Файлов с методами: {len(methods_by_file)}")
        
        # Top files by method count
        file_counts = [(file_path, len(methods)) for file_path, methods in methods_by_file.items()]
        file_counts.sort(key=lambda x: x[1], reverse=True)
        
        report.append("")
        report.append("### Топ файлов по количеству методов:")
        for file_path, count in file_counts[:10]:
            report.append(f"- {file_path}: {count} методов")

        return "\n".join(report)

    def save_method_index(self) -> None:
        """Save method index report to file."""
        report = self.generate_method_index()
        output_file = self.output_dir / "method_index.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Индекс методов сохранен в файл: {output_file}")

    def generate_yaml_method_index(self) -> str:
        """Generate YAML method index."""
        import yaml
        
        yaml_data = {}
        
        # Group methods by file and class
        for method_key, method_info in self.code_map["functions"].items():
            file_path, method_name = method_key.split("::")
            
            # Find class for this method
            class_name = None
            for class_key, class_info in self.code_map["classes"].items():
                if class_key.startswith(file_path + "::"):
                    if method_name in [m["name"] for m in class_info["methods"]]:
                        class_name = class_info["name"]
                        break
            
            # Create file entry if not exists
            if file_path not in yaml_data:
                yaml_data[file_path] = {}
            
            # Create class entry if not exists
            if class_name and class_name not in yaml_data[file_path]:
                yaml_data[file_path][class_name] = {}
            
            # Method entry
            method_entry = {
                "description": method_info["docstring"] or "Нет описания",
                "returns": method_info["returns"] or "None",
                "parameters": {}
            }
            
            # Add parameters
            for i, arg in enumerate(method_info["args"]):
                # Try to get type from method signature if available
                param_type = "Any"  # Default type
                param_desc = f"Параметр {i+1}"
                
                method_entry["parameters"][arg] = {
                    "type": param_type,
                    "description": param_desc
                }
            
            # Add decorators if any
            if method_info["decorators"]:
                method_entry["decorators"] = method_info["decorators"]
            
            # Add async flag
            if method_info["is_async"]:
                method_entry["async"] = True
            
            # Store method
            if class_name:
                yaml_data[file_path][class_name][method_name] = method_entry
            else:
                # Standalone function
                if "functions" not in yaml_data[file_path]:
                    yaml_data[file_path]["functions"] = {}
                yaml_data[file_path]["functions"][method_name] = method_entry
        
        return yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True, sort_keys=True)

    def save_yaml_method_index(self) -> None:
        """Save YAML method index to file."""
        yaml_content = self.generate_yaml_method_index()
        output_file = self.output_dir / "method_index.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        logger.info(f"YAML индекс методов сохранен в файл: {output_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code mapper for BHLFF project")
    parser.add_argument("--root-dir", default=".", help="Root directory to scan")
    parser.add_argument("--output-dir", default="code_analysis", help="Output directory for reports")
    
    args = parser.parse_args()
    
    mapper = CodeMapper(root_dir=args.root_dir, output_dir=args.output_dir)
    mapper.scan_directory(args.root_dir)
    
    # Save all reports in YAML format
    mapper.save_yaml_code_map()
    mapper.save_yaml_issues_report()
    mapper.save_yaml_method_index()
    
    print("Анализ завершен!")
    print(f"Созданы YAML отчеты в каталоге: {mapper.output_dir}")
    print("- code_map.yaml - карта кода")
    print("- code_issues.yaml - проблемы в коде")
    print("- method_index.yaml - индекс методов")


if __name__ == "__main__":
    main()

