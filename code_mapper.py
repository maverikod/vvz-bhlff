#!/usr/bin/env python3
"""
Code Mapper - скрипт для анализа кода и формирования карты по файлам.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Этот скрипт проходит по всем файлам кода и формирует карту:
- Файл - описание
- Докстринг описания  
- ИмяКласса - описание
- Сигнатура метода - описание

Usage:
    python code_mapper.py [--output OUTPUT_FILE] [--include-pattern PATTERN] [--exclude-pattern PATTERN]
"""

import os
import sys
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


class CodeMapper:
    """
    Анализатор кода для создания карты файлов.
    
    Physical Meaning:
        Анализирует структуру Python кода, извлекая информацию о классах,
        методах, функциях и их документации для создания карты проекта.
    """
    
    def __init__(self, root_dir: str, output_file: str = "code_map.md", output_dir: str = None):
        """
        Инициализация анализатора кода.
        
        Args:
            root_dir (str): Корневая директория для анализа.
            output_file (str): Файл для записи результатов.
            output_dir (str): Директория для записи выходных файлов.
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.output_file = output_file
        self.code_map = {}
        
    def _ensure_output_dir(self) -> None:
        """Создание выходной директории если она не существует."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_output_path(self, filename: str) -> Path:
        """Получение полного пути к выходному файлу."""
        if self.output_dir:
            return self.output_dir / filename
        else:
            return Path(filename)
        
    def analyze_file(self, file_path: Path) -> Dict:
        """
        Анализ одного Python файла.
        
        Args:
            file_path (Path): Путь к файлу для анализа.
            
        Returns:
            Dict: Словарь с информацией о файле.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Парсинг AST
            tree = ast.parse(content, filename=str(file_path))
            
            file_info = {
                'path': str(file_path.relative_to(self.root_dir)),
                'module_docstring': self._extract_module_docstring(tree),
                'classes': [],
                'functions': [],
                'imports': []
            }
            
            # Анализ узлов AST
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    file_info['classes'].append(class_info)
                elif isinstance(node, ast.FunctionDef):
                    if not self._is_method(node, tree):
                        func_info = self._analyze_function(node)
                        file_info['functions'].append(func_info)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_import(node)
                    file_info['imports'].append(import_info)
            
            return file_info
            
        except Exception as e:
            return {
                'path': str(file_path.relative_to(self.root_dir)),
                'error': f"Ошибка анализа: {e}",
                'classes': [],
                'functions': [],
                'imports': []
            }
    
    def _extract_module_docstring(self, tree: ast.AST) -> str:
        """Извлечение докстринга модуля."""
        if (tree.body and isinstance(tree.body[0], ast.Expr) 
            and isinstance(tree.body[0].value, ast.Constant) 
            and isinstance(tree.body[0].value.value, str)):
            return tree.body[0].value.value.strip()
        return ""
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict:
        """Анализ класса."""
        class_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node) or "",
            'methods': [],
            'attributes': [],
            'inheritance': [base.id for base in node.bases if isinstance(base, ast.Name)]
        }
        
        # Анализ методов класса
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_method(item)
                class_info['methods'].append(method_info)
            elif isinstance(item, ast.Assign):
                # Простые атрибуты класса
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info['attributes'].append(target.id)
        
        return class_info
    
    def _analyze_method(self, node: ast.FunctionDef) -> Dict:
        """Анализ метода класса."""
        return {
            'name': node.name,
            'signature': self._get_method_signature(node),
            'docstring': ast.get_docstring(node) or "",
            'is_private': node.name.startswith('_'),
            'is_abstract': any(isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod' 
                             for decorator in node.decorator_list)
        }
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict:
        """Анализ функции."""
        return {
            'name': node.name,
            'signature': self._get_function_signature(node),
            'docstring': ast.get_docstring(node) or "",
            'is_private': node.name.startswith('_')
        }
    
    def _analyze_import(self, node) -> Dict:
        """Анализ импорта."""
        if isinstance(node, ast.Import):
            return {
                'type': 'import',
                'names': [alias.name for alias in node.names]
            }
        elif isinstance(node, ast.ImportFrom):
            return {
                'type': 'from_import',
                'module': node.module or "",
                'names': [alias.name for alias in node.names]
            }
        return {}
    
    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Проверка, является ли функция методом класса."""
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                for item in class_node.body:
                    if item == node:
                        return True
        return False
    
    def _get_method_signature(self, node: ast.FunctionDef) -> str:
        """Получение сигнатуры метода."""
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            if arg_name == 'self':
                continue
            args.append(arg_name)
        
        signature = f"{node.name}({', '.join(args)})"
        return signature
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Получение сигнатуры функции."""
        args = [arg.arg for arg in node.args.args]
        signature = f"{node.name}({', '.join(args)})"
        return signature
    
    def scan_directory(self, include_pattern: str = "*.py", exclude_pattern: str = None) -> None:
        """
        Сканирование директории для поиска Python файлов.
        
        Args:
            include_pattern (str): Паттерн для включения файлов.
            exclude_pattern (str): Паттерн для исключения файлов.
        """
        print(f"Сканирование директории: {self.root_dir}")
        
        for file_path in self.root_dir.rglob(include_pattern):
            # Исключение файлов по паттерну
            if exclude_pattern and re.search(exclude_pattern, str(file_path)):
                continue
            
            # Исключение служебных директорий и файлов
            excluded_dirs = ['__pycache__', '.git', '.venv', 'venv', 'env', 'node_modules', 
                           '.pytest_cache', '.coverage', 'htmlcov', 'dist', 'build', 
                           '.tox', '.mypy_cache', '.ruff_cache']
            
            if any(excluded_dir in str(file_path) for excluded_dir in excluded_dirs):
                continue
            
            print(f"Анализ файла: {file_path.relative_to(self.root_dir)}")
            file_info = self.analyze_file(file_path)
            self.code_map[str(file_path.relative_to(self.root_dir))] = file_info
    
    def generate_report(self) -> str:
        """
        Генерация отчета в формате Markdown.
        
        Returns:
            str: Сгенерированный отчет.
        """
        report = []
        report.append("# Карта кода проекта BHLFF")
        report.append("")
        report.append(f"**Автор:** Vasiliy Zdanovskiy")
        report.append(f"**Email:** vasilyvz@gmail.com")
        report.append(f"**Дата:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## Обзор")
        report.append("")
        report.append(f"Проанализировано файлов: {len(self.code_map)}")
        report.append("")
        
        # Статистика
        total_classes = sum(len(file_info.get('classes', [])) for file_info in self.code_map.values())
        total_functions = sum(len(file_info.get('functions', [])) for file_info in self.code_map.values())
        total_methods = sum(
            len(class_info.get('methods', [])) 
            for file_info in self.code_map.values() 
            for class_info in file_info.get('classes', [])
        )
        
        report.append("### Статистика")
        report.append("")
        report.append(f"- **Всего файлов:** {len(self.code_map)}")
        report.append(f"- **Всего классов:** {total_classes}")
        report.append(f"- **Всего функций:** {total_functions}")
        report.append(f"- **Всего методов:** {total_methods}")
        report.append("")
        
        # Анализ по файлам
        report.append("## Анализ по файлам")
        report.append("")
        
        for file_path, file_info in sorted(self.code_map.items()):
            report.append(f"### {file_path}")
            report.append("")
            
            # Ошибка анализа
            if 'error' in file_info:
                report.append(f"❌ **Ошибка:** {file_info['error']}")
                report.append("")
                continue
            
            # Описание модуля
            if file_info.get('module_docstring'):
                report.append("**Описание модуля:**")
                report.append("")
                report.append(f"```")
                report.append(file_info['module_docstring'])
                report.append("```")
                report.append("")
            
            # Классы
            if file_info.get('classes'):
                report.append("**Классы:**")
                report.append("")
                for class_info in file_info['classes']:
                    report.append(f"- **{class_info['name']}**")
                    if class_info.get('inheritance'):
                        report.append(f"  - Наследование: {', '.join(class_info['inheritance'])}")
                    if class_info.get('docstring'):
                        report.append(f"  - Описание: {class_info['docstring'][:100]}...")
                    report.append("")
                    
                    # Методы класса
                    if class_info.get('methods'):
                        report.append("  **Методы:**")
                        for method in class_info['methods']:
                            signature = method['signature']
                            docstring = method.get('docstring', '')
                            private_mark = "🔒 " if method.get('is_private') else ""
                            abstract_mark = "🔸 " if method.get('is_abstract') else ""
                            report.append(f"  - {private_mark}{abstract_mark}`{signature}`")
                            if docstring:
                                report.append(f"    - {docstring[:80]}...")
                        report.append("")
            
            # Функции
            if file_info.get('functions'):
                report.append("**Функции:**")
                report.append("")
                for func_info in file_info['functions']:
                    signature = func_info['signature']
                    docstring = func_info.get('docstring', '')
                    private_mark = "🔒 " if func_info.get('is_private') else ""
                    report.append(f"- {private_mark}`{signature}`")
                    if docstring:
                        report.append(f"  - {docstring[:80]}...")
                report.append("")
            
            # Импорты (только основные)
            if file_info.get('imports'):
                main_imports = []
                for imp in file_info['imports'][:5]:  # Показываем только первые 5
                    if imp['type'] == 'import':
                        main_imports.extend(imp['names'])
                    elif imp['type'] == 'from_import':
                        main_imports.extend([f"{imp['module']}.{name}" for name in imp['names']])
                
                if main_imports:
                    report.append("**Основные импорты:**")
                    report.append("")
                    for imp in main_imports[:10]:  # Показываем только первые 10
                        report.append(f"- `{imp}`")
                    report.append("")
            
            report.append("---")
            report.append("")
        
        return "\n".join(report)
    
    def generate_method_index(self) -> str:
        """
        Генерация индекса методов с ссылками на файлы и строки.
        
        Returns:
            str: Сгенерированный индекс методов.
        """
        index = []
        index.append("# Индекс методов проекта BHLFF")
        index.append("")
        index.append(f"**Автор:** Vasiliy Zdanovskiy")
        index.append(f"**Email:** vasilyvz@gmail.com")
        index.append(f"**Дата:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        index.append("")
        index.append("## Описание")
        index.append("")
        index.append("Этот индекс содержит все методы проекта с указанием файлов и строк, где они определены.")
        index.append("Формат: ИмяКласса -> ИмяМетода -> Файл, строка")
        index.append("")
        
        # Собираем все методы по классам
        methods_by_class = {}
        
        for file_path, file_info in self.code_map.items():
            if 'error' in file_info:
                continue
                
            # Читаем файл для получения номеров строк
            try:
                with open(self.root_dir / file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except:
                continue
            
            for class_info in file_info.get('classes', []):
                class_name = class_info['name']
                
                if class_name not in methods_by_class:
                    methods_by_class[class_name] = {}
                
                for method in class_info.get('methods', []):
                    method_name = method['name']
                    
                    # Находим номер строки метода в файле
                    line_number = self._find_method_line_number(lines, method_name)
                    
                    if method_name not in methods_by_class[class_name]:
                        methods_by_class[class_name][method_name] = []
                    
                    methods_by_class[class_name][method_name].append({
                        'file': file_path,
                        'line': line_number,
                        'signature': method['signature'],
                        'docstring': method.get('docstring', ''),
                        'is_private': method.get('is_private', False),
                        'is_abstract': method.get('is_abstract', False)
                    })
        
        # Генерируем индекс
        for class_name in sorted(methods_by_class.keys()):
            index.append(f"## {class_name}")
            index.append("")
            
            for method_name in sorted(methods_by_class[class_name].keys()):
                method_info = methods_by_class[class_name][method_name]
                
                # Заголовок метода
                private_mark = "🔒 " if any(m.get('is_private') for m in method_info) else ""
                abstract_mark = "🔸 " if any(m.get('is_abstract') for m in method_info) else ""
                index.append(f"### {private_mark}{abstract_mark}{method_name}")
                index.append("")
                
                # Описание метода (из первого вхождения)
                if method_info[0].get('docstring'):
                    docstring = method_info[0]['docstring'][:200]
                    index.append(f"**Описание:** {docstring}...")
                    index.append("")
                
                # Ссылки на файлы
                index.append("**Определения:**")
                index.append("")
                
                for method in method_info:
                    file_path = method['file']
                    line_number = method['line']
                    signature = method['signature']
                    
                    # Создаем ссылку в формате, удобном для IDE
                    link_text = f"`{signature}`"
                    if line_number > 0:
                        link_text += f" - {file_path}:{line_number}"
                    else:
                        link_text += f" - {file_path}"
                    
                    index.append(f"- {link_text}")
                
                index.append("")
        
        return "\n".join(index)
    
    def _find_method_line_number(self, lines: List[str], method_name: str) -> int:
        """
        Поиск номера строки метода в файле.
        
        Args:
            lines (List[str]): Строки файла.
            method_name (str): Имя метода.
            
        Returns:
            int: Номер строки метода (1-based) или 0 если не найден.
        """
        for i, line in enumerate(lines, 1):
            # Ищем определение метода
            if f"def {method_name}(" in line:
                return i
        return 0
    
    def save_report(self) -> None:
        """Сохранение отчета в файл."""
        self._ensure_output_dir()
        report = self.generate_report()
        
        output_path = self._get_output_path(self.output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Отчет сохранен в файл: {output_path}")
    
    def save_method_index(self, index_file: str = "method_index.md") -> None:
        """Сохранение индекса методов в файл."""
        self._ensure_output_dir()
        index = self.generate_method_index()
        
        output_path = self._get_output_path(index_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(index)
        
        print(f"Индекс методов сохранен в файл: {output_path}")


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(description='Анализатор кода для создания карты проекта')
    parser.add_argument('--root-dir', default='.', help='Корневая директория для анализа (по умолчанию: текущая)')
    parser.add_argument('--output', default='code_map.md', help='Файл для записи результатов (по умолчанию: code_map.md)')
    parser.add_argument('--index-output', default='method_index.md', help='Файл для записи индекса методов (по умолчанию: method_index.md)')
    parser.add_argument('--output-dir', help='Директория для записи выходных файлов (по умолчанию: текущая директория)')
    parser.add_argument('--include-pattern', default='*.py', help='Паттерн для включения файлов (по умолчанию: *.py)')
    parser.add_argument('--exclude-pattern', help='Паттерн для исключения файлов')
    parser.add_argument('--generate-index', action='store_true', help='Создать индекс методов')
    parser.add_argument('--index-only', action='store_true', help='Создать только индекс методов (без карты кода)')
    
    args = parser.parse_args()
    
    # Создание анализатора
    mapper = CodeMapper(args.root_dir, args.output, args.output_dir)
    
    # Сканирование директории
    mapper.scan_directory(args.include_pattern, args.exclude_pattern)
    
    # Генерация и сохранение отчета
    if not args.index_only:
        mapper.save_report()
    
    # Генерация и сохранение индекса методов
    if args.generate_index or args.index_only:
        mapper.save_method_index(args.index_output)
    
    print("Анализ завершен!")


if __name__ == "__main__":
    main()
