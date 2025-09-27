# Step 01: Максимально детализированная базовая структура проекта

## Цель
Создать фундаментальную архитектуру проекта для реализации 7-мерной теории материи в фазовом пространстве-времени с учетом всех теоретических принципов и требований.

## Теоретические основы

### 1. 7-мерное фазовое пространство-время
- **3 пространственные координаты** (x, y, z) - привычная геометрия
- **3 фазовых параметра** (φ₁, φ₂, φ₃) - внутренние состояния поля
- **1 временная координата** (t) - динамика эволюции

### 2. Ключевые принципы теории
- **Материя как фазовый рисунок**: элементарные частицы = устойчивые фазовые конфигурации
- **Трехуровневая структура частиц**: ядро + переходная зона + хвост
- **Баланс сжатия-разрежения**: локальные изменения компенсируются глобально
- **Скорость передачи фазы**: c_φ ≫ c (на десятки порядков выше скорости света)
- **Топологическая стабилизация**: через обмотки и дефекты

### 3. Математическая основа
- **Фазовое поле**: θ(x,φ,t) на многообразии M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
- **Энергетический функционал**: E[θ] = ∫(f_φ²|∇θ|² + β₄(Δθ)² + γ₆|∇θ|⁶ + ...)dV
- **Оператор Рисса**: L_β a = μ(-Δ)^β a + λa
- **Граничные условия**: непрерывность, однородность на бесконечности

## Детальная структура проекта

### 1. Корневая структура
```
bhlff/
├── README.md                    # Основная документация проекта
├── LICENSE                      # Лицензия проекта
├── pyproject.toml              # Современная конфигурация Python
├── setup.py                    # Установка пакета
├── requirements.txt            # Зависимости с точными версиями
├── .gitignore                  # Игнорируемые файлы
├── .env.example                # Пример переменных окружения
├── .venv/                      # Виртуальное окружение
├── docs/                       # Документация
├── src/                        # Исходный код
├── tests/                      # Тесты
├── configs/                    # Конфигурации
├── data/                       # Данные
├── output/                     # Результаты расчетов
├── scripts/                    # Скрипты
└── tools/                      # Инструменты разработки
```

### 2. Структура исходного кода (src/)
```
src/bhlff/
├── __init__.py                 # Основной пакет
├── core/                       # Ядро системы
│   ├── __init__.py
│   ├── base/                   # Базовые классы и интерфейсы
│   │   ├── __init__.py
│   │   ├── abstract_solver.py  # Абстрактный базовый класс решателя
│   │   ├── field.py            # Базовый класс фазового поля
│   │   ├── domain.py           # Класс области расчета
│   │   └── parameters.py       # Управление параметрами
│   ├── fft/                    # FFT решатели
│   │   ├── __init__.py
│   │   ├── fft_solver.py       # Основной FFT решатель
│   │   ├── frac_laplacian.py   # Фракционный оператор Рисса
│   │   └── spectral_ops.py     # Спектральные операции
│   ├── time/                   # Временные интеграторы
│   │   ├── __init__.py
│   │   ├── integrators.py      # Временные интеграторы
│   │   ├── schemes.py          # Схемы интегрирования
│   │   └── stability.py        # Анализ устойчивости
│   ├── phase/                  # Фазовое поле
│   │   ├── __init__.py
│   │   ├── phase_field.py      # Фазовое поле и его свойства
│   │   ├── topology.py         # Топологические дефекты
│   │   ├── winding.py          # Вычисление обмоток
│   │   └── defects.py          # Дефекты и солитоны
│   └── physics/                # Физические модели
│       ├── __init__.py
│       ├── operators.py        # Физические операторы
│       ├── boundary.py         # Граничные условия
│       └── energy.py           # Энергетические функционалы
├── models/                     # Модели различных уровней
│   ├── __init__.py
│   ├── level_a/                # Уровень A: базовые решатели
│   │   ├── __init__.py
│   │   ├── validation.py       # Валидация решателей
│   │   ├── scaling.py          # Масштабирование
│   │   └── benchmarks.py       # Бенчмарки
│   ├── level_b/                # Уровень B: фундаментальные свойства
│   │   ├── __init__.py
│   │   ├── power_law.py        # Степенные хвосты
│   │   ├── nodes.py            # Анализ узлов
│   │   ├── charge.py           # Топологический заряд
│   │   └── zones.py            # Разделение зон
│   ├── level_c/                # Уровень C: границы и ячейки
│   │   ├── __init__.py
│   │   ├── boundaries.py       # Граничные эффекты
│   │   ├── resonators.py       # Резонаторы
│   │   ├── memory.py           # Квенч-память
│   │   └── beating.py          # Биения мод
│   ├── level_d/                # Уровень D: многомодовые модели
│   │   ├── __init__.py
│   │   ├── superposition.py    # Наложение мод
│   │   ├── projections.py      # Проекции полей
│   │   └── streamlines.py      # Линии потока
│   ├── level_e/                # Уровень E: солитоны и дефекты
│   │   ├── __init__.py
│   │   ├── solitons.py         # Солитоны
│   │   ├── dynamics.py         # Динамика дефектов
│   │   ├── interactions.py     # Взаимодействия
│   │   └── formation.py        # Образование дефектов
│   ├── level_f/                # Уровень F: коллективные эффекты
│   │   ├── __init__.py
│   │   ├── multi_particle.py   # Многочастичные системы
│   │   ├── collective.py       # Коллективные моды
│   │   ├── transitions.py      # Фазовые переходы
│   │   └── nonlinear.py        # Нелинейные эффекты
│   └── level_g/                # Уровень G: космологические модели
│       ├── __init__.py
│       ├── cosmology.py        # Космологическая эволюция
│       ├── structure.py        # Крупномасштабная структура
│       ├── astrophysics.py     # Астрофизические объекты
│       └── gravity.py          # Гравитационные эффекты
├── utils/                      # Утилиты
│   ├── __init__.py
│   ├── config/                 # Управление конфигурацией
│   │   ├── __init__.py
│   │   ├── loader.py           # Загрузка конфигураций
│   │   ├── validator.py        # Валидация параметров
│   │   └── defaults.py         # Значения по умолчанию
│   ├── io/                     # Ввод-вывод
│   │   ├── __init__.py
│   │   ├── hdf5.py             # HDF5 файлы
│   │   ├── numpy.py            # NumPy файлы
│   │   └── json.py             # JSON файлы
│   ├── math/                   # Математические утилиты
│   │   ├── __init__.py
│   │   ├── interpolation.py    # Интерполяция
│   │   ├── integration.py      # Интегрирование
│   │   └── statistics.py       # Статистика
│   ├── visualization/          # Визуализация
│   │   ├── __init__.py
│   │   ├── plots.py            # Графики
│   │   ├── animations.py       # Анимации
│   │   ├── 3d.py               # 3D визуализация
│   │   └── export.py           # Экспорт графиков
│   ├── analysis/               # Анализ данных
│   │   ├── __init__.py
│   │   ├── statistics.py       # Статистический анализ
│   │   ├── comparison.py       # Сравнение с теорией
│   │   ├── quality.py          # Метрики качества
│   │   └── trends.py           # Трендовый анализ
│   └── reporting/              # Отчетность
│       ├── __init__.py
│       ├── generator.py        # Генерация отчетов
│       ├── templates.py        # Шаблоны отчетов
│       └── export.py           # Экспорт отчетов
└── cli/                        # Командная строка
    ├── __init__.py
    ├── main.py                 # Основная команда
    ├── run.py                  # Запуск экспериментов
    ├── analyze.py              # Анализ результатов
    └── report.py               # Генерация отчетов
```

### 3. Структура тестов (tests/)
```
tests/
├── __init__.py
├── conftest.py                 # Конфигурация pytest
├── fixtures/                   # Фикстуры для тестов
│   ├── __init__.py
│   ├── domains.py              # Тестовые области
│   ├── fields.py               # Тестовые поля
│   └── parameters.py           # Тестовые параметры
├── unit/                       # Юнит-тесты
│   ├── __init__.py
│   ├── test_core/              # Тесты ядра
│   │   ├── __init__.py
│   │   ├── test_fft_solver.py
│   │   ├── test_frac_laplacian.py
│   │   ├── test_integrators.py
│   │   └── test_phase_field.py
│   ├── test_models/            # Тесты моделей
│   │   ├── __init__.py
│   │   ├── test_level_a.py
│   │   ├── test_level_b.py
│   │   ├── test_level_c.py
│   │   ├── test_level_d.py
│   │   ├── test_level_e.py
│   │   ├── test_level_f.py
│   │   └── test_level_g.py
│   └── test_utils/             # Тесты утилит
│       ├── __init__.py
│       ├── test_config.py
│       ├── test_io.py
│       ├── test_visualization.py
│       └── test_analysis.py
├── integration/                # Интеграционные тесты
│   ├── __init__.py
│   ├── test_full_pipeline.py   # Полный пайплайн
│   ├── test_level_integration.py # Интеграция уровней
│   └── test_performance.py     # Тесты производительности
└── benchmarks/                 # Бенчмарки
    ├── __init__.py
    ├── benchmark_solvers.py    # Бенчмарки решателей
    ├── benchmark_memory.py     # Бенчмарки памяти
    └── benchmark_accuracy.py   # Бенчмарки точности
```

### 4. Структура конфигураций (configs/)
```
configs/
├── default.json                # Конфигурация по умолчанию
├── level_a/                    # Конфигурации уровня A
│   ├── validation.json
│   ├── scaling.json
│   └── benchmarks.json
├── level_b/                    # Конфигурации уровня B
│   ├── power_law.json
│   ├── nodes.json
│   ├── charge.json
│   └── zones.json
├── level_c/                    # Конфигурации уровня C
│   ├── boundaries.json
│   ├── resonators.json
│   ├── memory.json
│   └── beating.json
├── level_d/                    # Конфигурации уровня D
│   ├── superposition.json
│   ├── projections.json
│   └── streamlines.json
├── level_e/                    # Конфигурации уровня E
│   ├── solitons.json
│   ├── dynamics.json
│   ├── interactions.json
│   └── formation.json
├── level_f/                    # Конфигурации уровня F
│   ├── multi_particle.json
│   ├── collective.json
│   ├── transitions.json
│   └── nonlinear.json
├── level_g/                    # Конфигурации уровня G
│   ├── cosmology.json
│   ├── structure.json
│   ├── astrophysics.json
│   └── gravity.json
└── templates/                  # Шаблоны конфигураций
    ├── base.json
    ├── physics.json
    ├── domain.json
    └── output.json
```

## Ключевые компоненты

### 1. Базовые классы и интерфейсы

#### AbstractSolver (src/bhlff/core/base/abstract_solver.py)
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Abstract base class for all solvers in the 7D phase field theory.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

class AbstractSolver(ABC):
    """
    Abstract base class for all numerical solvers.
    
    This class defines the interface that all solvers must implement,
    ensuring consistency across different levels of the theory.
    """
    
    def __init__(self, domain: 'Domain', parameters: Dict[str, Any]):
        """
        Initialize the solver with domain and parameters.
        
        Args:
            domain: Computational domain
            parameters: Solver-specific parameters
        """
        self.domain = domain
        self.parameters = parameters
        self._validate_parameters()
    
    @abstractmethod
    def solve(self, source: np.ndarray) -> np.ndarray:
        """
        Solve the equation for given source.
        
        Args:
            source: Source term
            
        Returns:
            Solution field
        """
        pass
    
    @abstractmethod
    def validate_solution(self, solution: np.ndarray, source: np.ndarray) -> Dict[str, float]:
        """
        Validate the solution quality.
        
        Args:
            solution: Computed solution
            source: Source term
            
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    def _validate_parameters(self) -> None:
        """Validate solver parameters."""
        # Implementation depends on specific solver
        pass
```

#### PhaseField (src/bhlff/core/phase/phase_field.py)
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Core phase field implementation for 7D theory.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..base.field import BaseField

class PhaseField(BaseField):
    """
    Phase field implementation for 7D phase space-time theory.
    
    Represents the phase field θ(x,φ,t) on the 7D manifold
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
    """
    
    def __init__(self, domain: 'Domain', initial_condition: Optional[np.ndarray] = None):
        """
        Initialize phase field.
        
        Args:
            domain: Computational domain
            initial_condition: Initial field values
        """
        super().__init__(domain)
        self.field = initial_condition if initial_condition is not None else np.zeros(domain.shape)
        self.phase_velocity = 1e15  # c_φ >> c (phase velocity much higher than light speed)
        
    def compute_energy(self) -> float:
        """
        Compute the energy functional E[θ].
        
        Returns:
            Total energy of the field
        """
        # Implementation of energy functional
        # E[θ] = ∫(f_φ²|∇θ|² + β₄(Δθ)² + γ₆|∇θ|⁶ + ...)dV
        pass
    
    def compute_topological_charge(self, center: Tuple[float, float, float]) -> float:
        """
        Compute topological charge around a point.
        
        Args:
            center: Center point for charge calculation
            
        Returns:
            Topological charge (integer multiple of 2π)
        """
        # Implementation of topological charge calculation
        # ∮∇φ·dl = 2πq
        pass
    
    def separate_zones(self, thresholds: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Separate field into core, transition, and tail zones.
        
        Args:
            thresholds: Threshold values for zone separation
            
        Returns:
            Dictionary with zone masks
        """
        # Implementation of zone separation
        pass
```

### 2. Система конфигурации

#### ConfigLoader (src/bhlff/utils/config/loader.py)
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Configuration loading and management system.
"""

import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path

class ConfigLoader:
    """
    Configuration loader for the 7D phase field theory project.
    
    Supports JSON and YAML formats with validation and defaults.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path("configs")
        self._defaults = self._load_defaults()
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Merge with defaults
        config = self._merge_with_defaults(config)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "domain": {
                "L": 1.0,
                "N": 256,
                "dimensions": 3
            },
            "physics": {
                "mu": 1.0,
                "beta": 1.0,
                "lambda": 0.0,
                "nu": 1.0
            },
            "solver": {
                "precision": "float64",
                "fft_plan": "MEASURE",
                "tolerance": 1e-12
            },
            "output": {
                "save_fields": True,
                "save_spectra": True,
                "save_analysis": True,
                "format": "hdf5"
            }
        }
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with defaults."""
        # Recursive merge implementation
        pass
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        # Validation implementation
        pass
```

### 3. Система логирования

#### Logger (src/bhlff/utils/logging.py)
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Structured logging system for the 7D phase field theory project.
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class StructuredLogger:
    """
    Structured logger for the 7D phase field theory project.
    
    Provides structured logging with different levels of detail
    and export to various formats.
    """
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def log_experiment_start(self, experiment_name: str, parameters: Dict[str, Any]) -> None:
        """Log experiment start."""
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.debug(f"Parameters: {json.dumps(parameters, indent=2)}")
    
    def log_experiment_end(self, experiment_name: str, results: Dict[str, Any]) -> None:
        """Log experiment end."""
        self.logger.info(f"Completed experiment: {experiment_name}")
        self.logger.debug(f"Results: {json.dumps(results, indent=2)}")
    
    def log_solver_step(self, step: int, residual: float, time: float) -> None:
        """Log solver step."""
        self.logger.debug(f"Step {step}: residual={residual:.2e}, time={time:.3f}s")
    
    def log_validation_result(self, test_name: str, passed: bool, metrics: Dict[str, float]) -> None:
        """Log validation result."""
        status = "PASS" if passed else "FAIL"
        self.logger.info(f"Validation {test_name}: {status}")
        self.logger.debug(f"Metrics: {json.dumps(metrics, indent=2)}")
```

### 4. Система управления зависимостями

#### requirements.txt
```
# Core scientific computing
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
matplotlib>=3.7.0,<4.0.0
h5py>=3.8.0,<4.0.0

# FFT and numerical methods
pyfftw>=0.13.0,<1.0.0
numba>=0.57.0,<1.0.0

# Configuration and data handling
pyyaml>=6.0,<7.0
pydantic>=2.0.0,<3.0.0
pandas>=2.0.0,<3.0.0

# Visualization
plotly>=5.15.0,<6.0.0
mayavi>=4.8.0,<5.0.0
vtk>=9.2.0,<10.0.0

# Testing and development
pytest>=7.4.0,<8.0.0
pytest-cov>=4.1.0,<5.0.0
pytest-benchmark>=4.0.0,<5.0.0
black>=23.0.0,<24.0.0
flake8>=6.0.0,<7.0.0
mypy>=1.5.0,<2.0.0

# Documentation
sphinx>=7.0.0,<8.0.0
sphinx-rtd-theme>=1.3.0,<2.0.0

# CLI
click>=8.1.0,<9.0.0
rich>=13.0.0,<14.0.0
```

#### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bhlff"
version = "0.1.0"
description = "7D Phase Field Theory Implementation"
authors = [
    {name = "Vasiliy Zdanovskiy", email = "vasilyvz@gmail.com"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Mathematics",
]

dependencies = [
    "numpy>=1.24.0,<2.0.0",
    "scipy>=1.10.0,<2.0.0",
    "matplotlib>=3.7.0,<4.0.0",
    "h5py>=3.8.0,<4.0.0",
    "pyfftw>=0.13.0,<1.0.0",
    "numba>=0.57.0,<1.0.0",
    "pyyaml>=6.0,<7.0",
    "pydantic>=2.0.0,<3.0.0",
    "pandas>=2.0.0,<3.0.0",
    "plotly>=5.15.0,<6.0.0",
    "click>=8.1.0,<9.0.0",
    "rich>=13.0.0,<14.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0,<8.0.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "pytest-benchmark>=4.0.0,<5.0.0",
    "black>=23.0.0,<24.0.0",
    "flake8>=6.0.0,<7.0.0",
    "mypy>=1.5.0,<2.0.0",
]
docs = [
    "sphinx>=7.0.0,<8.0.0",
    "sphinx-rtd-theme>=1.3.0,<2.0.0",
]
viz = [
    "mayavi>=4.8.0,<5.0.0",
    "vtk>=9.2.0,<10.0.0",
]

[project.scripts]
bhlff = "bhlff.cli.main:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src/bhlff",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "benchmark: marks tests as benchmarks",
]
```

### 5. Система тестирования

#### conftest.py (tests/conftest.py)
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Pytest configuration and fixtures for the 7D phase field theory project.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, Generator

@pytest.fixture
def test_domain() -> Dict[str, Any]:
    """Test domain configuration."""
    return {
        "L": 2.0,
        "N": 64,
        "dimensions": 3
    }

@pytest.fixture
def test_physics_params() -> Dict[str, Any]:
    """Test physics parameters."""
    return {
        "mu": 1.0,
        "beta": 1.0,
        "lambda": 0.1,
        "nu": 1.0
    }

@pytest.fixture
def test_source_field(test_domain: Dict[str, Any]) -> np.ndarray:
    """Test source field."""
    N = test_domain["N"]
    # Create a simple Gaussian source
    x = np.linspace(0, test_domain["L"], N, endpoint=False)
    y = np.linspace(0, test_domain["L"], N, endpoint=False)
    z = np.linspace(0, test_domain["L"], N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    center = test_domain["L"] / 2
    sigma = test_domain["L"] / 8
    
    source = np.exp(-((X - center)**2 + (Y - center)**2 + (Z - center)**2) / (2 * sigma**2))
    return source.astype(np.complex128)

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir
    # Cleanup is handled by tmp_path fixture

@pytest.fixture(scope="session")
def reference_solutions() -> Dict[str, np.ndarray]:
    """Reference solutions for validation tests."""
    # Load or generate reference solutions
    # This would typically load from a file or generate analytically
    return {}
```

## Критерии готовности

### 1. Структура проекта
- [ ] Создана полная структура директорий согласно спецификации
- [ ] Все `__init__.py` файлы созданы и содержат правильные импорты
- [ ] Структура соответствует принципам модульности и расширяемости

### 2. Базовые классы и интерфейсы
- [ ] Реализован `AbstractSolver` с полным интерфейсом
- [ ] Реализован `PhaseField` с основными методами
- [ ] Реализован `Domain` для управления областью расчета
- [ ] Все абстрактные методы содержат `NotImplemented` вместо `pass`

### 3. Система конфигурации
- [ ] Реализован `ConfigLoader` с поддержкой JSON/YAML
- [ ] Созданы шаблоны конфигураций для всех уровней
- [ ] Реализована валидация параметров
- [ ] Система слияния с значениями по умолчанию работает

### 4. Система логирования
- [ ] Реализован `StructuredLogger` с различными уровнями
- [ ] Логирование в файлы и консоль настроено
- [ ] Структурированные сообщения для экспериментов
- [ ] Экспорт логов в различные форматы

### 5. Управление зависимостями
- [ ] `requirements.txt` создан с точными версиями
- [ ] `pyproject.toml` настроен для современной упаковки
- [ ] Виртуальное окружение `.venv` создано и активировано
- [ ] Все зависимости установлены и работают

### 6. Система тестирования
- [ ] `conftest.py` настроен с фикстурами
- [ ] Базовые тесты для всех основных компонентов созданы
- [ ] Тесты проходят успешно
- [ ] Покрытие кода тестами > 80%

### 7. Документация
- [ ] `README.md` создан с описанием проекта
- [ ] Докстринги на английском языке во всех модулях
- [ ] Примеры использования созданы
- [ ] API документация сгенерирована

### 8. Инструменты разработки
- [ ] `black` настроен и запускается без ошибок
- [ ] `flake8` настроен и проходит проверку
- [ ] `mypy` настроен и проходит проверку типов
- [ ] Pre-commit хуки настроены

### 9. CLI интерфейс
- [ ] Базовая команда `bhlff` работает
- [ ] Команды для запуска экспериментов реализованы
- [ ] Команды для анализа результатов реализованы
- [ ] Справка и документация CLI созданы

### 10. Интеграция и валидация
- [ ] Все импорты работают корректно
- [ ] Базовые тесты проходят
- [ ] Конфигурации загружаются без ошибок
- [ ] Логирование работает во всех компонентах

## Следующий шаг
После завершения Step 01 переходим к Step 02: Реализация ядра FFT решателя для фракционного оператора Рисса, который будет использовать созданную архитектуру для решения основных уравнений теории.
