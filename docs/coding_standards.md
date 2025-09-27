# Стандарты проекта BHLFF для оформления кода

## 1. Структура докстрингов

### 1.1. Докстринг файла (обязательный в начале каждого файла)
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Brief description of the module's purpose and its role in the 7D phase field theory.

Detailed description of the module's functionality, including:
- Physical meaning and theoretical background
- Key algorithms and methods implemented
- Dependencies and relationships with other modules
- Usage examples and typical workflows

Theoretical Background:
    Brief explanation of the physical principles this module implements,
    including relevant equations and concepts from the 7D phase field theory.

Example:
    Basic usage example showing how to use the main classes/functions.
"""
```

### 1.2. Докстринг класса
```python
class ExampleClass:
    """
    Brief description of the class purpose.
    
    Detailed description of the class functionality, including:
    - Physical meaning and role in the 7D theory
    - Key properties and their physical interpretation
    - Main methods and their purposes
    - Relationships with other classes
    
    Physical Meaning:
        Detailed explanation of what this class represents in the context
        of the 7D phase field theory, including relevant equations and
        physical principles.
        
    Mathematical Foundation:
        Key equations and mathematical concepts implemented by this class.
        
    Attributes:
        attr1 (type): Description of the attribute and its physical meaning.
        attr2 (type): Description of the attribute and its physical meaning.
        
    Example:
        Basic usage example.
    """
```

### 1.3. Докстринг свойства
```python
@property
def property_name(self) -> type:
    """
    Brief description of the property.
    
    Physical Meaning:
        Detailed explanation of what this property represents in the
        context of the 7D phase field theory.
        
    Mathematical Definition:
        Mathematical expression or definition if applicable.
        
    Returns:
        type: Description of the return value and its physical meaning.
    """
```

### 1.4. Докстринг метода
```python
def method_name(self, param1: type, param2: type) -> return_type:
    """
    Brief description of the method's purpose.
    
    Physical Meaning:
        Detailed explanation of what this method computes or represents
        in the context of the 7D phase field theory.
        
    Mathematical Foundation:
        Key equations or algorithms implemented by this method.
        
    Args:
        param1 (type): Description of the parameter and its physical meaning.
        param2 (type): Description of the parameter and its physical meaning.
        
    Returns:
        return_type: Description of the return value and its physical meaning.
        
    Raises:
        ExceptionType: Description of when this exception is raised.
        
    Example:
        Basic usage example.
    """
```

## 2. Критические требования к реализации

### 2.1. Запрет на NotImplemented
- **КРИТИЧНО**: `NotImplemented` разрешен ТОЛЬКО в абстрактных методах
- Все остальные методы должны содержать полную реализацию
- Заглушки типа `raise NotImplementedError("Not implemented")` запрещены

### 2.2. Запрет на pass
- **КРИТИЧНО**: `pass` запрещен как единственное выражение в методе
- Каждый метод должен содержать реальную логику
- Исключение: только в абстрактных методах базовых классов

### 2.3. Запрет на fallback отступления
- **КРИТИЧНО**: Запрещены упрощения алгоритма "для простоты"
- Все методы должны реализовывать полную функциональность
- Временные заглушки должны быть заменены на полную реализацию

## 3. Стандарты именования

### 3.1. Переменные и атрибуты
```python
# Константы - ВЕРХНИЙ_РЕГИСТР
PHASE_VELOCITY = 1e15
LIGHT_SPEED = 299792458

# Обычные переменные - snake_case
phase_field = np.array(...)
topological_charge = 1.0
energy_density = compute_energy()

# Приватные атрибуты - с префиксом _
_private_field = None
_internal_state = {}

# Защищенные атрибуты - с префиксом __
__protected_method = None
```

### 3.2. Методы и функции
```python
# Публичные методы - snake_case
def compute_phase_field(self) -> np.ndarray:
    pass

def calculate_topological_charge(self) -> float:
    pass

# Приватные методы - с префиксом _
def _internal_computation(self) -> None:
    pass

# Защищенные методы - с префиксом __
def __protected_method(self) -> None:
    pass
```

### 3.3. Классы
```python
# Классы - PascalCase
class PhaseFieldSolver:
    pass

class TopologicalDefect:
    pass

class EnergyFunctional:
    pass
```

### 3.4. Файлы и модули
```python
# Файлы - snake_case
phase_field_solver.py
topological_defect.py
energy_functional.py

# Модули - snake_case
src/bhlff/core/phase/
src/bhlff/models/level_a/
src/bhlff/utils/visualization/
```

### 3.5. Конфигурационные файлы
```python
# Конфигурации - snake_case с описательным именем
level_a_validation.json
level_b_power_law.json
level_c_boundaries.json
```

## 4. Стандарт задания начальных значений

### 4.1. Порядок приоритета
```python
class ConfigManager:
    """
    Configuration manager with priority-based value assignment.
    
    Physical Meaning:
        Manages configuration parameters for the 7D phase field theory
        simulations, ensuring proper parameter hierarchy and validation.
        
    Priority Order:
        1. CLI arguments (highest priority)
        2. Environment variables
        3. Configuration file (lowest priority)
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_file (str): Path to the configuration file.
        """
        self.config_file = config_file
        self._config = {}
        self._load_config()
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get parameter value following priority order.
        
        Physical Meaning:
            Retrieves configuration parameters in the correct priority order,
            ensuring that CLI arguments override environment variables,
            which override configuration file values.
            
        Args:
            key (str): Parameter name.
            default (Any): Default value if parameter not found.
            
        Returns:
            Any: Parameter value following priority order.
        """
        # 1. Check CLI arguments (highest priority)
        cli_value = self._get_cli_argument(key)
        if cli_value is not None:
            return cli_value
        
        # 2. Check environment variables
        env_value = self._get_environment_variable(key)
        if env_value is not None:
            return env_value
        
        # 3. Check configuration file
        config_value = self._config.get(key)
        if config_value is not None:
            return config_value
        
        # 4. Return default value
        return default
```

### 4.2. Формат конфигурационного файла
```json
{
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
        "save_fields": true,
        "save_spectra": true,
        "save_analysis": true,
        "format": "hdf5"
    }
}
```

## 5. Ограничения размера файлов

### 5.1. Критические требования
- **КРИТИЧНО**: Максимальный размер файла - 400 строк
- **ЦЕЛЬ**: Размер файла не более 350 строк
- При превышении лимита файл должен быть разбит на модули

### 5.2. Стратегии разбиения
```python
# Вместо одного большого файла:
# phase_field_solver.py (500+ строк) - НЕДОПУСТИМО

# Разбить на несколько модулей:
# phase_field_solver.py (основной класс)
# phase_field_core.py (базовые операции)
# phase_field_utils.py (вспомогательные функции)
# phase_field_validation.py (методы валидации)
```

### 5.3. Пример разбиения
```python
# phase_field_solver.py (основной класс, ~200 строк)
class PhaseFieldSolver:
    """
    Main phase field solver class.
    
    Physical Meaning:
        Solves the phase field equations in 7D space-time,
        implementing the core algorithms for phase field evolution.
    """
    
    def __init__(self, domain: Domain, parameters: Dict[str, Any]):
        """Initialize solver."""
        self.domain = domain
        self.parameters = parameters
        self._core_ops = PhaseFieldCore(domain)
        self._validator = PhaseFieldValidator(domain)
    
    def solve(self, source: np.ndarray) -> np.ndarray:
        """Main solving method."""
        # Основная логика решения
        pass

# phase_field_core.py (базовые операции, ~150 строк)
class PhaseFieldCore:
    """
    Core operations for phase field computations.
    
    Physical Meaning:
        Implements fundamental mathematical operations for phase field
        calculations, including FFT operations and spectral methods.
    """
    
    def compute_spectral_operator(self, field: np.ndarray) -> np.ndarray:
        """Compute spectral operator."""
        # Реализация спектральных операций
        pass

# phase_field_utils.py (вспомогательные функции, ~100 строк)
def compute_energy_density(field: np.ndarray) -> np.ndarray:
    """
    Compute energy density of the phase field.
    
    Physical Meaning:
        Calculates the energy density distribution in the phase field,
        representing the local energy content of the field configuration.
    """
    # Реализация вычисления плотности энергии
    pass
```

## 6. Примеры правильного оформления

### 6.1. Правильный класс
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Phase field solver implementation for the 7D phase field theory.

This module implements the core phase field solver that solves the
fractional Laplacian equation in 7D space-time, representing the
evolution of phase field configurations.

Theoretical Background:
    The phase field represents the fundamental field in 7D space-time
    M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ, where the evolution is governed by the
    fractional Laplacian operator L_β = μ(-Δ)^β + λ.

Example:
    >>> solver = PhaseFieldSolver(domain, parameters)
    >>> solution = solver.solve(source_field)
"""

import numpy as np
from typing import Dict, Any, Optional
from ..base.abstract_solver import AbstractSolver

class PhaseFieldSolver(AbstractSolver):
    """
    Phase field solver for 7D space-time theory.
    
    Physical Meaning:
        Solves the phase field evolution equation in 7D space-time,
        representing the dynamics of phase field configurations that
        give rise to particle-like structures through topological
        defects and phase coherence.
        
    Mathematical Foundation:
        Implements the fractional Laplacian equation:
        L_β a = μ(-Δ)^β a + λa = s(x,t)
        where β ∈ (0,2) is the fractional order, μ > 0 is the
        diffusion coefficient, and λ ≥ 0 is the damping parameter.
        
    Attributes:
        domain (Domain): Computational domain for the simulation.
        parameters (Dict[str, Any]): Solver parameters including
            μ, β, λ, and numerical settings.
        _fft_plan (FFTPlan): Pre-computed FFT plan for efficiency.
        _spectral_coeffs (np.ndarray): Spectral coefficients for
            the fractional Laplacian operator.
    """
    
    def __init__(self, domain: 'Domain', parameters: Dict[str, Any]):
        """
        Initialize phase field solver.
        
        Physical Meaning:
            Sets up the solver with the computational domain and
            physical parameters, pre-computing spectral coefficients
            for efficient solution of the fractional Laplacian equation.
            
        Args:
            domain (Domain): Computational domain with grid information.
            parameters (Dict[str, Any]): Dictionary containing:
                - mu (float): Diffusion coefficient μ > 0
                - beta (float): Fractional order β ∈ (0,2)
                - lambda (float): Damping parameter λ ≥ 0
                - precision (str): Numerical precision ('float64')
        """
        super().__init__(domain, parameters)
        self._setup_spectral_coefficients()
        self._setup_fft_plan()
    
    def solve(self, source: np.ndarray) -> np.ndarray:
        """
        Solve the phase field equation for given source.
        
        Physical Meaning:
            Computes the phase field configuration that satisfies
            the fractional Laplacian equation with the given source
            term, representing the response of the phase field to
            external excitations or initial conditions.
            
        Mathematical Foundation:
            Solves L_β a = s in spectral space:
            â(k) = ŝ(k) / (μ|k|^(2β) + λ)
            where k is the wave vector and |k| is its magnitude.
            
        Args:
            source (np.ndarray): Source term s(x) in real space.
                Represents external excitations or initial conditions
                that drive the phase field evolution.
                
        Returns:
            np.ndarray: Solution field a(x) in real space.
                Represents the phase field configuration that
                satisfies the equation and describes the spatial
                distribution of phase values.
                
        Raises:
            ValueError: If source has incompatible shape with domain.
            RuntimeError: If FFT operations fail.
        """
        if source.shape != self.domain.shape:
            raise ValueError(f"Source shape {source.shape} incompatible with domain shape {self.domain.shape}")
        
        # Transform to spectral space
        source_spectral = np.fft.fftn(source)
        
        # Apply spectral operator
        solution_spectral = source_spectral / self._spectral_coeffs
        
        # Transform back to real space
        solution = np.fft.ifftn(solution_spectral)
        
        return solution.real
    
    def _setup_spectral_coefficients(self) -> None:
        """
        Setup spectral coefficients for fractional Laplacian.
        
        Physical Meaning:
            Pre-computes the spectral representation of the fractional
            Laplacian operator, which is essential for efficient
            solution of the equation in spectral space.
        """
        mu = self.parameters['mu']
        beta = self.parameters['beta']
        lambda_param = self.parameters['lambda']
        
        # Compute wave vectors
        kx = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        ky = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        kz = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Compute spectral coefficients
        self._spectral_coeffs = mu * (k_magnitude ** (2 * beta)) + lambda_param
        
        # Handle k=0 mode
        if lambda_param == 0:
            self._spectral_coeffs[0, 0, 0] = 1.0  # Avoid division by zero
    
    def _setup_fft_plan(self) -> None:
        """
        Setup FFT plan for efficient computations.
        
        Physical Meaning:
            Pre-computes FFT plans to optimize the spectral
            transformations required for solving the phase field
            equation efficiently.
        """
        # Implementation of FFT plan setup
        pass
```

### 6.2. Неправильный класс (нарушения стандартов)
```python
# НЕПРАВИЛЬНО - нарушает все стандарты
class BadSolver:
    def __init__(self):
        pass  # КРИТИЧНО: pass запрещен
    
    def solve(self, source):
        raise NotImplementedError("Not implemented")  # КРИТИЧНО: NotImplemented запрещен
    
    def compute(self):
        # Упрощение для простоты - КРИТИЧНО: fallback запрещен
        return 0
```

## 7. Проверка соответствия стандартам

### 7.1. Автоматические проверки
```bash
# Проверка размера файлов
find src/ -name "*.py" -exec wc -l {} + | awk '$1 > 400 {print "ERROR: " $2 " has " $1 " lines"}'

# Проверка на pass и NotImplemented
grep -r "pass$" src/ --include="*.py" | grep -v "abstract"
grep -r "NotImplemented" src/ --include="*.py" | grep -v "abstract"

# Проверка докстрингов
python -m pydocstyle src/
```

### 7.2. Ручные проверки
- [ ] Все файлы содержат докстринг с автором и email
- [ ] Все классы имеют подробные докстринги с физическим смыслом
- [ ] Все методы описаны с указанием физического смысла
- [ ] Нет использования `pass` или `NotImplemented` в неабстрактных методах
- [ ] Размер всех файлов не превышает 400 строк
- [ ] Конфигурации используют только JSON формат
- [ ] Именование соответствует стандартам (snake_case, PascalCase)
