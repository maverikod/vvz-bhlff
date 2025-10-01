# План рефакторинга для устранения логических дублей

**Дата**: $(date)  
**Статус**: 📋 ПЛАН  
**Аналитик**: AI Assistant

## 🎯 Цель рефакторинга

Устранить логические дубли в проекте BHLFF для улучшения архитектуры, поддерживаемости и производительности кода.

## 📋 Детальный план действий

### Этап 1: 🔴 КРИТИЧНО - ResidualComputer (1-2 дня)

#### 1.1 Анализ различий
- [ ] Сравнить интерфейсы двух ResidualComputer классов
- [ ] Выявить общую и специфичную функциональность
- [ ] Определить параметры для унификации

#### 1.2 Создание базового класса
```python
# bhlff/core/bvp/residual_computer_base.py
class ResidualComputerBase(ABC):
    """
    Base class for residual computation in BVP envelope equation.
    
    Physical Meaning:
        Provides common interface for computing residuals of the
        7D BVP envelope equation with different domain types.
    """
    
    @abstractmethod
    def compute_residual(self, envelope: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Compute residual of the envelope equation."""
        pass
    
    @abstractmethod
    def _compute_div_kappa_grad(self, envelope: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """Compute divergence of kappa times gradient."""
        pass
```

#### 1.3 Создание универсального класса
```python
# bhlff/core/bvp/residual_computer.py
class ResidualComputer(ResidualComputerBase):
    """
    Universal residual computer for BVP envelope equation.
    
    Physical Meaning:
        Computes residuals for the 7D BVP envelope equation
        with support for different domain types and configurations.
    """
    
    def __init__(self, domain: Union[Domain, Domain7D], config: Dict[str, Any]):
        self.domain = domain
        self.config = config
        self.domain_type = self._detect_domain_type()
        self._setup_parameters()
    
    def _detect_domain_type(self) -> str:
        """Detect domain type and return appropriate handler."""
        if hasattr(self.domain, 'N_phi'):
            return "7d_bvp"
        else:
            return "standard"
```

#### 1.4 Обновление зависимостей
- [ ] Обновить импорты в `envelope_solver_core.py`
- [ ] Обновить импорты в `envelope_equation/`
- [ ] Обновить тесты
- [ ] Удалить старые файлы

#### 1.5 Тестирование
- [ ] Создать unit тесты для нового класса
- [ ] Проверить совместимость с существующим кодом
- [ ] Запустить интеграционные тесты

### Этап 2: 🟡 ВАЖНО - Validation Classes (2-3 дня)

#### 2.1 Создание базового валидатора
```python
# bhlff/core/validation/base_validator.py
class BaseValidator(ABC):
    """
    Base class for solution validation.
    
    Physical Meaning:
        Provides common interface for validating solutions
        to different types of equations in the BHLFF framework.
    """
    
    @abstractmethod
    def validate_solution(self, solution: np.ndarray, source: np.ndarray, 
                         tolerance: float = 1e-12) -> Dict[str, Any]:
        """Validate solution against source."""
        pass
    
    @abstractmethod
    def check_energy_conservation(self, field: np.ndarray, 
                                 expected_energy: Optional[float] = None,
                                 tolerance: float = 1e-10) -> Dict[str, Any]:
        """Check energy conservation."""
        pass
    
    @abstractmethod
    def check_boundary_conditions(self, field: np.ndarray, 
                                 boundary_type: str = "periodic") -> Dict[str, Any]:
        """Check boundary conditions."""
        pass
```

#### 2.2 Создание специализированных валидаторов
```python
# bhlff/core/validation/fft_validator.py
class FFTValidator(BaseValidator):
    """FFT-specific validation methods."""
    
    def __init__(self, domain: Domain, parameters: Parameters, fractional_laplacian):
        self.domain = domain
        self.parameters = parameters
        self._fractional_laplacian = fractional_laplacian

# bhlff/core/validation/bvp_validator.py
class BVPValidator(BaseValidator):
    """BVP-specific validation methods."""
    
    def __init__(self, core: BVPSolverCore, parameters: Parameters7DBVP):
        self.core = core
        self.parameters = parameters
        self.domain = core.domain
```

#### 2.3 Фабрика валидаторов
```python
# bhlff/core/validation/validator_factory.py
class ValidatorFactory:
    """Factory for creating appropriate validators."""
    
    @staticmethod
    def create_validator(validator_type: str, **kwargs) -> BaseValidator:
        if validator_type == "fft":
            return FFTValidator(**kwargs)
        elif validator_type == "bvp":
            return BVPValidator(**kwargs)
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")
```

#### 2.4 Обновление зависимостей
- [ ] Обновить импорты в FFT solvers
- [ ] Обновить импорты в BVP solvers
- [ ] Обновить тесты
- [ ] Удалить старые файлы

### Этап 3: 🟡 ВАЖНО - FFT Solvers (2-3 дня)

#### 3.1 Анализ архитектуры
- [ ] Определить общую функциональность
- [ ] Выявить специфичные особенности
- [ ] Спроектировать новую архитектуру

#### 3.2 Создание базового FFT решателя
```python
# bhlff/core/fft/base_fft_solver.py
class BaseFFTSolver(ABC):
    """
    Base class for FFT-based solvers.
    
    Physical Meaning:
        Provides common interface for FFT-based solution
        of equations in 7D space-time.
    """
    
    @abstractmethod
    def solve_stationary(self, source: np.ndarray) -> np.ndarray:
        """Solve stationary equation."""
        pass
    
    @abstractmethod
    def solve_time_evolution(self, initial_field: np.ndarray, 
                           time_steps: int) -> np.ndarray:
        """Solve time evolution."""
        pass
```

#### 3.3 Создание специализированных решателей
```python
# bhlff/core/fft/fft_solver_7d_universal.py
class FFTSolver7DUniversal(BaseFFTSolver):
    """
    Universal 7D FFT solver with configurable behavior.
    
    Physical Meaning:
        Solves equations in 7D space-time with support
        for different equation types and configurations.
    """
    
    def __init__(self, domain: Domain, parameters: Parameters, 
                 equation_type: str = "fractional_laplacian"):
        self.domain = domain
        self.parameters = parameters
        self.equation_type = equation_type
        self._setup_for_equation_type()
```

#### 3.4 Фабрика решателей
```python
# bhlff/core/fft/solver_factory.py
class SolverFactory:
    """Factory for creating appropriate FFT solvers."""
    
    @staticmethod
    def create_solver(solver_type: str, domain: Domain, 
                     parameters: Parameters) -> BaseFFTSolver:
        if solver_type == "7d_universal":
            return FFTSolver7DUniversal(domain, parameters)
        elif solver_type == "7d_bvp":
            return FFTSolver7DBVP(domain, parameters)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
```

### Этап 4: 🟡 ВАЖНО - BVP Core (1-2 дня)

#### 4.1 Унификация через фасад
```python
# bhlff/core/bvp/bvp_core_unified.py
class BVPCoreUnified:
    """
    Unified BVP core with facade pattern.
    
    Physical Meaning:
        Provides single interface for all BVP operations
        including envelope solving, quench detection, and impedance computation.
    """
    
    def __init__(self, domain: Domain, config: Dict[str, Any], 
                 domain_7d: Optional[Domain7D] = None):
        self.domain = domain
        self.config = config
        self.domain_7d = domain_7d
        self._setup_components()
    
    def _setup_components(self):
        """Setup all BVP components."""
        self._operations = BVPCoreOperations(self.domain, self.config, self.domain_7d)
        self._7d_interface = BVPCore7DInterface(self.domain_7d, self.config) if self.domain_7d else None
```

#### 4.2 Обновление интерфейсов
- [ ] Создать единый интерфейс для BVP операций
- [ ] Обновить все зависимости
- [ ] Обеспечить обратную совместимость

### Этап 5: 🟢 ОПТИМИЗАЦИЯ - Parameters & Config (1 день)

#### 5.1 Унификация параметров
```python
# bhlff/core/domain/parameters_unified.py
class ParametersUnified:
    """
    Unified parameter management for BHLFF.
    
    Physical Meaning:
        Manages all parameters for the 7D phase field theory
        with support for different equation types and configurations.
    """
    
    def __init__(self, parameter_type: str, **kwargs):
        self.parameter_type = parameter_type
        self._setup_parameters(**kwargs)
    
    def _setup_parameters(self, **kwargs):
        """Setup parameters based on type."""
        if self.parameter_type == "standard":
            self._setup_standard_parameters(**kwargs)
        elif self.parameter_type == "7d_bvp":
            self._setup_7d_bvp_parameters(**kwargs)
```

#### 5.2 Унификация конфигурации
```python
# bhlff/core/domain/config_unified.py
class ConfigUnified:
    """
    Unified configuration management.
    
    Physical Meaning:
        Manages configuration parameters for different
        components of the BHLFF framework.
    """
    
    def __init__(self, config_type: str, **kwargs):
        self.config_type = config_type
        self._setup_config(**kwargs)
```

## 🧪 План тестирования

### Unit тесты
- [ ] Тесты для каждого нового класса
- [ ] Тесты совместимости
- [ ] Тесты производительности

### Интеграционные тесты
- [ ] Тесты полного пайплайна
- [ ] Тесты с реальными данными
- [ ] Тесты регрессии

### Тесты покрытия
- [ ] Покрытие кода > 90%
- [ ] Покрытие всех новых методов
- [ ] Покрытие edge cases

## 📊 Метрики успеха

### Количественные метрики
- **Устранение дублирования**: ~530 строк кода
- **Уменьшение классов**: -6 дублирующих классов
- **Улучшение покрытия**: > 90% покрытие тестами

### Качественные метрики
- **Единообразие**: Стандартизированные интерфейсы
- **Простота**: Меньше классов для изучения
- **Надежность**: Меньше мест для ошибок
- **Производительность**: Оптимизированные алгоритмы

## 🚀 Временной план

### Неделя 1
- **День 1-2**: ResidualComputer рефакторинг
- **День 3-4**: Validation classes рефакторинг
- **День 5**: Тестирование и отладка

### Неделя 2
- **День 1-2**: FFT Solvers рефакторинг
- **День 3**: BVP Core рефакторинг
- **День 4**: Parameters & Config оптимизация
- **День 5**: Финальное тестирование

## ⚠️ Риски и митигация

### Риски
1. **Нарушение существующего API**
   - Митигация: Обеспечить обратную совместимость
2. **Снижение производительности**
   - Митигация: Профилирование и оптимизация
3. **Ошибки в рефакторинге**
   - Митигация: Тщательное тестирование

### План отката
- [ ] Сохранить резервные копии
- [ ] Подготовить план отката для каждого этапа
- [ ] Документировать все изменения

## ✅ Критерии завершения

### Функциональные критерии
- [ ] Все дублирующие классы устранены
- [ ] Все тесты проходят
- [ ] Покрытие кода > 90%
- [ ] Производительность не ухудшилась

### Качественные критерии
- [ ] Код соответствует стандартам проекта
- [ ] Документация обновлена
- [ ] Архитектура стала более чистой
- [ ] Поддерживаемость улучшилась

## 📝 Заключение

Данный план рефакторинга позволит устранить все выявленные логические дубли в проекте BHLFF, улучшить архитектуру и повысить поддерживаемость кода. План рассчитан на 2 недели работы с поэтапным подходом и тщательным тестированием на каждом этапе.
