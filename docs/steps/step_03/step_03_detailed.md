# Step 03: Детализированное задание - Временные интеграторы для динамических задач

## Физическая основа и теоретический контекст

### 1. 7D Phase Field Dynamic Equation

In the framework of 7D phase field theory, the phase field dynamics is described by the equation:

```
∂a/∂t + ν(-Δ)^β a + λa = s(x,φ,t)
```

where:
- `a(x,φ,t)` - **U(1)³ phase vector field** in 7D space-time M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
- **Phase vector structure**: a = (a₁, a₂, a₃) with three U(1) components
- `ν > 0` - diffusion coefficient (related to phase velocity c_φ)
- `β ∈ (0,2)` - fractional order of Riesz operator
- `λ ≥ 0` - damping parameter (related to dissipation)
- `s(x,φ,t)` - source (external excitations, initial conditions)

### 2. 7D Spectral Form of Equation

In 7D k-space (after FFT) the equation takes the form:

```
∂â/∂t + (ν|k|^(2β) + λ)â = ŝ(k_x, k_φ, k_t)
```

where:
- `â(k_x, k_φ, k_t)` and `ŝ(k_x, k_φ, k_t)` - spectral components of field and source
- **7D wave vector**: |k|² = |k_x|² + |k_φ|² + k_t²
- **k_x** - spatial wave vectors (3D)
- **k_φ** - phase wave vectors (3D)
- **k_t** - temporal frequency

### 3. Физический смысл параметров

- **ν (коэффициент диффузии)**: Связан с фазовой скоростью c_φ >> c, обеспечивает сверхсветовую передачу фазы
- **β (фракционный порядок)**: Определяет характер дальнодействия и степенные хвосты
- **λ (параметр затухания)**: Обеспечивает диссипацию и стабилизацию системы

## Математические требования к интеграторам

### 1. Точность решения

#### 1.1. Аналитические решения для гармонических источников
Для источника вида `s(x,t) = s₀(x)e^(-iωt)` точное решение:

```
â(k,t) = â₀(k)e^(-(ν|k|^(2β)+λ)t) + ŝ₀(k)/(ν|k|^(2β)+λ+iω)(1-e^(-(ν|k|^(2β)+λ+iω)t))
```

#### 1.2. Требования к точности
- Локальная ошибка: O(dt^p), где p ≥ 2
- Глобальная ошибка: O(dt^(p-1))
- Сохранение энергии в консервативном случае (λ=0)
- Правильное воспроизведение степенных хвостов

### 2. Устойчивость

#### 2.1. CFL условия
Для явных схем:
```
dt ≤ c / (ν k_max^(2β) + λ)
```

где `k_max` - максимальная частота в спектре.

#### 2.2. Анализ устойчивости
- Условие фон Неймана для спектрального радиуса
- Контроль численных неустойчивостей
- Обработка жестких систем (большие ν или λ)

### 3. Адаптивность

#### 3.1. Контроль ошибки
- Оценка локальной ошибки усечения
- Адаптивный выбор шага по времени
- Контроль глобальной ошибки

#### 3.2. Оптимизация производительности
- Минимизация количества вычислений FFT
- Кэширование спектральных коэффициентов
- Векторизованные операции

## Детализированные требования к реализации

### 1. Базовый класс TimeIntegrator

#### 1.1. Абстрактный интерфейс
```python
class TimeIntegrator(ABC):
    """
    Abstract base class for time integrators in 7D phase field theory.
    
    Physical Meaning:
        Provides the interface for temporal evolution of phase fields
        in 7D space-time, ensuring proper handling of fractional
        Laplacian dynamics and phase coherence.
    """
    
    @abstractmethod
    def integrate(self, initial_field: np.ndarray, 
                  source_field: Callable[[float], np.ndarray],
                  time_range: Tuple[float, float]) -> np.ndarray:
        """
        Integrate the phase field equation over time.
        
        Physical Meaning:
            Evolves the phase field according to the fractional
            Laplacian equation, maintaining phase coherence and
            topological stability.
        """
        pass
    
    @abstractmethod
    def validate_stability(self) -> bool:
        """
        Validate numerical stability of the integrator.
        
        Physical Meaning:
            Ensures that the time step and numerical scheme
            maintain the physical stability of the phase field
            configuration.
        """
        pass
```

#### 1.2. Общие методы
- `get_time_step()` - получение текущего шага
- `set_time_step(dt)` - установка шага с валидацией
- `get_energy()` - вычисление энергии поля
- `get_phase_coherence()` - оценка когерентности фазы

### 2. ВБП-модуляционный интегратор (BVPModulationIntegrator)

#### 2.1. Математическая основа
Для ВБП-модуляций с квенчами уравнение огибающей:
```
∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x)
```
где:
- `κ(|a|) = κ₀ + κ₂|a|²` - нелинейная жёсткость БВП
- `χ(|a|) = χ' + iχ''(|a|)` - эффективная восприимчивость с квенчами
- `χ''(|a|)` - потери, рост которых фиксирует квенч

#### 2.2. Реализация
```python
class BVPModulationIntegrator(TimeIntegrator):
    """
    BVP modulation integrator for envelope evolution with quenches.
    
    Physical Meaning:
        Integrates the envelope of the Base High-Frequency Field (BVP),
        handling quench events when local thresholds are reached.
        All observed "modes" are envelope modulations and beatings of BVP.
    """
    
    def integrate(self, initial_envelope: np.ndarray,
                  source_field: Callable[[float], np.ndarray],
                  time_range: Tuple[float, float]) -> np.ndarray:
        """
        Integration of BVP envelope with quench detection.
        
        Mathematical Foundation:
            Solves the envelope equation for BVP with nonlinear
            stiffness κ(|a|) and susceptibility χ(|a|) that includes
            quench-induced losses χ''(|a|).
        """
        # Implementation with BVP envelope evolution and quench detection
        pass
```

#### 2.3. Особенности
- Интеграция огибающей ВБП с нелинейной жёсткостью
- Детекция квенчей при достижении пороговых значений
- Обработка диссипативного "сброса" энергии БВП в среду
- Сохранение высокочастотной структуры носителя

### 3. Crank-Nicolson интегратор (CrankNicolsonIntegrator)

#### 3.1. Математическая схема
```
(â^(n+1) - â^n)/dt + (ν|k|^(2β) + λ)(â^(n+1) + â^n)/2 = (ŝ^(n+1) + ŝ^n)/2
```

#### 3.2. Реализация
```python
class CrankNicolsonIntegrator(TimeIntegrator):
    """
    Crank-Nicolson integrator for general source terms.
    
    Physical Meaning:
        Provides second-order accurate temporal evolution with
        unconditional stability, suitable for nonlinear dynamics.
    """
    
    def integrate(self, initial_field: np.ndarray,
                  source_field: Callable[[float], np.ndarray],
                  time_range: Tuple[float, float]) -> np.ndarray:
        """
        Second-order accurate integration with unconditional stability.
        
        Mathematical Foundation:
            Implements Crank-Nicolson scheme with spectral operator
            for fractional Laplacian, ensuring stability and accuracy.
        """
        # Implementation with Crank-Nicolson scheme
        pass
```

#### 3.3. Особенности
- Второй порядок точности по времени
- Безусловная устойчивость
- Эффективная реализация через FFT
- Поддержка нелинейных источников

### 4. Адаптивный интегратор (AdaptiveIntegrator)

#### 4.1. Алгоритм адаптации
- Оценка локальной ошибки усечения
- Автоматический выбор шага по времени
- Контроль устойчивости
- Оптимизация производительности

#### 4.2. Реализация
```python
class AdaptiveIntegrator(TimeIntegrator):
    """
    Adaptive integrator with error control and stability monitoring.
    
    Physical Meaning:
        Automatically adjusts time step to maintain accuracy while
        ensuring numerical stability of phase field evolution.
    """
    
    def integrate(self, initial_field: np.ndarray,
                  source_field: Callable[[float], np.ndarray],
                  time_range: Tuple[float, float]) -> np.ndarray:
        """
        Adaptive integration with automatic time step control.
        
        Mathematical Foundation:
            Uses embedded Runge-Kutta methods with error estimation
            and automatic step size adjustment for optimal performance.
        """
        # Implementation with adaptive step control
        pass
```

#### 4.3. Особенности
- Автоматический контроль ошибки
- Адаптивный выбор шага
- Мониторинг устойчивости
- Оптимизация производительности

## Конфигурация и параметры

### 1. Структура конфигурации
```json
{
    "time_integrator": {
        "type": "exponential|cn|adaptive",
        "dt": 1e-3,
        "T": 1.0,
        "tolerance": 1e-8,
        "max_dt": 1e-2,
        "min_dt": 1e-6,
        "safety_factor": 0.9,
        "stability_check": true,
        "energy_conservation": true,
        "phase_coherence_check": true
    },
    "physics": {
        "nu": 1.0,
        "beta": 1.0,
        "lambda": 0.0
    },
    "domain": {
        "L": 1.0,
        "N": 256,
        "dimensions": 3
    }
}
```

### 2. Параметры валидации
- `tolerance` - допустимая ошибка
- `max_dt` - максимальный шаг
- `min_dt` - минимальный шаг
- `safety_factor` - коэффициент безопасности
- `stability_check` - проверка устойчивости
- `energy_conservation` - контроль энергии
- `phase_coherence_check` - проверка когерентности

## Тесты и валидация

### 1. Аналитические тесты

#### 1.1. ВБП-модуляционный источник
```python
def test_bvp_modulation_source():
    """
    Test with BVP modulation source s(x,t) = s₀(x) modulating BVP envelope.
    
    Physical Meaning:
        Validates BVP envelope evolution with source modulations,
        crucial for understanding how all observed "modes" are
        envelope modulations and beatings of the Base High-Frequency Field.
    """
    # Test with BVP envelope modulation
    pass
```

#### 1.2. Квенч-индуцированное затухание
```python
def test_quench_induced_dissipation():
    """
    Test quench-induced dissipation when local thresholds are reached.
    
    Physical Meaning:
        Validates how BVP dissipatively "dumps" energy into the medium
        when local thresholds (amplitude/detuning/gradient) are reached,
        causing growth of losses and change in Q-factor.
    """
    # Test quench detection and energy dissipation
    pass
```

#### 1.3. Установившийся режим
```python
def test_steady_state():
    """
    Test convergence to steady state.
    
    Physical Meaning:
        Validates that the system reaches equilibrium
        when driven by constant sources.
    """
    # Test steady state convergence
    pass
```

### 2. Численные тесты

#### 2.1. Сходимость по времени
```python
def test_time_convergence():
    """
    Test convergence order with respect to time step.
    
    Physical Meaning:
        Validates that the numerical scheme achieves
        the expected order of accuracy.
    """
    # Test convergence order
    pass
```

#### 2.2. Сохранение энергии
```python
def test_energy_conservation():
    """
    Test energy conservation for conservative systems.
    
    Physical Meaning:
        Validates that energy is properly conserved
        in the absence of dissipation (λ=0).
    """
    # Test energy conservation
    pass
```

#### 2.3. Устойчивость
```python
def test_stability():
    """
    Test numerical stability for large time steps.
    
    Physical Meaning:
        Validates that the integrator remains stable
        under challenging numerical conditions.
    """
    # Test stability
    pass
```

### 3. Граничные случаи

#### 3.1. β→0 (обычная диффузия)
```python
def test_beta_zero():
    """
    Test limit β→0 (ordinary diffusion).
    
    Physical Meaning:
        Validates that the integrator correctly
        handles the limit to ordinary diffusion.
    """
    # Test β→0 limit
    pass
```

#### 3.2. β→2 (волновое уравнение)
```python
def test_beta_two():
    """
    Test limit β→2 (wave equation).
    
    Physical Meaning:
        Validates that the integrator correctly
        handles the limit to wave equation.
    """
    # Test β→2 limit
    pass
```

#### 3.3. λ=0 (консервативная система)
```python
def test_lambda_zero():
    """
    Test conservative system (λ=0).
    
    Physical Meaning:
        Validates that the integrator properly
        handles conservative dynamics.
    """
    # Test λ=0 case
    pass
```

## CFL условия для фракционных операторов

### 1. Теоретические основы
```python
class CFLFractionalOperator:
    """
    CFL условия для фракционных операторов в 7D.
    
    Physical Meaning:
        Определяет условия устойчивости для временных интеграторов
        при работе с фракционным оператором Рисса (-Δ)^β.
        
    Mathematical Foundation:
        - Классическое CFL: dt ≤ C * dx² / (2*ν)
        - Фракционное CFL: dt ≤ C * dx^(2β) / (2*ν)
        - 7D адаптация: учет всех пространственных размерностей
    """
    
    @staticmethod
    def compute_cfl_condition(beta: float, nu: float, dx: float, 
                            safety_factor: float = 0.5) -> float:
        """
        Вычисление CFL условия для фракционного оператора.
        
        Physical Meaning:
            Определяет максимальный временной шаг, обеспечивающий
            численную устойчивость для фракционного оператора.
            
        Mathematical Foundation:
            dt_max = C * dx^(2β) / (2*ν)
            где C - коэффициент безопасности
            
        Args:
            beta: Фракционный порядок (0 < β < 2)
            nu: Коэффициент диффузии
            dx: Размер пространственной сетки
            safety_factor: Коэффициент безопасности
            
        Returns:
            Максимальный временной шаг
        """
        if beta <= 0 or beta >= 2:
            raise ValueError("Фракционный порядок β должен быть в интервале (0, 2)")
        
        if nu <= 0:
            raise ValueError("Коэффициент диффузии ν должен быть положительным")
        
        # Фракционное CFL условие
        dt_max = safety_factor * (dx ** (2 * beta)) / (2 * nu)
        
        return dt_max
    
    @staticmethod
    def compute_7d_cfl_condition(beta: float, nu: float, dx: Tuple[float, ...],
                               safety_factor: float = 0.5) -> float:
        """
        Вычисление CFL условия для 7D фракционного оператора.
        
        Physical Meaning:
            Адаптирует CFL условие для 7D пространства, учитывая
            все пространственные размерности.
            
        Mathematical Foundation:
            dt_max = C * min(dx_i^(2β)) / (2*ν)
            где dx_i - размеры сетки по всем измерениям
            
        Args:
            beta: Фракционный порядок
            nu: Коэффициент диффузии
            dx: Размеры пространственной сетки по всем измерениям
            safety_factor: Коэффициент безопасности
            
        Returns:
            Максимальный временной шаг для 7D
        """
        # Находим минимальный размер сетки
        min_dx = min(dx)
        
        # Применяем фракционное CFL условие
        dt_max = CFLFractionalOperator.compute_cfl_condition(
            beta, nu, min_dx, safety_factor
        )
        
        return dt_max
    
    @staticmethod
    def adaptive_time_step(current_dt: float, cfl_condition: float,
                          error_estimate: float, target_error: float) -> float:
        """
        Адаптивный выбор временного шага.
        
        Physical Meaning:
            Адаптирует временной шаг на основе CFL условия и
            оценки ошибки для обеспечения устойчивости и точности.
            
        Mathematical Foundation:
            dt_new = min(dt_cfl, dt_error)
            где dt_cfl - ограничение по CFL, dt_error - ограничение по ошибке
            
        Args:
            current_dt: Текущий временной шаг
            cfl_condition: CFL ограничение
            error_estimate: Оценка ошибки
            target_error: Целевая ошибка
            
        Returns:
            Новый временной шаг
        """
        # Ограничение по CFL
        dt_cfl = cfl_condition
        
        # Ограничение по ошибке
        if error_estimate > 0:
            dt_error = current_dt * (target_error / error_estimate) ** (1/3)
        else:
            dt_error = current_dt * 1.2  # Увеличиваем шаг
        
        # Выбираем минимальный шаг
        dt_new = min(dt_cfl, dt_error)
        
        return dt_new
```

### 2. Анализ фон Неймана для фракционных операторов
```python
class VonNeumannAnalysis:
    """
    Анализ фон Неймана для фракционных операторов.
    
    Physical Meaning:
        Анализирует устойчивость численных схем для фракционного
        оператора через анализ роста ошибок в спектральном пространстве.
        
    Mathematical Foundation:
        - Коэффициент усиления: G(k) = |1 + dt*L_β(k)|
        - Условие устойчивости: |G(k)| ≤ 1 для всех k
        - Фракционная адаптация: учет спектральных свойств (-Δ)^β
    """
    
    @staticmethod
    def amplification_factor(k: float, beta: float, nu: float, dt: float) -> complex:
        """
        Вычисление коэффициента усиления.
        
        Physical Meaning:
            Вычисляет коэффициент усиления ошибки для волнового
            числа k при использовании фракционного оператора.
            
        Mathematical Foundation:
            G(k) = 1 - dt * ν * |k|^(2β)
            
        Args:
            k: Волновое число
            beta: Фракционный порядок
            nu: Коэффициент диффузии
            dt: Временной шаг
            
        Returns:
            Коэффициент усиления
        """
        return 1 - dt * nu * (abs(k) ** (2 * beta))
    
    @staticmethod
    def stability_condition(beta: float, nu: float, dt: float, 
                          k_max: float) -> bool:
        """
        Проверка условия устойчивости.
        
        Physical Meaning:
            Проверяет, что численная схема устойчива для всех
            волновых чисел в диапазоне [0, k_max].
            
        Args:
            beta: Фракционный порядок
            nu: Коэффициент диффузии
            dt: Временной шаг
            k_max: Максимальное волновое число
            
        Returns:
            True если схема устойчива
        """
        # Проверяем устойчивость для максимального k
        g_max = VonNeumannAnalysis.amplification_factor(k_max, beta, nu, dt)
        
        return abs(g_max) <= 1.0
```

## Производительность и оптимизация

### 1. Оптимизация FFT операций
- Кэширование спектральных коэффициентов
- Оптимизация планов FFT
- Векторизованные операции
- Параллельные вычисления

### 2. Управление памятью
- Эффективное использование памяти
- Избежание копирования больших массивов
- Оптимизация доступа к данным

### 3. Масштабируемость
- Поддержка больших размеров сетки
- Эффективная работа с многомерными полями
- Оптимизация для различных архитектур

### 4. Адаптивные временные шаги
- CFL условия для фракционных операторов
- Анализ фон Неймана для устойчивости
- Адаптивный контроль ошибки

## Критерии готовности

### 1. Функциональные требования
- [ ] Реализован базовый класс TimeIntegrator
- [ ] Реализован ExponentialIntegrator
- [ ] Реализован CrankNicolsonIntegrator
- [ ] Реализован AdaptiveIntegrator
- [ ] Все интеграторы поддерживают 7D поля
- [ ] Корректная обработка фракционного оператора

### 2. Точность и стабильность
- [ ] Все аналитические тесты проходят
- [ ] Численные тесты показывают правильную сходимость
- [ ] Адаптивный контроль ошибки работает
- [ ] Устойчивость при больших шагах
- [ ] Сохранение энергии в консервативном случае

### 3. Производительность
- [ ] Производительность соответствует требованиям
- [ ] Эффективное использование FFT
- [ ] Оптимизированное управление памятью
- [ ] Поддержка параллельных вычислений

### 4. Документация и тестирование
- [ ] Полная документация всех классов
- [ ] Примеры использования созданы
- [ ] Все тесты покрывают граничные случаи
- [ ] Валидация против аналитических решений

### 5. Интеграция
- [ ] Совместимость с FFT решателем
- [ ] Корректная работа с доменом
- [ ] Поддержка конфигурационных файлов
- [ ] Интеграция с системой логирования

## Следующий шаг

После завершения Step 03 переходим к **Step 04: Реализация валидационных тестов уровня A (базовые решатели)**, где будут созданы комплексные тесты для проверки корректности работы всех компонентов системы.

## Дополнительные требования

### 1. Соответствие стандартам проекта
- Все классы должны содержать подробные докстринги с физическим смыслом
- Размер файлов не должен превышать 400 строк
- Использование только английского языка в коде и документации
- Следование принципам декларативного программирования

### 2. Интеграция с 7D теорией
- Поддержка 7D пространства-времени M₇
- Корректная обработка фазовых координат
- Учет топологических ограничений
- Сохранение фазовой когерентности

### 3. Научная обоснованность
- Все алгоритмы должны иметь четкое физическое обоснование
- Математические методы должны соответствовать теории
- Валидация против известных аналитических решений
- Документирование всех физических предположений
