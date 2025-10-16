# План доработок для полной готовности к проверке гипотезы А

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com  
**Date:** 2024-12-19

## 📊 Текущий статус готовности: 60%

### ✅ Что готово:
- Базовая инфраструктура BVP Core Framework
- 7D Domain и Parameters реализованы
- FFT Solver базовая версия частично работает
- Fractional Laplacian оператор реализован
- Spectral Operations функционируют
- 50 из 95 тестов уровня A проходят

### ❌ Критические проблемы:
- **28 тестов падают** из-за отсутствия метода `solve()` в `FFTSolver7DBasic`
- **17 тестов падают** из-за неправильной инициализации `FractionalLaplacian`
- Отсутствует полная интеграция BVP Core
- Quench detection не интегрирован
- U(1)³ Phase Vector не полностью работает

---

## 🎯 ПОШАГОВЫЙ ПЛАН ДОРАБОТОК

### ЭТАП 1: КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ (Приоритет: ВЫСОКИЙ)

#### 1.1 Исправить FFTSolver7DBasic
**Проблема:** Отсутствует метод `solve()` - основной метод решения
**Файл:** `bhlff/core/fft/fft_solver_7d_basic.py`

**Действия:**
1. Добавить метод `solve()` как алиас для `solve_stationary()`
2. Добавить метод `get_spectral_coefficients()`
3. Добавить метод `solve_time_dependent()` для временных задач
4. Добавить метод `solve_nonlinear()` для нелинейных задач

**Код для добавления:**
```python
def solve(self, source: np.ndarray) -> np.ndarray:
    """
    Main solve method - alias for solve_stationary.
    
    Physical Meaning:
        Solves the fractional Laplacian equation L_β a = s
        using spectral methods in 7D space-time.
    """
    return self.solve_stationary(source)

def get_spectral_coefficients(self) -> np.ndarray:
    """
    Get spectral coefficients for the fractional Laplacian.
    
    Returns:
        np.ndarray: Spectral coefficients |k|^(2β) + λ
    """
    return self.spectral_coefficients

def solve_time_dependent(self, source: np.ndarray, dt: float) -> np.ndarray:
    """
    Solve time-dependent fractional Laplacian equation.
    
    Physical Meaning:
        Solves the time-dependent equation:
        ∂a/∂t + L_β a = s(x,t)
    """
    # Implementation for time-dependent solving
    pass

def solve_nonlinear(self, source: np.ndarray, nonlinear_params: Dict[str, Any]) -> np.ndarray:
    """
    Solve nonlinear fractional Laplacian equation.
    
    Physical Meaning:
        Solves nonlinear equation with amplitude-dependent coefficients.
    """
    # Implementation for nonlinear solving
    pass
```

#### 1.2 Исправить FractionalLaplacian инициализацию
**Проблема:** Неправильная передача параметров в конструктор
**Файл:** `bhlff/core/operators/fractional_laplacian.py`

**Действия:**
1. Исправить конструктор для правильной обработки параметров
2. Добавить поддержку Parameters7DBVP объекта
3. Добавить валидацию параметров

**Код для исправления:**
```python
def __init__(self, domain: "Domain", beta: Union[float, "Parameters"], lambda_param: float = 0.0):
    """
    Initialize fractional Laplacian operator.
    
    Args:
        domain (Domain): Computational domain.
        beta (Union[float, Parameters]): Fractional order or parameters object.
        lambda_param (float): Damping parameter.
    """
    # Handle both float and Parameters object
    if hasattr(beta, 'beta'):
        # Parameters object
        self.beta = beta.beta
        self.lambda_param = getattr(beta, 'lambda_param', lambda_param)
    else:
        # Direct float value
        self.beta = beta
        self.lambda_param = lambda_param
    
    # Validate parameters
    if not (0 < self.beta < 2):
        raise ValueError("Fractional order beta must be in (0,2)")
    
    # Rest of initialization...
```

#### 1.3 Добавить недостающие методы в BVP Core
**Проблема:** Отсутствует полная интеграция BVP Core
**Файл:** `bhlff/core/bvp/bvp_core/bvp_core_facade_impl.py`

**Действия:**
1. Реализовать `solve_envelope()` метод
2. Добавить интеграцию с QuenchDetector
3. Добавить поддержку U(1)³ Phase Vector

**Код для добавления:**
```python
def solve_envelope(self, source: np.ndarray) -> np.ndarray:
    """
    Solve BVP envelope equation.
    
    Physical Meaning:
        Solves the BVP envelope equation:
        ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t)
    """
    # Use BVP operations for envelope solving
    return self._operations.solve_envelope(source)

def detect_quenches(self, envelope: np.ndarray) -> Dict[str, Any]:
    """
    Detect quench events in BVP envelope.
    
    Physical Meaning:
        Detects threshold events (amplitude/detuning/gradient)
        in the BVP envelope field.
    """
    return self._operations.detect_quenches(envelope)
```

### ЭТАП 2: ИНТЕГРАЦИЯ BVP КОМПОНЕНТОВ (Приоритет: ВЫСОКИЙ)

#### 2.1 Реализовать QuenchDetector интеграцию
**Файл:** `bhlff/core/bvp/quench_detector.py`

**Действия:**
1. Реализовать три порога детекции:
   - Amplitude threshold
   - Detuning threshold  
   - Gradient threshold
2. Добавить статистику детекции
3. Интегрировать с BVP Core

**Критерии:**
- Quench detection accuracy ≥99%
- False positive rate ≤1%

#### 2.2 Реализовать U(1)³ Phase Vector
**Файл:** `bhlff/core/bvp/phase_vector/phase_vector.py`

**Действия:**
1. Реализовать три независимых U(1) компонента Θ_a (a=1..3)
2. Добавить electroweak current generation
3. Добавить SU(2) coupling strength
4. Интегрировать с BVP Core

**Критерии:**
- Phase vector normalization
- SU(2) coupling strength validation

#### 2.3 Реализовать BVP Impedance Calculator
**Файл:** `bhlff/core/bvp/bvp_impedance_calculator.py`

**Действия:**
1. Реализовать расчет Y(ω), R(ω), T(ω)
2. Добавить boundary conditions handling
3. Интегрировать с BVP Core

### ЭТАП 3: ТЕСТОВАЯ ИНТЕГРАЦИЯ (Приоритет: СРЕДНИЙ)

#### 3.1 Исправить тесты уровня A
**Проблема:** 28 тестов падают из-за отсутствующих методов

**Действия:**
1. Обновить все тесты для использования правильных методов
2. Исправить инициализацию FractionalLaplacian в тестах
3. Добавить поддержку Parameters7DBVP в тестах

**Файлы для исправления:**
- `tests/unit/test_level_a/test_A01_plane_wave_basic.py`
- `tests/unit/test_level_a/test_A01_plane_wave_advanced.py`
- `tests/unit/test_level_a/test_A02_multi_plane_basic.py`
- `tests/unit/test_level_a/test_A02_multi_plane_advanced.py`
- `tests/unit/test_level_a/test_A05_residual_energy_basic.py`
- `tests/unit/test_level_a/test_A05_residual_energy_advanced.py`

#### 3.2 Добавить BVP интеграционные тесты
**Действия:**
1. Создать тесты для BVP envelope solver
2. Создать тесты для quench detection
3. Создать тесты для U(1)³ phase vector
4. Создать тесты для impedance calculation

### ЭТАП 4: ВАЛИДАЦИЯ И ОПТИМИЗАЦИЯ (Приоритет: СРЕДНИЙ)

#### 4.1 Добавить валидацию физических принципов
**Действия:**
1. BVP energy balance ≤1–3%
2. Relative BVP envelope error ≤10⁻¹²
3. Grid convergence validation (×2: N=128→256→512)
4. Anisotropy absence validation

#### 4.2 Оптимизировать производительность
**Действия:**
1. Оптимизировать FFT operations
2. Добавить CUDA поддержку где возможно
3. Оптимизировать memory management
4. Добавить parallel processing

### ЭТАП 5: ДОКУМЕНТАЦИЯ И ФИНАЛИЗАЦИЯ (Приоритет: НИЗКИЙ)

#### 5.1 Обновить документацию
**Действия:**
1. Обновить API документацию
2. Добавить примеры использования
3. Обновить README с новыми возможностями

#### 5.2 Финальное тестирование
**Действия:**
1. Запустить все тесты уровня A
2. Проверить покрытие кода ≥90%
3. Валидировать физические принципы
4. Проверить производительность

---

## 📋 ДЕТАЛЬНЫЙ ЧЕКЛИСТ

### Критические исправления (ДОЛЖНО БЫТЬ СДЕЛАНО):
- [ ] Добавить метод `solve()` в `FFTSolver7DBasic`
- [ ] Добавить метод `get_spectral_coefficients()` в `FFTSolver7DBasic`
- [ ] Исправить инициализацию `FractionalLaplacian`
- [ ] Реализовать `solve_envelope()` в BVP Core
- [ ] Исправить 28 падающих тестов

### BVP интеграция (ВАЖНО):
- [ ] Реализовать QuenchDetector с тремя порогами
- [ ] Реализовать U(1)³ Phase Vector
- [ ] Реализовать BVP Impedance Calculator
- [ ] Интегрировать все компоненты с BVP Core

### Тестирование (ВАЖНО):
- [ ] Исправить все падающие тесты уровня A
- [ ] Добавить BVP интеграционные тесты
- [ ] Валидировать физические критерии
- [ ] Проверить покрытие кода ≥90%

### Оптимизация (ЖЕЛАТЕЛЬНО):
- [ ] Оптимизировать FFT operations
- [ ] Добавить CUDA поддержку
- [ ] Оптимизировать memory management
- [ ] Добавить parallel processing

---

## ⏱️ ВРЕМЕННЫЕ ОЦЕНКИ

### Этап 1 (Критические исправления): 2-3 дня
- Исправление FFTSolver7DBasic: 1 день
- Исправление FractionalLaplacian: 0.5 дня
- Добавление методов в BVP Core: 1 день
- Тестирование исправлений: 0.5 дня

### Этап 2 (BVP интеграция): 3-4 дня
- QuenchDetector: 1 день
- U(1)³ Phase Vector: 1 день
- BVP Impedance Calculator: 1 день
- Интеграция компонентов: 1 день

### Этап 3 (Тестовая интеграция): 2-3 дня
- Исправление тестов: 1 день
- Добавление BVP тестов: 1 день
- Валидация: 1 день

### Этап 4 (Валидация): 1-2 дня
### Этап 5 (Документация): 1 день

**ОБЩЕЕ ВРЕМЯ: 9-13 дней**

---

## 🎯 КРИТЕРИИ ГОТОВНОСТИ

### Минимальная готовность (70%):
- Все критические исправления выполнены
- 28 падающих тестов исправлены
- Базовые BVP компоненты работают

### Полная готовность (100%):
- Все BVP компоненты интегрированы
- Все тесты уровня A проходят
- Физические критерии валидированы
- Покрытие кода ≥90%

---

## 🚀 ПЛАН ДЕЙСТВИЙ

1. **НЕМЕДЛЕННО** (сегодня):
   - Исправить FFTSolver7DBasic
   - Исправить FractionalLaplacian
   - Запустить тесты для проверки

2. **НА ЭТОЙ НЕДЕЛЕ**:
   - Реализовать BVP интеграцию
   - Исправить все падающие тесты
   - Добавить BVP тесты

3. **НА СЛЕДУЮЩЕЙ НЕДЕЛЕ**:
   - Валидация и оптимизация
   - Финальное тестирование
   - Документация

**ЦЕЛЬ: Полная готовность к проверке гипотезы А через 2 недели**
