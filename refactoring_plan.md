# План рефакторинга проекта BHLFF

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

## Обзор проблем

### 1. Файлы превышающие лимит 400 строк

#### Критические файлы (>400 строк):
1. **bhlff/core/bvp/resonance_quality_analysis.py** (432 строки)
2. **bhlff/models/level_c/beating/ml/beating_ml_optimization.py** (429 строк)
3. **bhlff/core/bvp/envelope_equation/solver_core_basic.py** (429 строк)
4. **bhlff/models/level_c/beating/beating_core_basic.py** (427 строк)
5. **bhlff/models/level_c/beating/ml/beating_ml_prediction.py** (416 строк)
6. **bhlff/core/fft/fft_plan_7d.py** (400 строк)

#### Файлы близкие к лимиту (350-400 строк):
- **bhlff/core/fft/fractional_laplacian.py** (396 строк)
- **bhlff/core/fft/unified_spectral_operations.py** (395 строк)
- **bhlff/models/level_b/power_law/critical_exponents.py** (387 строк)
- **bhlff/core/time/adaptive/adaptive_integrator.py** (387 строк)
- **bhlff/core/bvp/envelope_equation/advanced/solver_advanced_core.py** (383 строки)

### 2. Недописанный код

#### Файлы с `pass` в неабстрактных методах:
1. **bhlff/core/fft/advanced/fft_advanced_core.py** - 2 метода с `pass`
2. **tests/unit/test_core/test_time_integrators_physics.py** - множественные `pass`
3. **tests/unit/test_core/test_nonlinear_coefficients_physics.py** - множественные `pass`
4. **tests/unit/test_core/test_frequency_dependent_properties_physics.py** - множественные `pass`
5. **tests/unit/test_core/test_fft_solver_7d_validation.py** - множественные `pass`

#### Файлы с `NotImplemented` в неабстрактных методах:
1. **tests/unit/test_core/time_integrators/test_advanced_integrators.py**
2. **tests/unit/test_core/time_integrators/test_basic_integrators.py**
3. **bhlff/models/levels/bvp_integration_base.py**
4. **bhlff/solvers/integrators/time_integrator.py**
5. **bhlff/core/bvp/residual_computer_base.py**

### 3. Отклонения от теории и ТЗ

#### Отсутствие BVP-модуляционного подхода:
- Многие файлы не используют BVP (Base High-Frequency Field) как центральную основу
- Отсутствует реализация квенч-детекции
- Нет интеграции с U(1)³ фазовой структурой

#### Отклонения от 7D теории:
- Некоторые файлы работают только с 3D пространством
- Отсутствует поддержка фазовых координат (φ₁, φ₂, φ₃)
- Нет реализации 7D спектральных операций

### 4. Проблемы с импортами

#### Импорты не в начале файла:
- **bhlff/core/fft/advanced/fft_advanced_core.py** - импорты в середине файла (строки 15-31)

## ПЛАН РЕФАКТОРИНГА (НОВЫЕ ПРИОРИТЕТЫ)

### **ПРИОРИТЕТ 1: Недописанный код**

#### **1.1. Файлы с pass/NotImplemented (20 файлов):**
- **`bvp_integration_base.py`** - `integrate_level()` - **ОСТАВИТЬ** (абстрактный)
- **`time_integrator.py`** - `step()`, `get_integrator_type()` - **ОСТАВИТЬ** (абстрактный)
- **`residual_computer_base.py`** - `_setup_parameters()`, `compute_residual()` - **ОСТАВИТЬ** (абстрактный)
- **`abstract_bvp_facade.py`** - `solve_envelope()`, `detect_quenches()` - **ОСТАВИТЬ** (абстрактный)
- **`spectral_derivatives_base.py`** - `compute_gradient()`, `compute_divergence()` - **ОСТАВИТЬ** (абстрактный)
- **`abstract_solver.py`** - `solve()`, `solve_time_evolution()` - **ОСТАВИТЬ** (абстрактный)
- **`base_integrator.py`** - `integrate()`, `step()` - **ОСТАВИТЬ** (абстрактный)
- **`bvp_postulate_base.py`** - `apply()` - **ОСТАВИТЬ** (абстрактный)
- **`bvp_level_interface_base.py`** - `process_bvp_data()` - **ОСТАВИТЬ** (абстрактный)
- **`source.py`** - `generate()`, `get_source_type()` - **ОСТАВИТЬ** (абстрактный)

**Статус**: Все файлы содержат абстрактные методы - **НЕ ТРОГАТЬ**

### **ПРИОРИТЕТ 2: Замена классических паттернов**

#### **2.1. Критические файлы для замены (3 файла):**
- **`exponential_integrator.py`** → **ЗАМЕНИТЬ НА BVP envelope интегратор**
  - **Проблема**: Классическое экспоненциальное решение противоречит BVP-теории
  - **Решение**: BVP envelope уравнение с модуляциями огибающей

- **`fft_solver_time.py`** → **УДАЛИТЬ экспоненциальные интеграторы**
  - **Проблема**: Использует `BVPExponentialIntegrator` (классический паттерн)
  - **Решение**: Заменить на BVP envelope методы

- **`core/time/__init__.py`** → **УДАЛИТЬ импорт экспоненциального интегратора**
  - **Проблема**: Экспортирует `BVPExponentialIntegrator`
  - **Решение**: Заменить на BVP envelope интегратор

### **ПРИОРИТЕТ 3: Удаление лишних файлов**

#### **3.1. Файлы с избыточной функциональностью:**
- **`beating_validation_metrics.py`** - дублирует функциональность
- **`beating_validation_basic_main.py`** - избыточный координатор
- **`beating_validation_consistency.py`** - дублирует валидацию
- **`beating_validation_statistics.py`** - дублирует статистику
- **`validation.py`** - общий валидатор без специфики
- **`energy_analysis.py`** - дублирует энергетический анализ
- **`convergence_analysis.py`** - дублирует анализ сходимости

**Действие**: Удалить дублирующие файлы, оставить только уникальную функциональность

### **ПРИОРИТЕТ 4: Дробление файлов**

#### **4.1. Файлы для разделения (превышают 400 строк):**
- **`beating_ml_optimization.py` (429 строк)** → разделить на 3 модуля:
  - `beating_ml_optimization_core.py` (~150 строк) - основная логика
  - `beating_ml_optimization_algorithms.py` (~150 строк) - алгоритмы оптимизации  
  - `beating_ml_optimization_validation.py` (~130 строк) - валидация результатов

- **`beating_core_basic.py` (427 строк)** → разделить на 3 модуля:
  - `beating_core_basic_analysis.py` (~150 строк) - анализ биений
  - `beating_core_basic_computation.py` (~150 строк) - вычисления
  - `beating_core_basic_validation.py` (~130 строк) - валидация

- **`solver_core_basic.py` (429 строк)** → разделить на 3 модуля:
  - `solver_core_basic_newton.py` (~150 строк) - Newton-Raphson
  - `solver_core_basic_linear.py` (~150 строк) - линейные системы
  - `solver_core_basic_convergence.py` (~130 строк) - сходимость

- **`beating_ml_prediction.py` (416 строк)** → разделить на 3 модуля:
  - `beating_ml_prediction_core.py` (~150 строк) - основная логика
  - `beating_ml_prediction_models.py` (~150 строк) - ML модели
  - `beating_ml_prediction_validation.py` (~120 строк) - валидация

### **ИТОГОВЫЕ ДЕЙСТВИЯ:**

1. **НЕ ТРОГАТЬ** абстрактные классы с NotImplemented (20 файлов)
2. **Заменить 3 файла** с классическими паттернами на BVP envelope методы
3. **Удалить 7 файлов** с избыточной функциональностью
4. **Разделить 4 больших файла** на 12 модулей

## КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ

### **КРИТИЧЕСКОЕ УТОЧНЕНИЕ:**
- **Экспоненциальные решения** = ОСТАТКИ КЛАССИЧЕСКИХ ПАТТЕРНОВ (НЕ BVP!)
- **BVP-теория** = envelope modulations и beatings (НЕ экспоненциальные решения!)
- **FFT экспоненты** = математика FFT (twiddle factors) - ПРАВИЛЬНО
- **BVP envelope** = модуляции огибающей - ПРАВИЛЬНО

### **ИТОГОВЫЕ РЕЗУЛЬТАТЫ:**

**До рефакторинга**: 311 файлов
**После рефакторинга**: ~323 файла (увеличение на 4%)

**Конкретные изменения**:
- **+12 файлов** - разделение 4 больших файлов на модули
- **+1 файл** - новый BVP envelope интегратор
- **+0 файлов** - BVP интеграция в существующие файлы
- **-0 файлов** - НЕ удаляем существующий код

**Критические исправления**:
- **4 больших файла** → 12 модулей (соблюдение лимита 400 строк)
- **1 экспоненциальный интегратор** → BVP envelope интегратор (УДАЛИТЬ классические паттерны)
- **35 файлов** → добавление BVP функциональности
- **0 файлов** → НЕ трогаем абстрактные классы
- **0 файлов** → НЕ трогаем правильную архитектуру
