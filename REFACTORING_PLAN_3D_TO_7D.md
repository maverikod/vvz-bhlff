# План рефакторинга: Переход с 3D на полный 7D домен

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com  
**Date:** 2024

## Executive Summary

Обнаружено **15+ мест** в коде, где используется 3D домен (N_phi=1, N_t=1) вместо полного 7D домена с квадратной сеткой. Это критическая архитектурная проблема, которая может вызывать искажения в спектральных операциях и неправильные результаты тестов.

**Цель рефакторинга**: Заменить все использования 3D доменов на полные 7D домены с квадратной сеткой (N_phi=N, N_t=N).

---

## Критичность проблем

| Категория | Количество | Критичность | Приоритет |
|-----------|------------|-------------|-----------|
| **КРИТИЧНО**: N_phi=1, N_t=1 | 8 мест | КРИТИЧЕСКАЯ | ВЫСОКИЙ |
| **ВЫСОКО**: N_phi=1 или N_t=1 (частично) | 1 место | ВЫСОКАЯ | ВЫСОКИЙ |
| **СРЕДНЕ**: N_phi=2, N_t=2 (минимальные) | 6+ мест | СРЕДНЯЯ | СРЕДНИЙ |

---

## Детальный план рефакторинга

### Этап 1: Критические тесты Level A (N_phi=1, N_t=1) ⚠️ КРИТИЧНО

#### 1.1. `tests/unit/test_level_a/test_A11_scale_length.py`

**Текущее состояние**:
```python
def _create_3d_domain(L: float, N: int) -> Domain:
    return Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
```

**Проблема**: Это **главный тест** для проверки инвариантности масштаба длины. Использование 3D домена делает тест неполным.

**Действия**:
1. Переименовать `_create_3d_domain` → `_create_7d_domain`
2. Изменить на полный 7D домен:
   ```python
   def _create_7d_domain(L: float, N: int) -> Domain:
       """
       Create proper 7D domain with square grid for scale invariance testing.
       
       Physical Meaning:
           Creates full 7D domain M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ with square grid
           to properly test scale invariance in all 7 dimensions.
       """
       return Domain(
           L=L,
           N=N,
           N_phi=N,       # ✓ Квадратная сетка
           N_t=N,         # ✓ Квадратная сетка
           T=2*np.pi,     # ✓ Период для фазовых координат
           dimensions=7,
       )
   ```
3. Обновить все вызовы функции
4. Обновить комментарии и докстринги
5. Проверить масштабирование источника (Root Cause #1)

**Приоритет**: **КРИТИЧЕСКИЙ** (это главный тест!)

---

#### 1.2. `tests/unit/test_level_a/test_A12_units_invariance.py`

**Текущее состояние**:
```python
def _create_3d_domain(L: float, N: int) -> Domain:
    return Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
```

**Проблема**: Тест проверяет инвариантность к выбору единиц. Использование 3D домена делает проверку неполной.

**Действия**: Аналогично 1.1

**Приоритет**: **КРИТИЧЕСКИЙ**

---

#### 1.3. `tests/unit/test_level_a/test_A01_plane_wave.py`

**Текущее состояние**:
```python
def _create_3d_domain(L: float, N: int) -> Domain:
    return Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
```

**Проблема**: Тест проверяет базовую функциональность plane wave. Использование 3D домена может скрывать проблемы.

**Действия**: Аналогично 1.1

**Приоритет**: **ВЫСОКИЙ**

---

#### 1.4. `tests/unit/test_level_a/test_A02_multi_plane.py`

**Текущее состояние**:
```python
def _create_3d_domain(L: float, N: int) -> Domain:
    return Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
```

**Проблема**: Тест проверяет множественные plane waves. Использование 3D домена ограничивает проверку.

**Действия**: Аналогично 1.1

**Приоритет**: **ВЫСОКИЙ**

---

#### 1.5. `tests/unit/test_level_a/test_A03_zero_mode.py`

**Текущее состояние**:
```python
domain_7d = Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
```

**Проблема**: Прямое создание домена с N_phi=1, N_t=1.

**Действия**:
1. Создать функцию `_create_7d_domain(L, N)` (аналогично 1.1)
2. Заменить прямое создание домена на вызов функции
3. Использовать полный 7D домен с квадратной сеткой

**Приоритет**: **ВЫСОКИЙ**

---

#### 1.6. `tests/unit/test_level_a/test_A04_time_harmonic.py`

**Текущее состояние**:
```python
domain_7d = Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
```

**Проблема**: Тест проверяет time-harmonic решения. Использование N_t=1 делает проверку неполной.

**Действия**: Аналогично 1.5

**Приоритет**: **ВЫСОКИЙ**

---

#### 1.7. `tests/unit/test_level_a/test_A05_residual_energy.py`

**Текущее состояние**:
```python
domain_7d = Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
```

**Проблема**: Тест проверяет энергетический баланс. Использование 3D домена может влиять на результаты.

**Действия**: Аналогично 1.5

**Приоритет**: **ВЫСОКИЙ**

---

#### 1.8. `debug_a11_downsampling.py`

**Текущее состояние**:
```python
def _create_3d_domain(L: float, N: int) -> Domain:
    return Domain(L=L, N=N, N_phi=1, N_t=1, T=1.0, dimensions=7)
```

**Проблема**: Диагностический скрипт использует 3D домен, что может влиять на результаты анализа.

**Действия**: Аналогично 1.1

**Приоритет**: **СРЕДНИЙ** (диагностический скрипт)

---

### Этап 2: Частичные проблемы (N_phi или N_t = 1)

#### 2.1. `tests/unit/test_level_e/test_field_generator_gradients.py`

**Текущее состояние**:
```python
domain = Domain(L=2.0, N=N, dimensions=7, N_phi=8, N_t=1, T=1.0)
```

**Проблема**: N_phi=8 (хорошо), но N_t=1 (проблема). Для теста градиентов временное измерение важно.

**Действия**:
1. Изменить на N_t=N (квадратная сетка)
2. Проверить, что тест все еще работает корректно

**Приоритет**: **ВЫСОКИЙ**

---

### Этап 3: Минимальные значения (N_phi=2, N_t=2) - Опционально

#### 3.1. `tests/unit/test_level_a/test_A01_plane_wave_basic.py`

**Текущее состояние**:
```python
return Domain(L=L, N=N, N_phi=2, N_t=2, T=1.0, dimensions=7)
```

**Проблема**: Минимальные значения, но не квадратная сетка (N_phi=2, N_t=2 при N=64).

**Действия**: Рассмотреть переход на N_phi=N, N_t=N для полной проверки

**Приоритет**: **СРЕДНИЙ** (можно оставить для быстрых тестов)

---

#### 3.2. `tests/unit/test_level_a/test_batched_operations.py`

**Текущее состояние**:
```python
return Domain(L=L, N=N, N_phi=2, N_t=2, T=1.0, dimensions=7)
```

**Действия**: Аналогично 3.1

**Приоритет**: **СРЕДНИЙ**

---

#### 3.3. Другие места с минимальными значениями

- `tests/unit/test_level_a/test_final_summary.py`: N_phi=2, N_t=4
- `tests/unit/test_level_a/test_A01_minimal.py`: N_phi=2, N_t=4
- `tools/hypothesis_testing/commands/base.py`: N_phi=2, N_t=4

**Действия**: Рассмотреть для каждого случая отдельно

**Приоритет**: **НИЗКИЙ** (можно оставить для быстрых тестов)

---

## Общая стратегия рефакторинга

### Шаг 1: Создать утилиту для создания 7D домена

**Файл**: `tests/unit/test_level_a/test_utils_7d_domain.py` (новый)

```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Utility functions for creating proper 7D domains in tests.

Physical Meaning:
    Provides standardized functions for creating full 7D domains
    with square grids for proper testing of 7D phase field theory.
"""

import numpy as np
from bhlff.core.domain import Domain


def create_7d_domain_square(L: float, N: int, T: float = None) -> Domain:
    """
    Create proper 7D domain with square grid.
    
    Physical Meaning:
        Creates full 7D domain M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ with square grid
        (N_phi=N, N_t=N) for proper testing of 7D phase field theory.
        
    Mathematical Foundation:
        For proper 7D scale invariance testing, all dimensions must:
        1. Have the same resolution (square grid)
        2. Scale proportionally with L
        3. Participate in spectral operations correctly
        
    Args:
        L (float): Spatial domain size.
        N (int): Spatial resolution (will be used for all dimensions).
        T (float, optional): Temporal domain size. Defaults to 2π.
        
    Returns:
        Domain: Full 7D domain with square grid.
    """
    if T is None:
        T = 2 * np.pi  # Default period for phase coordinates
    
    return Domain(
        L=L,
        N=N,
        N_phi=N,       # ✓ Квадратная сетка
        N_t=N,         # ✓ Квадратная сетка
        T=T,
        dimensions=7,
    )


def create_7d_domain_minimal(L: float, N: int, T: float = 1.0) -> Domain:
    """
    Create minimal 7D domain for fast tests (N_phi=2, N_t=2).
    
    Physical Meaning:
        Creates minimal 7D domain for fast unit tests where
        full resolution is not critical.
        
    Args:
        L (float): Spatial domain size.
        N (int): Spatial resolution.
        T (float): Temporal domain size.
        
    Returns:
        Domain: Minimal 7D domain.
    """
    return Domain(
        L=L,
        N=N,
        N_phi=2,       # Минимальное для быстрых тестов
        N_t=2,         # Минимальное для быстрых тестов
        T=T,
        dimensions=7,
    )
```

---

### Шаг 2: Рефакторинг критических тестов

**Порядок выполнения**:
1. ✅ `test_A11_scale_length.py` - **КРИТИЧНО** (главный тест)
2. ✅ `test_A12_units_invariance.py` - **КРИТИЧНО**
3. ✅ `test_A01_plane_wave.py` - **ВЫСОКО**
4. ✅ `test_A02_multi_plane.py` - **ВЫСОКО**
5. ✅ `test_A03_zero_mode.py` - **ВЫСОКО**
6. ✅ `test_A04_time_harmonic.py` - **ВЫСОКО**
7. ✅ `test_A05_residual_energy.py` - **ВЫСОКО**
8. ✅ `debug_a11_downsampling.py` - **СРЕДНЕ**

---

### Шаг 3: Проверка после рефакторинга

После каждого изменения:
1. ✅ Запустить тесты: `pytest tests/unit/test_level_a/test_A11_scale_length.py -v`
2. ✅ Проверить, что ошибка уменьшилась (ожидается с 2.91e-02 до ~1e-3)
3. ✅ Проверить, что другие тесты не сломались
4. ✅ Запустить code_mapper для обновления индексов

---

## Ожидаемые результаты

### После рефакторинга критических тестов:

| Тест | Текущая ошибка | Ожидаемая ошибка | Улучшение |
|------|----------------|------------------|-----------|
| A1.1 Scale Length | 2.91e-02 | ≤ 1e-12 | 10^10× |
| A1.2 Units Invariance | ? | ≤ 1e-12 | ? |
| A0.1-A0.5 | ? | Должны пройти | ? |

### Общие улучшения:

1. ✅ Устранение искажений в FFT операциях
2. ✅ Правильная нормализация FFT для всех измерений
3. ✅ Полная проверка инвариантности во всех 7 измерениях
4. ✅ Правильные спектральные коэффициенты для 7D

---

## Риски и митигация

### Риск 1: Увеличение времени выполнения тестов

**Проблема**: Полный 7D домен (N×N×N×N×N×N×N) требует больше памяти и времени.

**Митигация**:
- Использовать меньшие значения N для тестов (например, N=32 вместо N=64)
- Использовать CUDA для ускорения
- Оптимизировать использование памяти

### Риск 2: Несовместимость с существующими тестами

**Проблема**: Изменение домена может сломать существующие тесты.

**Митигация**:
- Рефакторинг по одному тесту за раз
- Тщательное тестирование после каждого изменения
- Сохранение старых версий для сравнения

### Риск 3: Проблемы с памятью GPU

**Проблема**: Полный 7D домен может не поместиться в память GPU.

**Митигация**:
- Использовать блочную обработку
- Уменьшить N для тестов
- Использовать CPU fallback при необходимости

---

## Чеклист рефакторинга

### Для каждого файла:

- [ ] Создать/обновить функцию `_create_7d_domain` с квадратной сеткой
- [ ] Заменить все использования `_create_3d_domain` на `_create_7d_domain`
- [ ] Заменить все прямые создания домена с N_phi=1, N_t=1
- [ ] Обновить комментарии и докстринги
- [ ] Запустить тесты и проверить результаты
- [ ] Обновить code_mapper индексы
- [ ] Проверить размер файла (не более 400 строк)

---

## Приоритизация выполнения

1. **КРИТИЧНО**: `test_A11_scale_length.py` - главный тест инвариантности
2. **КРИТИЧНО**: `test_A12_units_invariance.py` - тест инвариантности единиц
3. **ВЫСОКО**: Остальные тесты Level A (A01-A05)
4. **СРЕДНЕ**: Диагностические скрипты
5. **НИЗКО**: Минимальные значения (N_phi=2, N_t=2) - можно оставить

---

## Заключение

Рефакторинг с 3D на полный 7D домен - это **критическая задача**, которая должна решить проблему ошибки 2.91e-02 в тесте A1.1. План рефакторинга разбит на этапы с четкими приоритетами и чеклистами для каждого файла.

**Следующий шаг**: Начать с рефакторинга `test_A11_scale_length.py` как наиболее критичного теста.
