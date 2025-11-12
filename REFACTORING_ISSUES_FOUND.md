# Найденные проблемы и недоделанные пункты рефакторинга

## Критические проблемы

### 1. `bhlff/core/bvp/bvp_block_processing_system.py`

#### Проблема: Использование `np.broadcast_to` вместо явного 7D построения
**Местоположение:** Строки 333, 335, 337, 389, 391, 393
**Текущий код:**
```python
if kappa.shape != a.shape:
    kappa = np.broadcast_to(kappa, a.shape)
if chi.shape != a.shape:
    chi = np.broadcast_to(chi, a.shape)
if bc_term.shape != a.shape:
    bc_term = np.broadcast_to(bc_term, a.shape)
```

**Требуется:** Заменить на явное 7D построение блоков с конкретными размерами по фазе/времени

#### Проблема: Заглушка в `_solve_block_iterative`
**Местоположение:** Строка 531
**Текущий код:**
```python
def _solve_block_iterative(
    self, lhs: np.ndarray, rhs: np.ndarray, initial: np.ndarray
) -> np.ndarray:
    """Solve block system iteratively."""
    # Simplified iterative solver - in practice would implement proper iterative method
    return rhs
```

**Требуется:** Реализовать полноценный итеративный решатель (Jacobi/Gauss-Seidel)

#### Проблема: Граничные условия - только penalty term
**Местоположение:** Метод `_apply_boundary_conditions` (строки 487-524)
**Текущий код:** Использует только penalty term, не реализует физически осмысленные Dirichlet/Neumann BC

**Требуется:** Реализовать физически осмысленные граничные условия на гранях блоков

### 2. `bhlff/models/level_c/cuda/cuda_admittance_processor.py`

#### Проблема: Нужно проверить использование `.flatten()` в редукциях
**Требуется:** Проверить методы `_compute_blocked_cuda` и внутренние редукции на использование `.flatten()` - должны использоваться axis-wise редукции

### 3. `bhlff/core/sources/bvp_source_core.py` и `bvp_source_envelope.py`

#### Проблема: Проверка на `np.broadcast_to`
**Статус:** ✅ Проверено - не найдено использование `np.broadcast_to`
**Примечание:** Используется `expand_block_to_7d_explicit` - корректно

### 4. `bhlff/models/level_b/power_law/critical_exponents.py`

#### Статус: ✅ Исправлено
**Примечание:** Методы `estimate_nu_from_correlation_length`, `estimate_beta_from_tail`, `estimate_chi_from_variance` используют robust Theil-Sen regression без фиксированных fallback значений

### 5. `bhlff/models/level_c/abcd_model/abcd_model.py`

#### Статус: ✅ Исправлено
**Примечание:** `np.eye(2)` используется только как multiplicative identity, не как generic criteria. `compute_resonator_determinants` использует физически осмысленные spectral metrics (poles/Q factors)

### 6. `bhlff/models/level_c/boundary/radial_analysis_core.py`

#### Статус: ✅ Исправлено
**Примечание:** Использует точные dtypes (float64/complex128) перед GPU transfer, нет CPU path

### 7. `bhlff/core/bvp/bvp_core/bvp_cuda_block_processor.py`

#### Статус: ✅ Исправлено
**Примечание:** Level C code paths помечены как GPU-only, CPU fallback только для non-C integration tests

## Требует проверки

### 8. `bhlff/core/sources/blocked_field_generator.py`
**Требуется:** Проверить методы `iterate_blocks` и `get_block_by_indices` - должны позволять большие количества блоков с предупреждением (не hard error)

### 9. `bhlff/models/level_c/memory/memory_evolution.py`
**Требуется:** Проверить метод `_create_initial_field` - должен извлекать 3D из 7D BlockedField усреднением по фазе/времени на блок

### 10. `bhlff/core/bvp/phase_vector/electroweak_coupling.py`
**Требуется:** Проверить strict GPU path с динамическим 7D block tiling на основе 80% free VRAM, удалить CPU fallbacks

### 11. `bhlff/core/domain/enhanced_block_processor.py`
**Требуется:** Для Level C контекстов отключить CPU fallback paths или защитить за explicit non-C flags

### 12. `bhlff/utils/cuda_utils.py`
**Требуется:** Удалить CPU fallback paths для Level C code paths, поднимать ошибку с инструкциями при недостатке GPU памяти

## Резюме

**Критические проблемы (требуют немедленного исправления):**
1. `bvp_block_processing_system.py`: `np.broadcast_to` → явное 7D построение
2. `bvp_block_processing_system.py`: Заглушка `_solve_block_iterative` → полноценный решатель
3. `bvp_block_processing_system.py`: Граничные условия → физически осмысленные BC

**Требует проверки:**
- Проверка `.flatten()` в `cuda_admittance_processor.py`
- Проверка остальных файлов из списка

