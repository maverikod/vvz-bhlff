# Анализ отсутствующих методов

## FFTBackend (bhlff/core/fft/fft_backend_core.py)

### Отсутствующие атрибуты:
- ❌ `N` - должен быть `domain.N`
- ❌ `N_phi` - должен быть `domain.N_phi` 
- ❌ `N_t` - должен быть `domain.N_t`

### Отсутствующие методы:
- ❌ `forward_transform()` - должен быть алиас для `fft()`
- ❌ `inverse_transform()` - должен быть алиас для `ifft()`
- ❌ `get_wave_vectors(dim)` - должен возвращать отдельные волновые векторы

## Domain (bhlff/core/domain/domain.py)

### Отсутствующие методы:
- ❌ `get_coordinates(dim)` - должен возвращать координаты для измерения

## SpectralOperations (bhlff/core/fft/spectral_operations.py)

### Отсутствующие методы:
- ❌ `compute_derivative(field, order, axis)`
- ❌ `compute_gradient(field)`
- ❌ `compute_divergence(vector_field)`
- ❌ `compute_curl(vector_field)`

## SpectralDerivatives (bhlff/core/fft/spectral_derivatives.py)

### Отсутствующие методы:
- ❌ `compute_first_derivative(field, axis)`
- ❌ `compute_second_derivative(field, axis)`
- ❌ `compute_nth_derivative(field, order, axis)`
- ❌ `compute_mixed_derivative(field, orders)`

## SpectralFiltering (bhlff/core/fft/spectral_filtering.py)

### Отсутствующие методы:
- ❌ `apply_low_pass_filter(field, cutoff)`
- ❌ `apply_high_pass_filter(field, cutoff)`
- ❌ `apply_band_pass_filter(field, low_cutoff, high_cutoff)`
- ❌ `apply_gaussian_filter(field, sigma)`

## FFTPlanManager (bhlff/core/fft/fft_plan_manager.py)

### Отсутствующие методы:
- ❌ `create_plan(field)`
- ❌ `get_plan(field)`
- ❌ `clear_plans()`

## FFTButterflyComputer (bhlff/core/fft/fft_butterfly_computer.py)

### Отсутствующие методы:
- ❌ `compute_butterfly(data)`
- ❌ `compute_inverse_butterfly(data)`

## FFTTwiddleComputer (bhlff/core/fft/fft_twiddle_computer.py)

### Отсутствующие методы:
- ❌ `get_twiddle_factor(dim1, dim2)`
- ❌ `compute_inverse_twiddle_factors()`

### Исправления:
- ✅ `compute_twiddle_factors(dimensions)` - уже есть, но требует параметр

## Приоритет исправлений

### Высокий приоритет (критично для тестов):
1. **FFTBackend** - добавить атрибуты и алиасы методов
2. **Domain** - добавить `get_coordinates()`
3. **SpectralOperations** - добавить основные методы

### Средний приоритет:
4. **SpectralDerivatives** - добавить методы производных
5. **SpectralFiltering** - добавить методы фильтрации

### Низкий приоритет:
6. **FFTPlanManager** - добавить методы управления планами
7. **FFTButterflyComputer** - добавить методы бабочки
8. **FFTTwiddleComputer** - добавить недостающие методы
