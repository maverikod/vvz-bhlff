# Step 05: Детализированная спецификация тестов фундаментальных свойств поля уровня B

## Физическая и математическая основа

### Теоретический контекст
Согласно 7D теории фазового поля, элементарные частицы представляют собой **устойчивые фазовые конфигурации** с трёхуровневой структурой:

1. **Ядро** — область высокой когерентности, топологический дефект поля
2. **Переходная зона** — нелинейная область согласования свойств ядра и волнового хвоста  
3. **Хвост** — область чисто волновой природы, описываемая волновой функцией

В **однородной "безынтервальной" среде** (космос) фазовое поле демонстрирует фундаментальные свойства, которые должны быть численно подтверждены.

### Математическая постановка

#### Оператор Рисса и спектральная форма
- **Область**: Ω = [0,L)³ с периодическими граничными условиями
- **Поле**: a(x) ∈ ℂ - комплексное фазовое поле
- **Оператор Рисса**: L_β a = μ(-Δ)^β a + λa
- **Стационарная задача**: L_β a = s(x)

#### Спектральное представление
- **Дискретные волновые числа**: k = (2π/L)m, m ∈ ℤ³
- **В k-пространстве**: â(k) = ŝ(k)/(μ|k|^(2β) + λ)
- **Символ оператора**: D(k) = μ|k|^(2β) + λ

#### Ключевые теоретические предсказания
1. **Степенной хвост**: A(r) ∝ r^(2β-3) при λ=0
2. **Отсутствие узлов**: В чистом фракционном режиме сферических стоячих узлов нет
3. **Топологическая стабильность**: Целочисленные топологические заряды
4. **Зонная структура**: Четкое разделение на ядро/переход/хвост

## Детализированные тесты уровня B

### B1. Степенной хвост в однородной среде («космос»)

#### Физический смысл
Подтверждение того, что в однородной безынтервальной среде фазовое поле создает **затухающие радиальные зоны сжатия-разрежения** вокруг топологического дефекта, описываемые степенным законом.

#### Математическая основа
- **Теоретический закон**: A(r) ∝ r^(2β-3) при λ=0
- **Физическая интерпретация**: Символ D(k) = μk^(2β) не имеет полюсов → нет стационарных осцилляций, только алгебраическое затухание
- **Связь с топологией**: Степенной хвост обеспечивает "мягкую" связь между ядром и дальним полем

#### Постановка задачи
```python
# Параметры теста
domain_params = {
    "L": 10.0,           # Размер области
    "N": 512,            # Количество точек по каждой оси
    "dimensions": 3      # Трёхмерная область
}

physics_params = {
    "mu": 1.0,           # Коэффициент диффузии
    "beta": 1.0,         # Фракционный порядок β ∈ (0,2)
    "lambda": 0.0        # КРИТИЧНО: λ=0 для чистого фракционного режима
}

source_params = {
    "type": "point_source",
    "center": [5.0, 5.0, 5.0],  # Центр точечного дефекта
    "amplitude": 1.0
}
```

#### Алгоритм анализа
```python
def analyze_power_law_tail(field, beta, center, min_decades=1.5):
    """
    Анализ степенного хвоста A(r) ∝ r^(2β-3)
    
    Physical Meaning:
        Проверяет соответствие радиального профиля поля теоретическому
        степенному закону, который является фундаментальным свойством
        фракционного оператора Рисса в однородной среде.
        
    Mathematical Foundation:
        В спектральном представлении символ D(k) = μk^(2β) не имеет
        полюсов, что исключает осцилляторное поведение и приводит
        к чисто алгебраическому затуханию по степенному закону.
    """
    # 1. Вычисление радиального профиля
    radial_profile = compute_radial_profile(field, center)
    
    # 2. Фильтрация области хвоста (исключение ядра)
    r_core = estimate_core_radius(field, center)
    tail_mask = radial_profile['r'] > 2 * r_core
    r_tail = radial_profile['r'][tail_mask]
    A_tail = radial_profile['A'][tail_mask]
    
    # 3. Проверка достаточности диапазона
    log_range = np.log10(r_tail.max() / r_tail.min())
    if log_range < min_decades:
        raise ValueError(f"Insufficient range: {log_range:.2f} < {min_decades}")
    
    # 4. Линейная регрессия в log-log координатах
    log_r = np.log(r_tail)
    log_A = np.log(np.abs(A_tail))
    
    # Исключение нулевых значений
    valid_mask = np.isfinite(log_A) & (A_tail != 0)
    log_r_valid = log_r[valid_mask]
    log_A_valid = log_A[valid_mask]
    
    # Регрессия
    slope, intercept, r_squared, _, _ = stats.linregress(log_r_valid, log_A_valid)
    
    # 5. Сравнение с теоретическим значением
    theoretical_slope = 2 * beta - 3
    relative_error = abs(slope - theoretical_slope) / abs(theoretical_slope)
    
    # 6. Критерии приёмки
    passed = (
        r_squared >= 0.99 and           # Высокая корреляция
        relative_error <= 0.05 and      # Ошибка ≤5%
        log_range >= min_decades        # Достаточный диапазон
    )
    
    return {
        'slope': slope,
        'theoretical_slope': theoretical_slope,
        'relative_error': relative_error,
        'r_squared': r_squared,
        'log_range': log_range,
        'passed': passed,
        'radial_profile': radial_profile
    }
```

#### Критерии приёмки
- **R² ≥ 0.99** на диапазоне ≥1.5 декады
- **Ошибка наклона ≤5%** относительно теоретического значения
- **Отсутствие узлов** в радиальном профиле
- **Монотонное убывание** амплитуды с ростом радиуса

#### Вариации параметров
```python
test_variations = {
    "beta_range": [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.4],
    "domain_sizes": [5.0, 10.0, 20.0],
    "resolutions": [256, 512, 1024],
    "boundary_conditions": ["periodic", "pml", "large_torus"]
}
```

### B2. Отсутствие сферических стоячих узлов

#### Физический смысл
Подтверждение того, что в однородной безынтервальной среде **сферические стоячие узлы не являются базовым решением**. Это фундаментальное отличие от классических волновых уравнений.

#### Математическая основа
- **Символ оператора**: D(k) = μk^(2β) не имеет полюсов
- **Следствие**: Нет стационарных осцилляций, только алгебраическое затухание
- **Критерий**: Число смен знака ∂_r A ≤ 1

#### Алгоритм проверки
```python
def check_spherical_nodes(field, center, max_sign_changes=1):
    """
    Проверка отсутствия сферических стоячих узлов
    
    Physical Meaning:
        В чистом фракционном режиме (λ=0) символ оператора D(k) = μk^(2β)
        не имеет полюсов, что исключает формирование сферических стоячих
        волн и приводит к монотонному затуханию поля.
        
    Mathematical Foundation:
        Стоячие узлы возникают при наличии полюсов в символе оператора,
        что соответствует резонансным частотам. Фракционный оператор
        Рисса таких полюсов не имеет.
    """
    # 1. Вычисление радиального профиля
    radial_profile = compute_radial_profile(field, center)
    r = radial_profile['r']
    A = radial_profile['A']
    
    # 2. Вычисление радиальной производной
    dA_dr = np.gradient(A, r)
    
    # 3. Подсчёт смен знака производной
    sign_changes = count_sign_changes(dA_dr)
    
    # 4. Поиск нулей амплитуды
    zeros = find_amplitude_zeros(A, r)
    
    # 5. Проверка на периодичность нулей
    periodic_zeros = check_periodicity(zeros)
    
    # 6. Анализ монотонности
    is_monotonic = check_monotonicity(A, r)
    
    # 7. Критерии приёмки
    passed = (
        sign_changes <= max_sign_changes and  # Минимальные осцилляции
        not periodic_zeros and               # Нет периодических нулей
        is_monotonic                         # Монотонное убывание
    )
    
    return {
        'sign_changes': sign_changes,
        'zeros': zeros,
        'periodic_zeros': periodic_zeros,
        'is_monotonic': is_monotonic,
        'passed': passed,
        'radial_derivative': dA_dr
    }

def count_sign_changes(derivative):
    """Подсчёт смен знака в производной"""
    signs = np.sign(derivative)
    sign_changes = np.sum(np.diff(signs) != 0)
    return sign_changes

def find_amplitude_zeros(amplitude, radius):
    """Поиск нулей амплитуды"""
    # Исключаем область ядра
    core_region = radius < 0.1 * radius.max()
    tail_amplitude = amplitude[~core_region]
    tail_radius = radius[~core_region]
    
    # Поиск пересечений нуля
    zero_crossings = []
    for i in range(len(tail_amplitude) - 1):
        if tail_amplitude[i] * tail_amplitude[i+1] < 0:
            # Линейная интерполяция для точного положения нуля
            r_zero = np.interp(0, 
                             [tail_amplitude[i], tail_amplitude[i+1]], 
                             [tail_radius[i], tail_radius[i+1]])
            zero_crossings.append(r_zero)
    
    return np.array(zero_crossings)

def check_periodicity(zeros, tolerance=0.1):
    """Проверка периодичности нулей"""
    if len(zeros) < 3:
        return False
    
    # Вычисление интервалов между нулями
    intervals = np.diff(zeros)
    
    # Проверка на постоянство интервалов
    if len(intervals) < 2:
        return False
    
    mean_interval = np.mean(intervals)
    relative_std = np.std(intervals) / mean_interval
    
    return relative_std < tolerance
```

#### Критерии приёмки
- **Число смен знака ∂_r A ≤ 1**
- **Отсутствие периодических нулей** A(r) при росте r
- **Монотонное убывание** амплитуды в хвостовой области

### B3. Топологический заряд дефекта

#### Физический смысл
Подтверждение **топологической стабильности** частицы-ядра через вычисление топологического заряда. Это ключевое свойство, обеспечивающее устойчивость элементарных частиц.

#### Математическая основа
- **Топологический заряд**: q = (1/2π) ∮∇φ·dl
- **Физическая интерпретация**: Степень "обмотки" фазового поля вокруг дефекта
- **Стабильность**: Целочисленные значения q ∈ ℤ защищают от непрерывных деформаций

#### Алгоритм вычисления
```python
def compute_topological_charge(field, center, radius=None):
    """
    Вычисление топологического заряда топологического дефекта
    
    Physical Meaning:
        Топологический заряд характеризует степень "обмотки" фазового
        поля вокруг дефекта и обеспечивает его топологическую стабильность.
        Целочисленные значения заряда защищают дефект от непрерывных
        деформаций, что является основой стабильности элементарных частиц.
        
    Mathematical Foundation:
        q = (1/2π) ∮∇φ·dl, где φ - фаза поля, интегрирование по замкнутому
        контуру вокруг дефекта. Для стабильных конфигураций q ∈ ℤ.
    """
    # 1. Вычисление фазы поля
    phase = np.angle(field)
    
    # 2. Определение радиуса интегрирования
    if radius is None:
        radius = estimate_integration_radius(field, center)
    
    # 3. Создание сферического контура
    contour_points = create_spherical_contour(center, radius, n_points=64)
    
    # 4. Вычисление градиента фазы
    grad_phase = compute_phase_gradient(phase, field.shape)
    
    # 5. Интегрирование по контуру
    charge = integrate_phase_around_contour(grad_phase, contour_points)
    
    # 6. Нормализация к 2π
    normalized_charge = charge / (2 * np.pi)
    
    # 7. Проверка на целочисленность
    integer_charge = round(normalized_charge)
    error = abs(normalized_charge - integer_charge)
    
    # 8. Критерии приёмки
    passed = error < 0.01  # Погрешность ≤1%
    
    return {
        'charge': normalized_charge,
        'integer_charge': integer_charge,
        'error': error,
        'passed': passed,
        'contour_points': contour_points,
        'integration_radius': radius
    }

def create_spherical_contour(center, radius, n_points=64):
    """Создание сферического контура для интегрирования"""
    # Создание точек на сфере
    phi = np.linspace(0, 2*np.pi, n_points)
    theta = np.pi/2  # Экваториальная плоскость
    
    x = center[0] + radius * np.cos(phi) * np.sin(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(theta)
    
    return np.column_stack([x, y, z])

def compute_phase_gradient(phase, field_shape):
    """Вычисление градиента фазы"""
    # Использование центральных разностей для точности
    grad_x = np.gradient(phase, axis=0)
    grad_y = np.gradient(phase, axis=1)
    grad_z = np.gradient(phase, axis=2)
    
    return np.stack([grad_x, grad_y, grad_z], axis=-1)

def integrate_phase_around_contour(grad_phase, contour_points):
    """Интегрирование градиента фазы по контуру"""
    charge = 0.0
    
    for i in range(len(contour_points)):
        # Текущая и следующая точки контура
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % len(contour_points)]
        
        # Интерполяция градиента в средней точке
        mid_point = (p1 + p2) / 2
        grad_mid = interpolate_gradient(grad_phase, mid_point)
        
        # Элемент контура
        dl = p2 - p1
        
        # Скалярное произведение
        charge += np.dot(grad_mid, dl)
    
    return charge

def estimate_integration_radius(field, center):
    """Оценка оптимального радиуса для интегрирования"""
    # Анализ радиального профиля для определения границы ядра
    radial_profile = compute_radial_profile(field, center)
    
    # Поиск точки, где амплитуда падает до 10% от максимума
    max_amplitude = np.max(radial_profile['A'])
    threshold = 0.1 * max_amplitude
    
    # Находим первый радиус, где амплитуда ниже порога
    below_threshold = radial_profile['A'] < threshold
    if np.any(below_threshold):
        radius = radial_profile['r'][np.where(below_threshold)[0][0]]
    else:
        # Если не найдено, используем половину размера области
        radius = 0.5 * np.min(field.shape)
    
    return radius
```

#### Критерии приёмки
- **Погрешность топологического заряда ≤1%**
- **Устойчивость к гладким возмущениям**
- **Целочисленные значения** q ∈ ℤ

### B4. Разделение зон (ядро/переход/хвост)

#### Физический смысл
Количественное разделение фазового поля на **три характерные зоны**, каждая из которых играет специфическую роль в формировании свойств частицы.

#### Математическая основа
- **Ядро**: Область высокой плотности и когерентности
- **Переходная зона**: Баланс между ядром и хвостом
- **Хвост**: Линейная волновая область

#### Индикаторы зон
```python
def compute_zone_indicators(field):
    """
    Вычисление индикаторов для разделения зон
    
    Physical Meaning:
        Индикаторы N, S, C характеризуют локальные свойства фазового
        поля и позволяют количественно разделить его на три зоны:
        ядро (высокая плотность, нелинейность), переходная зона
        (баланс), хвост (линейная волновая область).
    """
    # 1. Индикатор плотности N
    N = compute_norm_gradient(field)
    
    # 2. Индикатор кривизны S  
    S = compute_second_derivative(field)
    
    # 3. Индикатор когерентности C
    C = compute_curvature(field)
    
    return {'N': N, 'S': S, 'C': C}

def compute_norm_gradient(field):
    """Вычисление нормы градиента поля"""
    grad_x = np.gradient(field, axis=0)
    grad_y = np.gradient(field, axis=1)
    grad_z = np.gradient(field, axis=2)
    
    # Норма градиента
    N = np.sqrt(np.abs(grad_x)**2 + np.abs(grad_y)**2 + np.abs(grad_z)**2)
    
    return N

def compute_second_derivative(field):
    """Вычисление индикатора второй производной"""
    # Лапласиан поля
    laplacian = compute_laplacian(field)
    
    # Индикатор кривизны
    S = np.abs(laplacian)
    
    return S

def compute_curvature(field):
    """Вычисление индикатора когерентности"""
    # Амплитуда поля
    amplitude = np.abs(field)
    
    # Градиент амплитуды
    grad_amp = np.gradient(amplitude)
    
    # Индикатор когерентности (локальная "жёсткость")
    C = np.sqrt(np.sum([g**2 for g in grad_amp], axis=0))
    
    return C
```

#### Алгоритм разделения
```python
def separate_zones(field, center, thresholds=None):
    """
    Разделение поля на зоны (ядро/переход/хвост)
    
    Physical Meaning:
        Количественное разделение фазового поля на три характерные
        зоны на основе локальных индикаторов. Это позволяет
        проанализировать структуру частицы и понять роль каждой
        зоны в формировании её свойств.
    """
    if thresholds is None:
        thresholds = {
            'N_core': 3.0,      # Порог для ядра
            'S_core': 1.0,      # Порог для ядра
            'N_tail': 0.3,      # Порог для хвоста
            'S_tail': 0.3       # Порог для хвоста
        }
    
    # 1. Вычисление индикаторов
    indicators = compute_zone_indicators(field)
    N = indicators['N']
    S = indicators['S']
    C = indicators['C']
    
    # 2. Нормализация индикаторов
    N_norm = N / np.max(N)
    S_norm = S / np.max(S)
    
    # 3. Разделение по порогам
    core_mask = (N_norm > thresholds['N_core']) & (S_norm > thresholds['S_core'])
    tail_mask = (N_norm < thresholds['N_tail']) & (S_norm < thresholds['S_tail'])
    transition_mask = ~(core_mask | tail_mask)
    
    # 4. Вычисление радиусов зон
    r_core = compute_zone_radius(core_mask, center)
    r_tail = compute_zone_radius(tail_mask, center)
    r_transition = compute_zone_radius(transition_mask, center)
    
    # 5. Статистика по зонам
    zone_stats = compute_zone_statistics(field, core_mask, transition_mask, tail_mask)
    
    return {
        'core_mask': core_mask,
        'transition_mask': transition_mask,
        'tail_mask': tail_mask,
        'r_core': r_core,
        'r_tail': r_tail,
        'r_transition': r_transition,
        'indicators': indicators,
        'zone_stats': zone_stats,
        'thresholds': thresholds
    }

def compute_zone_radius(mask, center):
    """Вычисление эффективного радиуса зоны"""
    if not np.any(mask):
        return 0.0
    
    # Находим все точки зоны
    zone_points = np.where(mask)
    
    # Вычисляем расстояния до центра
    distances = []
    for i, j, k in zip(zone_points[0], zone_points[1], zone_points[2]):
        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
        distances.append(dist)
    
    # Эффективный радиус (среднее расстояние)
    return np.mean(distances) if distances else 0.0

def compute_zone_statistics(field, core_mask, transition_mask, tail_mask):
    """Вычисление статистики по зонам"""
    stats = {}
    
    for zone_name, mask in [('core', core_mask), 
                           ('transition', transition_mask), 
                           ('tail', tail_mask)]:
        if np.any(mask):
            zone_field = field[mask]
            stats[zone_name] = {
                'volume_fraction': np.sum(mask) / mask.size,
                'mean_amplitude': np.mean(np.abs(zone_field)),
                'max_amplitude': np.max(np.abs(zone_field)),
                'std_amplitude': np.std(np.abs(zone_field))
            }
        else:
            stats[zone_name] = {
                'volume_fraction': 0.0,
                'mean_amplitude': 0.0,
                'max_amplitude': 0.0,
                'std_amplitude': 0.0
            }
    
    return stats
```

#### Критерии приёмки
- **Четкое разделение зон** по пороговым значениям
- **Ядро**: N > 3, S > 1
- **Хвост**: N < 0.3, S < 0.3
- **Переходная зона**: Промежуточные значения

## Реализация тестов

### Структура тестового модуля
```python
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level B fundamental properties tests for the 7D phase field theory.

This module implements comprehensive tests for fundamental properties of the
phase field in homogeneous "interval-free" medium, validating the core
theoretical predictions of the 7D phase field theory.

Theoretical Background:
    Tests validate the fundamental behavior of the phase field governed by
    the Riesz operator L_β = μ(-Δ)^β + λ in homogeneous medium, including
    power law tails, absence of spherical nodes, topological charge
    quantization, and zone separation.

Example:
    >>> test_suite = LevelBTests()
    >>> results = test_suite.run_all_tests()
"""

import numpy as np
import pytest
from typing import Dict, Any, Tuple
from scipy import stats
import json
import matplotlib.pyplot as plt

class LevelBTests:
    """
    Comprehensive test suite for Level B fundamental properties.
    
    Physical Meaning:
        Validates the fundamental properties of the phase field in
        homogeneous medium, confirming theoretical predictions about
        power law behavior, topological stability, and zone structure.
        
    Mathematical Foundation:
        Tests are based on the Riesz operator L_β = μ(-Δ)^β + λ and
        its spectral properties in homogeneous medium with periodic
        boundary conditions.
    """
    
    def __init__(self, config_path: str = "configs/level_b_tests.json"):
        """
        Initialize Level B test suite.
        
        Args:
            config_path (str): Path to test configuration file.
        """
        self.config = self._load_config(config_path)
        self.results = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all Level B tests and return comprehensive results.
        
        Returns:
            Dict[str, Any]: Complete test results with analysis.
        """
        test_methods = [
            self.test_B1_power_law_tail,
            self.test_B2_no_spherical_nodes,
            self.test_B3_topological_charge,
            self.test_B4_zone_separation
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            print(f"Running {test_name}...")
            
            try:
                result = test_method()
                self.results[test_name] = result
                print(f"✓ {test_name} passed: {result['passed']}")
            except Exception as e:
                print(f"✗ {test_name} failed: {str(e)}")
                self.results[test_name] = {
                    'passed': False,
                    'error': str(e)
                }
        
        return self.results
    
    def test_B1_power_law_tail(self) -> Dict[str, Any]:
        """
        Test B1: Power law tail in homogeneous medium.
        
        Physical Meaning:
            Validates that the phase field exhibits power law decay
            A(r) ∝ r^(2β-3) in homogeneous medium, confirming the
            fundamental behavior of the Riesz operator.
        """
        # Implementation details...
        pass
    
    def test_B2_no_spherical_nodes(self) -> Dict[str, Any]:
        """
        Test B2: Absence of spherical standing nodes.
        
        Physical Meaning:
            Confirms that spherical standing nodes do not form in
            homogeneous medium, validating the spectral properties
            of the Riesz operator.
        """
        # Implementation details...
        pass
    
    def test_B3_topological_charge(self) -> Dict[str, Any]:
        """
        Test B3: Topological charge of defect.
        
        Physical Meaning:
            Validates the topological stability of the particle core
            through computation of the topological charge.
        """
        # Implementation details...
        pass
    
    def test_B4_zone_separation(self) -> Dict[str, Any]:
        """
        Test B4: Zone separation (core/transition/tail).
        
        Physical Meaning:
            Quantitatively separates the phase field into three
            characteristic zones and validates their properties.
        """
        # Implementation details...
        pass
```

### Конфигурационные файлы
```json
{
    "B1_power_law": {
        "domain": {
            "L": 10.0,
            "N": 512,
            "dimensions": 3
        },
        "physics": {
            "mu": 1.0,
            "beta": 1.0,
            "lambda": 0.0
        },
        "source": {
            "type": "point_source",
            "center": [5.0, 5.0, 5.0],
            "amplitude": 1.0
        },
        "analysis": {
            "min_decades": 1.5,
            "r_squared_threshold": 0.99,
            "error_threshold": 0.05
        },
        "variations": {
            "beta_range": [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.4],
            "domain_sizes": [5.0, 10.0, 20.0],
            "resolutions": [256, 512, 1024]
        }
    },
    "B2_no_nodes": {
        "domain": {
            "L": 10.0,
            "N": 512,
            "dimensions": 3
        },
        "physics": {
            "mu": 1.0,
            "beta": 1.0,
            "lambda": 0.0
        },
        "source": {
            "type": "point_source",
            "center": [5.0, 5.0, 5.0],
            "amplitude": 1.0
        },
        "analysis": {
            "max_sign_changes": 1,
            "periodicity_tolerance": 0.1
        }
    },
    "B3_topological_charge": {
        "domain": {
            "L": 10.0,
            "N": 512,
            "dimensions": 3
        },
        "physics": {
            "mu": 1.0,
            "beta": 1.0,
            "lambda": 0.0
        },
        "source": {
            "type": "point_source",
            "center": [5.0, 5.0, 5.0],
            "amplitude": 1.0
        },
        "analysis": {
            "error_threshold": 0.01,
            "contour_points": 64
        }
    },
    "B4_zone_separation": {
        "domain": {
            "L": 10.0,
            "N": 512,
            "dimensions": 3
        },
        "physics": {
            "mu": 1.0,
            "beta": 1.0,
            "lambda": 0.0
        },
        "source": {
            "type": "point_source",
            "center": [5.0, 5.0, 5.0],
            "amplitude": 1.0
        },
        "analysis": {
            "thresholds": {
                "N_core": 3.0,
                "S_core": 1.0,
                "N_tail": 0.3,
                "S_tail": 0.3
            }
        }
    }
}
```

## Детальные алгоритмы анализа

### 1. Алгоритм обнаружения узлов (B2)
```python
class NodeDetectionAlgorithm:
    """
    Алгоритм обнаружения сферических стоячих узлов.
    
    Physical Meaning:
        Обнаруживает сферические стоячие узлы в фазовом поле,
        которые должны отсутствовать согласно теории фракционного
        оператора Рисса.
        
    Mathematical Foundation:
        - Поиск нулей: A(r) = 0
        - Анализ производных: ∂A/∂r = 0
        - Проверка периодичности: A(r + λ) = A(r)
    """
    
    def __init__(self, tolerance: float = 1e-10, min_separation: float = 0.1):
        """
        Инициализация алгоритма обнаружения узлов.
        
        Args:
            tolerance: Точность обнаружения нулей
            min_separation: Минимальное расстояние между узлами
        """
        self.tolerance = tolerance
        self.min_separation = min_separation
    
    def detect_nodes(self, radial_profile: np.ndarray, r_values: np.ndarray) -> Dict[str, Any]:
        """
        Обнаружение узлов в радиальном профиле.
        
        Physical Meaning:
            Анализирует радиальный профиль A(r) на наличие
            сферических стоячих узлов.
            
        Mathematical Foundation:
            - Поиск пересечений нуля: sign(A(r_i)) ≠ sign(A(r_{i+1}))
            - Интерполяция для точного положения узлов
            - Проверка на периодичность узлов
            
        Args:
            radial_profile: Радиальный профиль A(r)
            r_values: Значения радиуса r
            
        Returns:
            Словарь с результатами анализа узлов
        """
        results = {
            'nodes_found': [],
            'node_count': 0,
            'is_periodic': False,
            'max_sign_changes': 0,
            'analysis_quality': 'unknown'
        }
        
        # Поиск пересечений нуля
        sign_changes = self._find_sign_changes(radial_profile)
        results['max_sign_changes'] = len(sign_changes)
        
        # Интерполяция точных положений узлов
        for i, j in sign_changes:
            node_position = self._interpolate_zero_crossing(
                r_values[i], r_values[j],
                radial_profile[i], radial_profile[j]
            )
            results['nodes_found'].append(node_position)
        
        results['node_count'] = len(results['nodes_found'])
        
        # Проверка на периодичность
        results['is_periodic'] = self._check_periodicity(results['nodes_found'])
        
        # Оценка качества анализа
        results['analysis_quality'] = self._assess_analysis_quality(results)
        
        return results
    
    def _find_sign_changes(self, profile: np.ndarray) -> List[Tuple[int, int]]:
        """
        Поиск смен знака в профиле.
        
        Physical Meaning:
            Находит индексы, где происходит смена знака
            функции A(r), что указывает на возможные нули.
        """
        sign_changes = []
        for i in range(len(profile) - 1):
            if np.sign(profile[i]) != np.sign(profile[i + 1]):
                # Проверяем, что это не просто шум
                if abs(profile[i]) > self.tolerance or abs(profile[i + 1]) > self.tolerance:
                    sign_changes.append((i, i + 1))
        
        return sign_changes
    
    def _interpolate_zero_crossing(self, r1: float, r2: float, 
                                 a1: float, a2: float) -> float:
        """
        Интерполяция точного положения нуля.
        
        Physical Meaning:
            Линейная интерполяция для определения точного
            положения нуля между двумя точками.
        """
        if abs(a2 - a1) < self.tolerance:
            return (r1 + r2) / 2
        
        # Линейная интерполяция
        t = -a1 / (a2 - a1)
        return r1 + t * (r2 - r1)
    
    def _check_periodicity(self, nodes: List[float]) -> bool:
        """
        Проверка периодичности узлов.
        
        Physical Meaning:
            Проверяет, образуют ли узлы периодическую
            структуру, что характерно для стоячих волн.
        """
        if len(nodes) < 3:
            return False
        
        # Вычисляем расстояния между соседними узлами
        distances = [nodes[i+1] - nodes[i] for i in range(len(nodes)-1)]
        
        # Проверяем константность расстояний
        if len(distances) < 2:
            return False
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Если стандартное отклонение мало, узлы периодичны
        return std_distance / mean_distance < 0.1
```

### 2. Алгоритм вычисления топологического заряда (B3)
```python
class TopologicalChargeAlgorithm:
    """
    Алгоритм вычисления топологического заряда.
    
    Physical Meaning:
        Вычисляет топологический заряд q = (1/2π) ∮∇φ·dl
        для фазового поля, характеризующий топологические
        дефекты и их устойчивость.
        
    Mathematical Foundation:
        - Контурное интегрирование: ∮∇φ·dl
        - Дискретизация: Σᵢ (φᵢ₊₁ - φᵢ)
        - Нормализация: q = (1/2π) × интеграл
    """
    
    def __init__(self, contour_points: int = 64, integration_method: str = 'trapezoidal'):
        """
        Инициализация алгоритма.
        
        Args:
            contour_points: Количество точек на контуре
            integration_method: Метод интегрирования
        """
        self.contour_points = contour_points
        self.integration_method = integration_method
    
    def compute_topological_charge(self, field: np.ndarray, center: Tuple[int, int, int],
                                 radius: float) -> Dict[str, float]:
        """
        Вычисление топологического заряда.
        
        Physical Meaning:
            Вычисляет топологический заряд для кругового
            контура вокруг центра дефекта.
            
        Mathematical Foundation:
            q = (1/2π) ∮∇φ·dl = (1/2π) ∮(∂φ/∂θ)dθ
            
        Args:
            field: Комплексное фазовое поле
            center: Центр контура (x, y, z)
            radius: Радиус контура
            
        Returns:
            Словарь с результатами вычисления
        """
        # Создание контура
        contour = self._create_circular_contour(center, radius)
        
        # Вычисление фазы на контуре
        phase_values = self._extract_phase_on_contour(field, contour)
        
        # Вычисление топологического заряда
        charge = self._integrate_phase_gradient(phase_values)
        
        # Анализ качества
        quality_metrics = self._assess_integration_quality(phase_values, charge)
        
        return {
            'topological_charge': charge,
            'charge_rounded': round(charge),
            'integration_quality': quality_metrics,
            'contour_points': len(contour),
            'radius': radius
        }
    
    def _create_circular_contour(self, center: Tuple[int, int, int], 
                               radius: float) -> List[Tuple[float, float, float]]:
        """
        Создание кругового контура.
        
        Physical Meaning:
            Создает дискретный круговой контур для
            контурного интегрирования.
        """
        contour = []
        for i in range(self.contour_points):
            angle = 2 * np.pi * i / self.contour_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]  # Контур в плоскости z = const
            contour.append((x, y, z))
        
        return contour
    
    def _extract_phase_on_contour(self, field: np.ndarray, 
                                contour: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Извлечение фазы на контуре.
        
        Physical Meaning:
            Вычисляет фазу φ поля в точках контура
            для последующего интегрирования.
        """
        phase_values = []
        for point in contour:
            # Интерполяция поля в точке контура
            field_value = self._interpolate_field_at_point(field, point)
            phase = np.angle(field_value)
            phase_values.append(phase)
        
        return np.array(phase_values)
    
    def _integrate_phase_gradient(self, phase_values: np.ndarray) -> float:
        """
        Интегрирование градиента фазы.
        
        Physical Meaning:
            Вычисляет контурный интеграл градиента фазы
            для определения топологического заряда.
        """
        # Вычисляем разности фаз
        phase_diffs = np.diff(phase_values)
        
        # Обрабатываем разрыв в 2π
        phase_diffs = np.unwrap(phase_diffs)
        
        # Интегрируем
        if self.integration_method == 'trapezoidal':
            integral = np.sum(phase_diffs)
        else:
            integral = np.sum(phase_diffs)
        
        # Нормализация
        topological_charge = integral / (2 * np.pi)
        
        return topological_charge
```

### 3. Алгоритм разделения зон (B4)
```python
class ZoneSeparationAlgorithm:
    """
    Алгоритм разделения поля на зоны.
    
    Physical Meaning:
        Разделяет фазовое поле на зоны: ядро, переходная
        зона и хвост, на основе анализа радиального профиля
        и его производных.
        
    Mathematical Foundation:
        - Ядро: |A(r)| > N_core, |∂A/∂r| > S_core
        - Хвост: |A(r)| < N_tail, |∂A/∂r| < S_tail
        - Переходная зона: между ядром и хвостом
    """
    
    def __init__(self, thresholds: Dict[str, float]):
        """
        Инициализация алгоритма.
        
        Args:
            thresholds: Пороговые значения для разделения зон
        """
        self.thresholds = thresholds
    
    def separate_zones(self, radial_profile: np.ndarray, 
                      r_values: np.ndarray) -> Dict[str, Any]:
        """
        Разделение поля на зоны.
        
        Physical Meaning:
            Анализирует радиальный профиль и разделяет
            его на зоны согласно физическим критериям.
            
        Args:
            radial_profile: Радиальный профиль A(r)
            r_values: Значения радиуса r
            
        Returns:
            Словарь с результатами разделения зон
        """
        # Вычисляем производную
        derivative = np.gradient(radial_profile, r_values)
        
        # Определяем зоны
        core_zone = self._identify_core_zone(radial_profile, derivative, r_values)
        tail_zone = self._identify_tail_zone(radial_profile, derivative, r_values)
        transition_zone = self._identify_transition_zone(core_zone, tail_zone, r_values)
        
        # Анализ качества разделения
        quality_metrics = self._assess_zone_separation_quality(
            core_zone, tail_zone, transition_zone
        )
        
        return {
            'core_zone': core_zone,
            'tail_zone': tail_zone,
            'transition_zone': transition_zone,
            'zone_boundaries': {
                'core_boundary': core_zone['r_max'] if core_zone else 0,
                'tail_boundary': tail_zone['r_min'] if tail_zone else float('inf')
            },
            'quality_metrics': quality_metrics
        }
    
    def _identify_core_zone(self, profile: np.ndarray, derivative: np.ndarray,
                          r_values: np.ndarray) -> Dict[str, Any]:
        """
        Идентификация ядра.
        
        Physical Meaning:
            Определяет ядро как область с высокой амплитудой
            и большими градиентами.
        """
        core_mask = (
            (np.abs(profile) > self.thresholds['N_core']) &
            (np.abs(derivative) > self.thresholds['S_core'])
        )
        
        if not np.any(core_mask):
            return None
        
        core_indices = np.where(core_mask)[0]
        r_min = r_values[core_indices[0]]
        r_max = r_values[core_indices[-1]]
        
        return {
            'r_min': r_min,
            'r_max': r_max,
            'indices': core_indices,
            'amplitude_range': [np.min(np.abs(profile[core_indices])),
                              np.max(np.abs(profile[core_indices]))],
            'derivative_range': [np.min(np.abs(derivative[core_indices])),
                               np.max(np.abs(derivative[core_indices]))]
        }
    
    def _identify_tail_zone(self, profile: np.ndarray, derivative: np.ndarray,
                          r_values: np.ndarray) -> Dict[str, Any]:
        """
        Идентификация хвоста.
        
        Physical Meaning:
            Определяет хвост как область с малой амплитудой
            и малыми градиентами.
        """
        tail_mask = (
            (np.abs(profile) < self.thresholds['N_tail']) &
            (np.abs(derivative) < self.thresholds['S_tail'])
        )
        
        if not np.any(tail_mask):
            return None
        
        tail_indices = np.where(tail_mask)[0]
        r_min = r_values[tail_indices[0]]
        r_max = r_values[tail_indices[-1]]
        
        return {
            'r_min': r_min,
            'r_max': r_max,
            'indices': tail_indices,
            'amplitude_range': [np.min(np.abs(profile[tail_indices])),
                              np.max(np.abs(profile[tail_indices]))],
            'derivative_range': [np.min(np.abs(derivative[tail_indices])),
                               np.max(np.abs(derivative[tail_indices]))]
        }
```

## Критерии готовности

### Обязательные требования
- [ ] **Все тесты B1-B4 реализованы** с полной функциональностью
- [ ] **Детальные алгоритмы анализа** реализованы и протестированы
- [ ] **Алгоритм обнаружения узлов** работает корректно
- [ ] **Алгоритм вычисления топологического заряда** точен
- [ ] **Алгоритм разделения зон** функционален
- [ ] **Все тесты проходят** с требуемой точностью
- [ ] **Визуализация результатов** создана и функциональна
- [ ] **Конфигурации тестов** настроены и валидированы
- [ ] **Документация** написана на английском языке
- [ ] **Примеры использования** созданы и протестированы

### Численные критерии
- **B1**: R² ≥ 0.99 на ≥1.5 декады, ошибка наклона ≤5%
- **B2**: Число смен знака ∂_r A ≤ 1, отсутствие периодических нулей
- **B3**: Погрешность топологического заряда ≤1%
- **B4**: Четкое разделение зон по пороговым значениям

### Технические требования
- **Высокая точность**: float64 для всех вычислений
- **Корректная обработка ГУ**: периодические граничные условия
- **Валидация данных**: проверка входных параметров
- **Детальное логирование**: полная трассировка процесса
- **Автоматизация**: интеграция в CI/CD pipeline

## Выходные данные

### Аналитические результаты
- `power_law_analysis.json` - детальный анализ степенного хвоста
- `nodes_analysis.json` - проверка отсутствия узлов
- `topological_charge.json` - результаты вычисления топологического заряда
- `zone_separation.json` - анализ разделения на зоны

### Визуализация
- `radial_profiles.png` - радиальные профили для всех тестов
- `zone_maps.png` - карты зон с индикаторами
- `topological_analysis.png` - визуализация топологического анализа
- `power_law_fits.png` - графики фитирования степенных законов

### Метрики и статистика
- Все численные метрики в JSON формате
- Статистика по различным параметрам и вариациям
- Сравнение с теоретическими предсказаниями
- Анализ чувствительности к параметрам

## Следующий шаг
После успешного завершения Step 05 переходим к **Step 06: Реализация моделей границ и ячеек уровня C**, где будут исследованы эффекты границ, резонаторов и квенч-памяти в неоднородной среде.
