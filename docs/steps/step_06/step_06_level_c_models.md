# Step 06: Модели границ и ячеек уровня C

## Цель
Реализовать модели границ, ячеек и принципа последовательных резонаторов для изучения взаимодействия фазового поля с граничными условиями.

## Математическая основа

### Модель огибающей
Разделяем быстрый носитель и медленную амплитуду:
```
a(x,t) = Σ_m A_m(T) φ_m(x) e^(-iω_m t), T ≫ ω_m^(-1)
```

### Уравнение на огибающую
```
∇·(κ(|a|) ∇a) + k₀²χ(|a|) a = s(x), k₀ = ω₀/c_φ
```
где:
- κ > 0 — «жёсткость»
- χ = χ' + iχ'' — восприимчивость (потери/квенч — в χ'')

### Энергетический функционал
```
E[a] = ∫_Ω (κ|∇a|² + k₀²χ'|a|²) dV + (диссипативная часть от χ'')
```

## Структура тестов

### C1. Одна «стенка» (контраст адмиттансы)
- **Цель**: Рождение ячейки/псевдорезонатора
- **Постановка**: Сферическая оболочка с контрастом η = |ΔY|/⟨Y⟩
- **Наблюдаемое**: Пики Y(ω), локальные максимумы A(r) между ядром и оболочкой
- **Критерии**: Порог η* для появления первой моды

### C2. Цепочка резонаторов (ABCD)
- **Цель**: Принцип последовательных резонаторов
- **Постановка**: 2–5 вложенных оболочек (разные η, Δr)
- **Критерии**: Согласование Y_tot(ω) с ABCD-моделью (ошибка мод менее 5% по ω_n, 10% по Q_n)

### C3. Квенч-память/пиннинг
- **Цель**: Статичность огибающей относительно дефекта
- **Постановка**: Добавить диссипативно-памятный член (ядро/оболочка)
- **Наблюдаемое**: Скорость дрейфа ячеек v_cell по кросс-корреляции I_eff(t)
- **Критерии**: v_cell → <10⁻³L/T₀; подавление биений

### C4. Биения двух близких мод
- **Цель**: Различить дрейф/«заморозку» решётки
- **Постановка**: Возбудить ω₁ ≈ ω₂, измерить Δω, Δk
- **Критерии**: v_cell ≈ Δω/|Δk| без пиннинга и → 0 при пиннинге

## Реализация моделей

### 1. Структура моделей (models/level_c.py)
```python
class LevelCModels:
    def create_spherical_boundary(self, domain, center, radius, contrast):
        """Создание сферической границы с контрастом"""
        
    def create_resonator_chain(self, domain, shells):
        """Создание цепочки резонаторов"""
        
    def add_quench_memory(self, field, memory_params):
        """Добавление квенч-памяти/пиннинга"""
        
    def analyze_mode_beating(self, field, frequencies):
        """Анализ биений мод"""
```

### 2. Модель резонатора
```python
class ResonatorModel:
    def __init__(self, geometry, material_params):
        """Инициализация модели резонатора"""
        
    def compute_admittance(self, frequency):
        """Вычисление адмиттансы Y(ω)"""
        
    def find_resonance_modes(self, frequency_range):
        """Поиск резонансных мод"""
        
    def analyze_quality_factors(self, modes):
        """Анализ добротности мод"""
```

### 3. ABCD модель
```python
class ABCDModel:
    def __init__(self, resonators):
        """Инициализация ABCD модели"""
        
    def compute_transfer_matrix(self, frequency):
        """Вычисление матрицы передачи"""
        
    def find_system_modes(self, frequency_range):
        """Поиск системных мод"""
        
    def compare_with_numerical(self, numerical_modes):
        """Сравнение с численными результатами"""
```

## Алгоритмы анализа

### 1. Анализ контраста адмиттансы
```python
def analyze_admittance_contrast(field, boundary_geometry):
    """Анализ контраста адмиттансы на границе"""
    # Вычисление адмиттансы внутри и снаружи границы
    Y_inside = compute_admittance(field, boundary_geometry, 'inside')
    Y_outside = compute_admittance(field, boundary_geometry, 'outside')
    
    # Контраст
    contrast = abs(Y_inside - Y_outside) / ((Y_inside + Y_outside) / 2)
    
    # Поиск резонансных мод
    resonance_modes = find_resonance_modes(field, boundary_geometry)
    
    return {
        'contrast': contrast,
        'resonance_modes': resonance_modes,
        'threshold': find_contrast_threshold(resonance_modes)
    }
```

### 2. Анализ цепочки резонаторов
```python
def analyze_resonator_chain(field, shells):
    """Анализ цепочки резонаторов"""
    # Создание ABCD модели
    abcd_model = ABCDModel(shells)
    
    # Вычисление системных мод
    system_modes = abcd_model.find_system_modes()
    
    # Сравнение с численными результатами
    numerical_modes = extract_numerical_modes(field)
    
    # Анализ ошибок
    frequency_errors = compute_frequency_errors(system_modes, numerical_modes)
    quality_errors = compute_quality_errors(system_modes, numerical_modes)
    
    return {
        'system_modes': system_modes,
        'numerical_modes': numerical_modes,
        'frequency_errors': frequency_errors,
        'quality_errors': quality_errors,
        'passed': max(frequency_errors) < 0.05 and max(quality_errors) < 0.10
    }
```

### 3. Анализ квенч-памяти
```python
def analyze_quench_memory(field, memory_params):
    """Анализ квенч-памяти и пиннинга"""
    # Добавление памяти к полю
    field_with_memory = add_memory_to_field(field, memory_params)
    
    # Временная эволюция
    time_evolution = evolve_field_in_time(field_with_memory)
    
    # Анализ дрейфа ячеек
    cell_drift = analyze_cell_drift(time_evolution)
    
    # Кросс-корреляция
    cross_correlation = compute_cross_correlation(time_evolution)
    
    return {
        'cell_drift': cell_drift,
        'cross_correlation': cross_correlation,
        'drift_velocity': compute_drift_velocity(cell_drift),
        'passed': cell_drift['velocity'] < 1e-3
    }
```

### 4. Анализ биений мод
```python
def analyze_mode_beating(field, frequencies):
    """Анализ биений двух близких мод"""
    # Возбуждение двух мод
    field_dual = excite_dual_modes(field, frequencies)
    
    # Временная эволюция
    time_evolution = evolve_field_in_time(field_dual)
    
    # Анализ биений
    beating_analysis = analyze_beating_pattern(time_evolution)
    
    # Вычисление скорости дрейфа
    drift_velocity = compute_drift_velocity(beating_analysis)
    
    # Сравнение с теоретическим значением
    theoretical_velocity = compute_theoretical_velocity(frequencies)
    
    return {
        'beating_analysis': beating_analysis,
        'drift_velocity': drift_velocity,
        'theoretical_velocity': theoretical_velocity,
        'error': abs(drift_velocity - theoretical_velocity) / theoretical_velocity,
        'passed': abs(drift_velocity - theoretical_velocity) / theoretical_velocity < 0.05
    }
```

## Конфигурации тестов

### 1. Конфигурация C1 (configs/level_c_tests.json)
```json
{
    "C1": {
        "domain": {"L": 10.0, "N": 512},
        "physics": {"mu": 1.0, "beta": 1.0, "lambda": 0.0},
        "boundary": {
            "type": "spherical",
            "center": [5.0, 5.0, 5.0],
            "radius": 2.0,
            "contrast": 0.5
        },
        "analysis": {"find_resonance_modes": true}
    }
}
```

### 2. Конфигурация C2
```json
{
    "C2": {
        "domain": {"L": 10.0, "N": 512},
        "physics": {"mu": 1.0, "beta": 1.0, "lambda": 0.0},
        "resonators": {
            "shells": [
                {"radius": 1.0, "contrast": 0.3},
                {"radius": 2.0, "contrast": 0.5},
                {"radius": 3.0, "contrast": 0.7}
            ]
        },
        "analysis": {"abcd_comparison": true}
    }
}
```

## Критерии приёмки

### Численные допуски
- C1: Порог η* для появления первой моды определен
- C2: Ошибка мод менее 5% по ω_n, 10% по Q_n
- C3: v_cell < 10⁻³L/T₀
- C4: Ошибка скорости дрейфа < 5%

### Требования к реализации
- Корректная обработка граничных условий
- Высокая точность вычислений
- Валидация результатов
- Детальное логирование

## Выходные данные

### 1. Аналитические результаты
- admittance_analysis.json - анализ адмиттансы
- resonator_chain_analysis.json - анализ цепочки резонаторов
- quench_memory_analysis.json - анализ квенч-памяти
- mode_beating_analysis.json - анализ биений мод

### 2. Визуализация
- boundary_effects.png - эффекты границ
- resonator_modes.png - моды резонаторов
- memory_effects.png - эффекты памяти
- beating_patterns.png - паттерны биений

## Критерии готовности
- [ ] Реализованы все модели C1–C4
- [ ] Алгоритмы анализа работают корректно
- [ ] Все тесты проходят с требуемой точностью
- [ ] ABCD модель реализована и протестирована
- [ ] Визуализация результатов создана
- [ ] Конфигурации тестов настроены
- [ ] Документация написана
- [ ] Примеры использования созданы

## Следующий шаг
Step 07: Создание многомодовых моделей и проекций полей уровня D
