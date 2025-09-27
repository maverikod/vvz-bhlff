# Step 05: Тесты фундаментальных свойств поля уровня B

## Цель
Реализовать тесты для подтверждения численно базовых свойств безынтервального поля (БВП) в однородной среде с периодическими ГУ.

## Математическая основа

### Область и оператор
- Область: Ω = [0,L)³, периодические ГУ
- Поле: a(x) ∈ ℂ
- Оператор Рисса: L_β a = μ(-Δ)^β a + λa
- Стационарная задача: L_β a = s(x)

### Спектральная форма
- Дискретные волновые числа: k = (2π/L)m, m ∈ ℤ³
- В k-области: â(k) = ŝ(k)/(μ|k|^(2β) + λ)

## Структура тестов

### B1. Степенной хвост в однородной среде («космос»)
- **Цель**: Подтвердить A(r) ∝ r^(2β-3) при λ=0
- **Постановка**: Точечный дефект (ядро), однородные параметры; стац. задача
- **Наблюдаемое**: Линейный тренд в log A – log r, R²≥0.99 на ≥1.5 декады; отсутствие узлов
- **Вариации**: β ∈ [0.5,1.4], разные объёмы, разные ГУ (PML/большой тор)

### B2. Отсутствие сферических стоячих узлов
- **Цель**: Показать, что в однородной «безынтервальной» среде узловые сферы не формируются
- **Постановка**: Как B1
- **Критерии**: Число смен знака ∂_r A ≤ 1; отсутствие периодических нулей A(r) при росте r

### B3. Топологический заряд дефекта
- **Цель**: Стабилизация «частицы»-ядра
- **Постановка**: Численно интегрировать ∮∇φ·dl вокруг ядра
- **Критерии**: 2πq с погрешностью ≤1%; устойчивость к гладким возмущениям

### B4. Разделение зон (ядро/переход/хвост)
- **Цель**: Количественно отделить три зоны
- **Постановка**: Посчитать индикаторы N,S,C и радиусы r_core, r_tail
- **Критерии**: Пороги, напр. ядро N>3, S>1; хвост N<0.3, S<0.3

## Реализация тестов

### 1. Структура тестов (tests/test_level_b.py)
```python
class TestLevelB:
    def test_B1_power_law_tail(self):
        """Тест B1: Степенной хвост в однородной среде"""
        
    def test_B2_no_spherical_nodes(self):
        """Тест B2: Отсутствие сферических стоячих узлов"""
        
    def test_B3_topological_charge(self):
        """Тест B3: Топологический заряд дефекта"""
        
    def test_B4_zone_separation(self):
        """Тест B4: Разделение зон (ядро/переход/хвост)"""
```

### 2. Модели для тестов (models/level_b.py)
```python
class LevelBModels:
    def create_point_source(self, domain, physics_params):
        """Создание точечного источника"""
        
    def analyze_power_law_tail(self, field, beta):
        """Анализ степенного хвоста"""
        
    def compute_topological_charge(self, field, center):
        """Вычисление топологического заряда"""
        
    def separate_zones(self, field, thresholds):
        """Разделение на зоны"""
```

### 3. Конфигурации тестов (configs/level_b_tests.json)
```json
{
    "B1": {
        "domain": {"L": 10.0, "N": 512},
        "physics": {"mu": 1.0, "beta": 1.0, "lambda": 0.0},
        "forcing": {"type": "point_source", "center": [5.0, 5.0, 5.0]},
        "analysis": {"power_law_fit": true, "min_decades": 1.5}
    },
    "B2": {
        "domain": {"L": 10.0, "N": 512},
        "physics": {"mu": 1.0, "beta": 1.0, "lambda": 0.0},
        "forcing": {"type": "point_source", "center": [5.0, 5.0, 5.0]},
        "analysis": {"check_nodes": true, "max_sign_changes": 1}
    }
}
```

## Алгоритмы анализа

### 1. Анализ степенного хвоста
```python
def analyze_power_law_tail(field, beta, center):
    """Анализ степенного хвоста A(r) ∝ r^(2β-3)"""
    # Вычисление радиального профиля
    radial_profile = compute_radial_profile(field, center)
    
    # Линейная регрессия в log-log координатах
    log_r = np.log(radial_profile['r'])
    log_A = np.log(radial_profile['A'])
    
    # Фитирование степенного закона
    slope, intercept, r_squared = linear_regression(log_r, log_A)
    
    # Проверка соответствия теоретическому значению
    theoretical_slope = 2*beta - 3
    error = abs(slope - theoretical_slope) / abs(theoretical_slope)
    
    return {
        'slope': slope,
        'theoretical_slope': theoretical_slope,
        'error': error,
        'r_squared': r_squared,
        'passed': error < 0.05 and r_squared > 0.99
    }
```

### 2. Проверка отсутствия узлов
```python
def check_spherical_nodes(field, center):
    """Проверка отсутствия сферических стоячих узлов"""
    radial_profile = compute_radial_profile(field, center)
    
    # Подсчет смен знака производной
    dA_dr = np.gradient(radial_profile['A'], radial_profile['r'])
    sign_changes = count_sign_changes(dA_dr)
    
    # Проверка на периодические нули
    zeros = find_zeros(radial_profile['A'])
    periodic_zeros = check_periodicity(zeros)
    
    return {
        'sign_changes': sign_changes,
        'zeros': zeros,
        'periodic_zeros': periodic_zeros,
        'passed': sign_changes <= 1 and not periodic_zeros
    }
```

### 3. Вычисление топологического заряда
```python
def compute_topological_charge(field, center):
    """Вычисление топологического заряда"""
    # Вычисление фазы поля
    phase = np.angle(field)
    
    # Интегрирование по контуру вокруг центра
    charge = integrate_phase_around_contour(phase, center)
    
    # Нормализация к 2π
    normalized_charge = charge / (2 * np.pi)
    
    # Проверка на целочисленность
    integer_charge = round(normalized_charge)
    error = abs(normalized_charge - integer_charge)
    
    return {
        'charge': normalized_charge,
        'integer_charge': integer_charge,
        'error': error,
        'passed': error < 0.01
    }
```

### 4. Разделение на зоны
```python
def separate_zones(field, thresholds):
    """Разделение поля на зоны (ядро/переход/хвост)"""
    # Вычисление индикаторов
    N = compute_norm_gradient(field)
    S = compute_second_derivative(field)
    C = compute_curvature(field)
    
    # Разделение по порогам
    core_mask = (N > thresholds['N_core']) & (S > thresholds['S_core'])
    tail_mask = (N < thresholds['N_tail']) & (S < thresholds['S_tail'])
    transition_mask = ~(core_mask | tail_mask)
    
    # Вычисление радиусов
    r_core = compute_radius(core_mask)
    r_tail = compute_radius(tail_mask)
    
    return {
        'core_mask': core_mask,
        'transition_mask': transition_mask,
        'tail_mask': tail_mask,
        'r_core': r_core,
        'r_tail': r_tail,
        'indicators': {'N': N, 'S': S, 'C': C}
    }
```

## Критерии приёмки

### Численные допуски
- B1: R² ≥ 0.99 на ≥1.5 декады, ошибка наклона ≤5%
- B2: Число смен знака ∂_r A ≤ 1, отсутствие периодических нулей
- B3: Погрешность топологического заряда ≤1%
- B4: Четкое разделение зон по пороговым значениям

### Требования к реализации
- Высокая точность вычислений (float64)
- Корректная обработка граничных условий
- Валидация входных данных
- Детальное логирование процесса

## Выходные данные

### 1. Аналитические результаты
- power_law_analysis.json - анализ степенного хвоста
- nodes_analysis.json - проверка узлов
- topological_charge.json - топологический заряд
- zone_separation.json - разделение на зоны

### 2. Визуализация
- radial_profiles.png - радиальные профили
- zone_maps.png - карты зон
- topological_analysis.png - топологический анализ

### 3. Метрики
- Все численные метрики в JSON формате
- Статистика по различным параметрам
- Сравнение с теоретическими предсказаниями

## Критерии готовности
- [ ] Реализованы все тесты B1–B4
- [ ] Алгоритмы анализа работают корректно
- [ ] Все тесты проходят с требуемой точностью
- [ ] Визуализация результатов создана
- [ ] Конфигурации тестов настроены
- [ ] Документация написана
- [ ] Примеры использования созданы
- [ ] Автоматизация тестирования настроена

## Следующий шаг
Step 06: Реализация моделей границ и ячеек уровня C
