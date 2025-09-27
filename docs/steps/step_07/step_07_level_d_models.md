# Step 07: Многомодовые модели и проекции полей уровня D

## Цель
Реализовать модели многомодового наложения и проекции полей (ЭМ/сильное/слабое) как огибающих различных частотных диапазонов.

## Математическая основа

### Многомодовое наложение
```
a(x,t) = Σ_m A_m(T) φ_m(x) e^(-iω_m t)
```

### Проекция полей как огибающих
- **EM поле**: Градиенты фазы огибающей (U(1)-тип), токи ∝ ∇φ
- **Сильное поле**: Локальные высоко-Q моды/крутые градиенты амплитуды у ядра
- **Слабое поле**: Хиральные/паритет-ломающие комбинации огибающих малых Q

## Структура тестов

### D1. Наложение мод на каркас
- **Цель**: Стабильность «каркаса» при добавлении новых мод
- **Постановка**: Последовательно добавлять возбуждения разных ω
- **Критерии**: Карта максимумов I_eff (Жаккар ≥0.8 до/после), пики Y(ω) сохраняют частоты в пределах 3–5%

### D2. «ЭМ/сильное/слабое» как окна огибающих (прокси-тест)
- **Цель**: Операциональная проверка проекции полей
- **Постановка**: Три частотно-амплитудных окна:
  - «EM»: слабонелинейная зона хвоста → проверка потоков ∝∇φ
  - «сильное»: высоко-Q локальные моды у ядра
  - «слабое»: хиральные комбинации, малый Q/утечки
- **Критерии**: Воспроизводимость характерных подписей (локализация/дальнодействие/анизотропия фазовых градиентов)

### D3. Пассивные спирали/линии потока
- **Цель**: Сопоставить «вихревые» линии с ∇φ
- **Постановка**: Трассировка линий уровня φ и поля ∇φ вокруг дефекта
- **Критерии**: Визуальное соответствие линий потока и фазовых градиентов

## Реализация моделей

### 1. Структура моделей (models/level_d.py)
```python
class LevelDModels:
    def create_multi_mode_field(self, domain, modes):
        """Создание многомодового поля"""
        
    def analyze_mode_superposition(self, field, new_modes):
        """Анализ наложения мод на каркас"""
        
    def project_field_windows(self, field, window_params):
        """Проекция полей на различные окна"""
        
    def trace_phase_streamlines(self, field, center):
        """Трассировка линий потока фазы"""
```

### 2. Модель многомодового наложения
```python
class MultiModeModel:
    def __init__(self, base_field, mode_parameters):
        """Инициализация многомодовой модели"""
        
    def add_mode(self, frequency, amplitude, phase):
        """Добавление новой моды"""
        
    def analyze_frame_stability(self, before, after):
        """Анализ стабильности каркаса"""
        
    def compute_jaccard_index(self, map1, map2):
        """Вычисление индекса Жаккара"""
```

### 3. Проекция полей
```python
class FieldProjection:
    def __init__(self, field, projection_params):
        """Инициализация проекции полей"""
        
    def project_em_field(self, field):
        """Проекция ЭМ поля"""
        
    def project_strong_field(self, field):
        """Проекция сильного поля"""
        
    def project_weak_field(self, field):
        """Проекция слабого поля"""
        
    def analyze_field_signatures(self, projections):
        """Анализ подписей полей"""
```

## Алгоритмы анализа

### 1. Анализ наложения мод
```python
def analyze_mode_superposition(base_field, new_modes):
    """Анализ наложения мод на каркас"""
    # Создание многомодового поля
    multi_mode_field = create_multi_mode_field(base_field, new_modes)
    
    # Анализ каркаса до и после
    frame_before = extract_frame(base_field)
    frame_after = extract_frame(multi_mode_field)
    
    # Вычисление индекса Жаккара
    jaccard_index = compute_jaccard_index(frame_before, frame_after)
    
    # Анализ частотных пиков
    peaks_before = extract_frequency_peaks(base_field)
    peaks_after = extract_frequency_peaks(multi_mode_field)
    
    # Проверка стабильности частот
    frequency_stability = check_frequency_stability(peaks_before, peaks_after)
    
    return {
        'jaccard_index': jaccard_index,
        'frequency_stability': frequency_stability,
        'frame_before': frame_before,
        'frame_after': frame_after,
        'passed': jaccard_index >= 0.8 and frequency_stability < 0.05
    }
```

### 2. Проекция полей на окна
```python
def project_field_windows(field, window_params):
    """Проекция полей на различные частотно-амплитудные окна"""
    # Проекция ЭМ поля
    em_projection = project_em_field(field, window_params['em'])
    
    # Проекция сильного поля
    strong_projection = project_strong_field(field, window_params['strong'])
    
    # Проекция слабого поля
    weak_projection = project_weak_field(field, window_params['weak'])
    
    # Анализ подписей
    signatures = analyze_field_signatures({
        'em': em_projection,
        'strong': strong_projection,
        'weak': weak_projection
    })
    
    return {
        'em_projection': em_projection,
        'strong_projection': strong_projection,
        'weak_projection': weak_projection,
        'signatures': signatures
    }
```

### 3. Анализ линий потока фазы
```python
def trace_phase_streamlines(field, center):
    """Трассировка линий потока фазы"""
    # Вычисление фазы поля
    phase = np.angle(field)
    
    # Вычисление градиента фазы
    phase_gradient = compute_phase_gradient(phase)
    
    # Трассировка линий потока
    streamlines = trace_streamlines(phase_gradient, center)
    
    # Анализ топологии линий
    topology = analyze_streamline_topology(streamlines)
    
    return {
        'phase': phase,
        'phase_gradient': phase_gradient,
        'streamlines': streamlines,
        'topology': topology
    }
```

## Конфигурации тестов

### 1. Конфигурация D1 (configs/level_d_tests.json)
```json
{
    "D1": {
        "domain": {"L": 10.0, "N": 512},
        "physics": {"mu": 1.0, "beta": 1.0, "lambda": 0.0},
        "base_modes": [
            {"frequency": 1.0, "amplitude": 1.0, "phase": 0.0},
            {"frequency": 2.0, "amplitude": 0.5, "phase": 0.0}
        ],
        "new_modes": [
            {"frequency": 1.5, "amplitude": 0.3, "phase": 0.0},
            {"frequency": 2.5, "amplitude": 0.2, "phase": 0.0}
        ],
        "analysis": {"jaccard_threshold": 0.8, "frequency_tolerance": 0.05}
    }
}
```

### 2. Конфигурация D2
```json
{
    "D2": {
        "domain": {"L": 10.0, "N": 512},
        "physics": {"mu": 1.0, "beta": 1.0, "lambda": 0.0},
        "field_windows": {
            "em": {"frequency_range": [0.1, 1.0], "amplitude_threshold": 0.1},
            "strong": {"frequency_range": [1.0, 10.0], "q_threshold": 100},
            "weak": {"frequency_range": [0.01, 0.1], "q_threshold": 10}
        },
        "analysis": {"signature_analysis": true}
    }
}
```

## Критерии приёмки

### Численные допуски
- D1: Индекс Жаккара ≥ 0.8, стабильность частот < 5%
- D2: Воспроизводимость характерных подписей полей
- D3: Визуальное соответствие линий потока и фазовых градиентов

### Требования к реализации
- Корректная обработка многомодовых полей
- Высокая точность вычислений
- Валидация результатов
- Детальное логирование

## Выходные данные

### 1. Аналитические результаты
- mode_superposition_analysis.json - анализ наложения мод
- field_projection_analysis.json - анализ проекции полей
- phase_streamline_analysis.json - анализ линий потока фазы

### 2. Визуализация
- mode_superposition.png - наложение мод
- field_projections.png - проекции полей
- phase_streamlines.png - линии потока фазы
- field_signatures.png - подписи полей

### 3. Метрики
- Все численные метрики в JSON формате
- Статистика по различным параметрам
- Сравнение с теоретическими предсказаниями

## Критерии готовности
- [ ] Реализованы все модели D1–D3
- [ ] Алгоритмы анализа работают корректно
- [ ] Все тесты проходят с требуемой точностью
- [ ] Проекция полей реализована и протестирована
- [ ] Визуализация результатов создана
- [ ] Конфигурации тестов настроены
- [ ] Документация написана
- [ ] Примеры использования созданы

## Следующий шаг
Step 08: Реализация численных экспериментов уровня E (солитоны и дефекты)
