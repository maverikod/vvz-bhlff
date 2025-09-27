# Step 11: Система анализа и визуализации результатов

## Цель
Создать комплексную систему анализа и визуализации результатов всех уровней экспериментов для эффективного изучения фазового поля.

## Основные компоненты

### 1. Система анализа данных (utils/analysis.py)
- Статистический анализ результатов
- Сравнение с теоретическими предсказаниями
- Вычисление метрик качества
- Экспорт результатов в различные форматы

### 2. Система визуализации (utils/visualization.py)
- 2D и 3D визуализация полей
- Анимации временной эволюции
- Интерактивные графики
- Экспорт в различные форматы

### 3. Система отчетности (utils/reporting.py)
- Автоматическая генерация отчетов
- Сводные таблицы результатов
- Сравнительный анализ
- Экспорт в PDF/HTML

## Структура системы

### 1. Анализ данных
```python
class DataAnalyzer:
    def __init__(self, data_format='json'):
        """Инициализация анализатора данных"""
        
    def load_experiment_data(self, experiment_path):
        """Загрузка данных эксперимента"""
        
    def compute_statistics(self, data):
        """Вычисление статистики"""
        
    def compare_with_theory(self, data, theory_params):
        """Сравнение с теорией"""
        
    def export_results(self, results, format='json'):
        """Экспорт результатов"""
```

### 2. Визуализация
```python
class Visualizer:
    def __init__(self, backend='matplotlib'):
        """Инициализация визуализатора"""
        
    def plot_field_2d(self, field, title, save_path=None):
        """2D визуализация поля"""
        
    def plot_field_3d(self, field, title, save_path=None):
        """3D визуализация поля"""
        
    def create_animation(self, time_series, title, save_path=None):
        """Создание анимации"""
        
    def plot_comparison(self, data1, data2, title, save_path=None):
        """Сравнительная визуализация"""
```

### 3. Отчетность
```python
class ReportGenerator:
    def __init__(self, template_path=None):
        """Инициализация генератора отчетов"""
        
    def generate_experiment_report(self, experiment_data):
        """Генерация отчета по эксперименту"""
        
    def generate_summary_report(self, all_experiments):
        """Генерация сводного отчета"""
        
    def export_to_pdf(self, report, output_path):
        """Экспорт в PDF"""
        
    def export_to_html(self, report, output_path):
        """Экспорт в HTML"""
```

## Алгоритмы анализа

### 1. Статистический анализ
```python
def compute_field_statistics(field):
    """Вычисление статистики поля"""
    stats = {
        'mean': np.mean(field),
        'std': np.std(field),
        'min': np.min(field),
        'max': np.max(field),
        'skewness': scipy.stats.skew(field.flatten()),
        'kurtosis': scipy.stats.kurtosis(field.flatten())
    }
    
    # Спектральный анализ
    fft_field = np.fft.fftn(field)
    power_spectrum = np.abs(fft_field)**2
    
    stats['spectral_mean'] = np.mean(power_spectrum)
    stats['spectral_std'] = np.std(power_spectrum)
    
    return stats
```

### 2. Сравнение с теорией
```python
def compare_with_theory(numerical_data, theory_params):
    """Сравнение численных результатов с теорией"""
    # Вычисление теоретических предсказаний
    theory_predictions = compute_theoretical_predictions(theory_params)
    
    # Сравнение
    comparison = {
        'error_absolute': np.abs(numerical_data - theory_predictions),
        'error_relative': np.abs(numerical_data - theory_predictions) / np.abs(theory_predictions),
        'correlation': np.corrcoef(numerical_data.flatten(), theory_predictions.flatten())[0, 1],
        'r_squared': compute_r_squared(numerical_data, theory_predictions)
    }
    
    return comparison
```

### 3. Анализ качества
```python
def analyze_quality_metrics(experiment_results):
    """Анализ метрик качества эксперимента"""
    quality_metrics = {}
    
    for level, results in experiment_results.items():
        level_metrics = {
            'convergence': check_convergence(results),
            'stability': check_stability(results),
            'accuracy': check_accuracy(results),
            'completeness': check_completeness(results)
        }
        
        quality_metrics[level] = level_metrics
    
    return quality_metrics
```

## Визуализация

### 1. 2D визуализация полей
```python
def plot_field_2d(field, title, save_path=None):
    """2D визуализация поля"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Визуализация амплитуды
    im1 = ax.imshow(np.abs(field), cmap='viridis', origin='lower')
    ax.set_title(f'{title} - Amplitude')
    plt.colorbar(im1, ax=ax)
    
    # Визуализация фазы
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im2 = ax2.imshow(np.angle(field), cmap='hsv', origin='lower')
    ax2.set_title(f'{title} - Phase')
    plt.colorbar(im2, ax=ax2)
    
    if save_path:
        fig.savefig(f'{save_path}_amplitude.png', dpi=300, bbox_inches='tight')
        fig2.savefig(f'{save_path}_phase.png', dpi=300, bbox_inches='tight')
    
    plt.show()
```

### 2. 3D визуализация
```python
def plot_field_3d(field, title, save_path=None):
    """3D визуализация поля"""
    fig = plt.figure(figsize=(12, 10))
    
    # Создание 3D сетки
    x, y, z = np.meshgrid(np.linspace(0, 1, field.shape[0]),
                         np.linspace(0, 1, field.shape[1]),
                         np.linspace(0, 1, field.shape[2]))
    
    # Визуализация изоповерхностей
    ax = fig.add_subplot(111, projection='3d')
    
    # Изоповерхность для амплитуды
    threshold = np.max(np.abs(field)) * 0.5
    ax.plot_surface(x[:, :, field.shape[2]//2], 
                   y[:, :, field.shape[2]//2], 
                   np.abs(field[:, :, field.shape[2]//2]),
                   cmap='viridis', alpha=0.7)
    
    ax.set_title(f'{title} - 3D Visualization')
    
    if save_path:
        fig.savefig(f'{save_path}_3d.png', dpi=300, bbox_inches='tight')
    
    plt.show()
```

### 3. Анимации
```python
def create_animation(time_series, title, save_path=None):
    """Создание анимации временной эволюции"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame):
        ax.clear()
        im = ax.imshow(time_series[frame], cmap='viridis', origin='lower')
        ax.set_title(f'{title} - Frame {frame}')
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(time_series),
                                 interval=100, blit=True)
    
    if save_path:
        anim.save(f'{save_path}.gif', writer='pillow', fps=10)
    
    plt.show()
```

## Система отчетности

### 1. Генерация отчетов
```python
def generate_experiment_report(experiment_data):
    """Генерация отчета по эксперименту"""
    report = {
        'experiment_info': {
            'name': experiment_data['name'],
            'level': experiment_data['level'],
            'timestamp': experiment_data['timestamp'],
            'parameters': experiment_data['parameters']
        },
        'results': {
            'statistics': compute_field_statistics(experiment_data['field']),
            'quality_metrics': analyze_quality_metrics(experiment_data),
            'comparison_with_theory': compare_with_theory(experiment_data['field'], 
                                                        experiment_data['theory_params'])
        },
        'visualizations': {
            'field_2d': f"{experiment_data['name']}_field_2d.png",
            'field_3d': f"{experiment_data['name']}_field_3d.png",
            'animation': f"{experiment_data['name']}_animation.gif"
        }
    }
    
    return report
```

### 2. Сводный отчет
```python
def generate_summary_report(all_experiments):
    """Генерация сводного отчета по всем экспериментам"""
    summary = {
        'total_experiments': len(all_experiments),
        'levels_completed': list(set(exp['level'] for exp in all_experiments)),
        'overall_statistics': compute_overall_statistics(all_experiments),
        'level_summaries': {}
    }
    
    # Сводка по уровням
    for level in summary['levels_completed']:
        level_experiments = [exp for exp in all_experiments if exp['level'] == level]
        summary['level_summaries'][level] = {
            'count': len(level_experiments),
            'success_rate': compute_success_rate(level_experiments),
            'average_quality': compute_average_quality(level_experiments)
        }
    
    return summary
```

## Конфигурация системы

### 1. Конфигурация анализа (configs/analysis_config.json)
```json
{
    "statistics": {
        "compute_basic": true,
        "compute_spectral": true,
        "compute_correlations": true
    },
    "visualization": {
        "backend": "matplotlib",
        "dpi": 300,
        "format": "png",
        "animation_fps": 10
    },
    "reporting": {
        "template": "default",
        "output_format": ["pdf", "html"],
        "include_visualizations": true
    }
}
```

## Критерии готовности
- [ ] Реализована система анализа данных
- [ ] Реализована система визуализации
- [ ] Реализована система отчетности
- [ ] Все алгоритмы анализа работают корректно
- [ ] Визуализация создает качественные графики
- [ ] Отчеты генерируются автоматически
- [ ] Конфигурация системы настроена
- [ ] Документация написана
- [ ] Примеры использования созданы

## Следующий шаг
Step 12: Реализация автоматизированной системы тестирования и отчетности
