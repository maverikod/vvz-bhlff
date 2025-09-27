# Step 11: Детализированная спецификация системы анализа и визуализации

## Обзор

На основе анализа технического задания и теории 7D фазового поля, система анализа и визуализации должна обеспечить комплексный анализ результатов всех уровней экспериментов (A-G) с акцентом на физические метрики, спектральный анализ и сравнение с теоретическими предсказаниями.

## 1. Архитектура системы

### 1.1. Структура модулей

```
src/bhlff/utils/
├── analysis/                    # Анализ данных
│   ├── __init__.py
│   ├── statistics.py           # Статистический анализ
│   ├── spectral.py             # Спектральный анализ
│   ├── radial.py               # Радиальный анализ
│   ├── quality.py              # Метрики качества
│   ├── comparison.py           # Сравнение с теорией
│   └── metrics.py              # Вычисление метрик
├── visualization/              # Визуализация
│   ├── __init__.py
│   ├── plots.py                # 2D/3D графики
│   ├── animations.py           # Анимации
│   ├── interactive.py          # Интерактивные графики
│   ├── export.py               # Экспорт графиков
│   └── styles.py               # Стили и темы
├── reporting/                  # Отчетность
│   ├── __init__.py
│   ├── generator.py            # Генерация отчетов
│   ├── templates.py            # Шаблоны отчетов
│   ├── export.py               # Экспорт в PDF/HTML
│   └── summary.py              # Сводные отчеты
└── config/                     # Конфигурация
    ├── __init__.py
    ├── loader.py               # Загрузка конфигураций
    ├── validator.py            # Валидация параметров
    └── defaults.py             # Значения по умолчанию
```

### 1.2. Основные классы

#### DataAnalyzer
```python
class DataAnalyzer:
    """
    Comprehensive data analysis system for 7D phase field experiments.
    
    Physical Meaning:
        Analyzes experimental results from all levels (A-G) of the 7D phase
        field theory, providing statistical, spectral, and quality metrics
        for comparison with theoretical predictions.
        
    Mathematical Foundation:
        Implements analysis algorithms for:
        - Radial profiles A(r) and their derivatives
        - Spectral analysis in k-space
        - Quality metrics (convergence, stability, accuracy)
        - Comparison with theoretical predictions
    """
```

#### Visualizer
```python
class Visualizer:
    """
    Advanced visualization system for phase field data.
    
    Physical Meaning:
        Creates comprehensive visualizations of phase field configurations,
        including 2D/3D field plots, animations of temporal evolution,
        and interactive analysis tools.
        
    Mathematical Foundation:
        Implements visualization of:
        - Field amplitude and phase distributions
        - Radial profiles and spectral data
        - Temporal evolution and animations
        - Comparative analysis plots
    """
```

#### ReportGenerator
```python
class ReportGenerator:
    """
    Automated report generation system.
    
    Physical Meaning:
        Generates comprehensive reports combining analysis results,
        visualizations, and theoretical comparisons for all experiment
        levels, providing both detailed and summary views.
        
    Mathematical Foundation:
        Integrates statistical analysis, quality metrics, and theoretical
        predictions into structured reports with proper documentation
        of physical meaning and mathematical foundations.
    """
```

## 2. Детальные требования к анализу

### 2.1. Статистический анализ (statistics.py)

#### Основные метрики
```python
def compute_field_statistics(field: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive field statistics.
    
    Physical Meaning:
        Calculates statistical properties of the phase field configuration,
        including moments, spectral properties, and distribution characteristics.
        
    Mathematical Foundation:
        - Basic moments: mean, std, skewness, kurtosis
        - Spectral statistics: power spectrum analysis
        - Distribution analysis: histogram and fitting
        
    Returns:
        Dict containing all statistical metrics
    """
```

#### Радиальный анализ
```python
def compute_radial_profile(field: np.ndarray, center: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    """
    Compute radial profile A(r) and derivatives.
    
    Physical Meaning:
        Analyzes the radial distribution of field amplitude around the defect
        center, providing the fundamental A(r) profile and its derivatives
        for comparison with theoretical predictions.
        
    Mathematical Foundation:
        - Radial averaging: A(r) = mean(|a(x)| : |x-c| ∈ [r, r+dr])
        - Derivatives: A'(r), A''(r) using Savitzky-Golay filtering
        - Zone detection: core, tail, transition zones
        
    Returns:
        Dict with r, A(r), A'(r), A''(r), and zone boundaries
    """
```

### 2.2. Спектральный анализ (spectral.py)

#### k-пространственный анализ
```python
def analyze_spectral_properties(field: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analyze spectral properties in k-space.
    
    Physical Meaning:
        Computes power spectrum and spectral characteristics of the phase
        field, essential for understanding the frequency content and
        comparing with theoretical predictions.
        
    Mathematical Foundation:
        - Power spectrum: P(k) = |â(k)|²
        - Shell averaging: P_shell(k) = mean(P(k) : |k| ∈ [k, k+dk])
        - Spectral slope analysis for power law tails
        
    Returns:
        Dict with k values, power spectrum, and spectral metrics
    """
```

#### Анализ мод
```python
def analyze_mode_structure(field: np.ndarray, domain: Domain) -> Dict[str, Any]:
    """
    Analyze mode structure and resonances.
    
    Physical Meaning:
        Identifies resonant modes and their properties, including frequencies,
        quality factors, and spatial distributions.
        
    Mathematical Foundation:
        - Peak finding in power spectrum
        - Quality factor estimation: Q = ω/Δω
        - Mode spatial structure analysis
        
    Returns:
        Dict with mode frequencies, Q-factors, and spatial patterns
    """
```

### 2.3. Метрики качества (quality.py)

#### Критерии сходимости
```python
def check_convergence(results: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check convergence criteria for experiments.
    
    Physical Meaning:
        Validates numerical convergence of the solution, ensuring that
        the results are reliable and not affected by numerical artifacts.
        
    Mathematical Foundation:
        - Grid convergence: error reduction with mesh refinement
        - Iterative convergence: residual reduction
        - Spectral convergence: high-frequency error analysis
        
    Returns:
        Dict with convergence flags and error estimates
    """
```

#### Анализ устойчивости
```python
def check_stability(results: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check numerical stability of the solution.
    
    Physical Meaning:
        Ensures that the numerical solution is stable and does not
        exhibit spurious oscillations or instabilities.
        
    Mathematical Foundation:
        - Energy conservation: |E(t) - E(0)|/E(0) < threshold
        - Passivity: Re(Y(ω)) ≥ 0 for all frequencies
        - Stability analysis: eigenvalue analysis of discretized operator
        
    Returns:
        Dict with stability flags and stability metrics
    """
```

### 2.4. Сравнение с теорией (comparison.py)

#### Теоретические предсказания
```python
def compute_theoretical_predictions(params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Compute theoretical predictions for comparison.
    
    Physical Meaning:
        Calculates theoretical predictions based on the 7D phase field
        theory, including power law tails, mode frequencies, and
        topological properties.
        
    Mathematical Foundation:
        - Power law tail: A(r) ~ r^(2β-3) for λ=0
        - Mode frequencies: ω_n from boundary conditions
        - Topological charge: q from phase winding
        
    Returns:
        Dict with theoretical predictions for all relevant quantities
    """
```

#### Метрики сравнения
```python
def compare_with_theory(numerical: np.ndarray, theoretical: np.ndarray) -> Dict[str, float]:
    """
    Compare numerical results with theoretical predictions.
    
    Physical Meaning:
        Quantifies the agreement between numerical simulations and
        theoretical predictions, providing error metrics and correlation
        analysis.
        
    Mathematical Foundation:
        - Relative error: |num - theo|/|theo|
        - Correlation coefficient: corr(num, theo)
        - R-squared: 1 - SS_res/SS_tot
        
    Returns:
        Dict with comparison metrics and error estimates
    """
```

## 3. Детальные требования к визуализации

### 3.1. 2D/3D визуализация (plots.py)

#### Визуализация полей
```python
def plot_field_2d(field: np.ndarray, title: str, save_path: Optional[str] = None) -> None:
    """
    Create 2D field visualization.
    
    Physical Meaning:
        Visualizes the phase field configuration in 2D, showing both
        amplitude and phase distributions with proper physical interpretation.
        
    Mathematical Foundation:
        - Amplitude plot: |a(x,y)|
        - Phase plot: arg(a(x,y))
        - Combined visualization with proper color mapping
        
    Args:
        field: 2D complex field array
        title: Plot title with physical meaning
        save_path: Optional path for saving the plot
    """
```

#### 3D визуализация
```python
def plot_field_3d(field: np.ndarray, title: str, save_path: Optional[str] = None) -> None:
    """
    Create 3D field visualization.
    
    Physical Meaning:
        Visualizes the 3D phase field configuration using isosurfaces
        and volume rendering, showing the spatial structure of the field.
        
    Mathematical Foundation:
        - Isosurfaces: |a(x,y,z)| = const
        - Volume rendering with transparency
        - Cross-sections and projections
        
    Args:
        field: 3D complex field array
        title: Plot title with physical meaning
        save_path: Optional path for saving the plot
    """
```

#### 7D визуализация
```python
class Visualization7D:
    """
    Специализированные методы визуализации для 7D фазовых полей.
    
    Physical Meaning:
        Предоставляет методы для визуализации 7D фазовых полей,
        включая проекции, сечения и многомерные представления.
        
    Mathematical Foundation:
        - Проекции: P_3d(a_7d) = ∫∫∫ a_7d dψ dχ dζ
        - Сечения: S_3d(a_7d) = a_7d(ψ₀, χ₀, ζ₀)
        - Изоповерхности: |a_7d| = const в 7D пространстве
    """
    
    def __init__(self, field_7d: np.ndarray, domain: 'Domain7D'):
        """
        Инициализация 7D визуализатора.
        
        Args:
            field_7d: 7D фазовое поле
            domain: 7D область расчета
        """
        self.field_7d = field_7d
        self.domain = domain
        self._setup_visualization_tools()
    
    def _setup_visualization_tools(self) -> None:
        """
        Настройка инструментов визуализации.
        
        Physical Meaning:
            Инициализирует инструменты для различных типов
            7D визуализации.
        """
        self.projection_methods = {
            'spatial': self._spatial_projection,
            'internal': self._internal_projection,
            'mixed': self._mixed_projection
        }
        
        self.slice_methods = {
            'spatial_slice': self._spatial_slice,
            'internal_slice': self._internal_slice,
            'diagonal_slice': self._diagonal_slice
        }
    
    def create_7d_visualization(self, visualization_type: str, 
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание 7D визуализации.
        
        Physical Meaning:
            Создает визуализацию 7D фазового поля в зависимости
            от выбранного типа визуализации.
            
        Args:
            visualization_type: Тип визуализации
            params: Параметры визуализации
            
        Returns:
            Словарь с результатами визуализации
        """
        if visualization_type == 'projection':
            return self._create_projection_visualization(params)
        elif visualization_type == 'slice':
            return self._create_slice_visualization(params)
        elif visualization_type == 'isosurface':
            return self._create_isosurface_visualization(params)
        elif visualization_type == 'streamline':
            return self._create_streamline_visualization(params)
        else:
            raise ValueError(f"Неизвестный тип визуализации: {visualization_type}")
    
    def _create_projection_visualization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание проекционной визуализации.
        
        Physical Meaning:
            Создает проекции 7D поля на 3D подпространства
            для визуализации структуры поля.
        """
        projection_type = params.get('type', 'spatial')
        projection_method = self.projection_methods[projection_type]
        
        # Выполнение проекции
        projected_field = projection_method(params)
        
        # Создание визуализации
        visualization = self._plot_3d_field(projected_field, 
                                          f"7D Projection - {projection_type}")
        
        return {
            'projected_field': projected_field,
            'visualization': visualization,
            'projection_type': projection_type,
            'parameters': params
        }
    
    def _spatial_projection(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Проекция на 3D пространственные координаты.
        
        Physical Meaning:
            Проецирует 7D поле на 3D пространственные координаты
            (x, y, z), интегрируя по внутренним координатам.
            
        Mathematical Foundation:
            P_spatial(x,y,z) = ∫∫∫ a(x,y,z,ψ,χ,ζ) dψ dχ dζ
        """
        # Интегрирование по внутренним координатам (ψ, χ, ζ)
        spatial_projection = np.sum(self.field_7d, axis=(4, 5, 6))
        
        return spatial_projection
    
    def _internal_projection(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Проекция на 3D внутренние координаты.
        
        Physical Meaning:
            Проецирует 7D поле на 3D внутренние координаты
            (ψ, χ, ζ), интегрируя по пространственным координатам.
            
        Mathematical Foundation:
            P_internal(ψ,χ,ζ) = ∫∫∫ a(x,y,z,ψ,χ,ζ) dx dy dz
        """
        # Интегрирование по пространственным координатам (x, y, z)
        internal_projection = np.sum(self.field_7d, axis=(1, 2, 3))
        
        return internal_projection
    
    def _mixed_projection(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Смешанная проекция.
        
        Physical Meaning:
            Создает смешанную проекцию, комбинируя пространственные
            и внутренние координаты.
        """
        # Выбор координат для проекции
        coord_indices = params.get('coord_indices', [0, 1, 4])  # x, y, ψ
        
        # Создание проекции по выбранным координатам
        mixed_projection = np.sum(self.field_7d, 
                                axis=tuple(i for i in range(7) if i not in coord_indices))
        
        return mixed_projection
    
    def _create_slice_visualization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание визуализации сечений.
        
        Physical Meaning:
            Создает сечения 7D поля по различным координатам
            для изучения локальной структуры.
        """
        slice_type = params.get('type', 'spatial_slice')
        slice_method = self.slice_methods[slice_type]
        
        # Выполнение сечения
        sliced_field = slice_method(params)
        
        # Создание визуализации
        visualization = self._plot_3d_field(sliced_field, 
                                          f"7D Slice - {slice_type}")
        
        return {
            'sliced_field': sliced_field,
            'visualization': visualization,
            'slice_type': slice_type,
            'parameters': params
        }
    
    def _spatial_slice(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Сечение по пространственным координатам.
        
        Physical Meaning:
            Создает сечение 7D поля по пространственным
            координатам при фиксированных внутренних координатах.
        """
        # Фиксированные внутренние координаты
        psi_fixed = params.get('psi', 0)
        chi_fixed = params.get('chi', 0)
        zeta_fixed = params.get('zeta', 0)
        
        # Создание сечения
        spatial_slice = self.field_7d[:, :, :, psi_fixed, chi_fixed, zeta_fixed]
        
        return spatial_slice
    
    def _internal_slice(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Сечение по внутренним координатам.
        
        Physical Meaning:
            Создает сечение 7D поля по внутренним
            координатам при фиксированных пространственных координатах.
        """
        # Фиксированные пространственные координаты
        x_fixed = params.get('x', 0)
        y_fixed = params.get('y', 0)
        z_fixed = params.get('z', 0)
        
        # Создание сечения
        internal_slice = self.field_7d[x_fixed, y_fixed, z_fixed, :, :, :]
        
        return internal_slice
    
    def _diagonal_slice(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Диагональное сечение.
        
        Physical Meaning:
            Создает диагональное сечение 7D поля,
            комбинируя пространственные и внутренние координаты.
        """
        # Параметры диагонального сечения
        diagonal_params = params.get('diagonal_params', {})
        
        # Создание диагонального сечения
        # Упрощенная реализация - в реальности нужен более сложный алгоритм
        diagonal_slice = np.zeros((self.domain.N, self.domain.N, self.domain.N))
        
        for i in range(self.domain.N):
            for j in range(self.domain.N):
                for k in range(self.domain.N):
                    # Диагональное индексирование
                    idx_7d = (i, j, k, i, j, k, 0)
                    diagonal_slice[i, j, k] = self.field_7d[idx_7d]
        
        return diagonal_slice
    
    def _create_isosurface_visualization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание визуализации изоповерхностей.
        
        Physical Meaning:
            Создает изоповерхности в 7D пространстве,
            показывающие области постоянной амплитуды поля.
        """
        # Пороговое значение для изоповерхности
        threshold = params.get('threshold', 0.5)
        
        # Вычисление амплитуды поля
        field_amplitude = np.abs(self.field_7d)
        
        # Создание изоповерхности
        isosurface = field_amplitude > threshold
        
        # Проекция на 3D для визуализации
        isosurface_3d = self._spatial_projection({'type': 'spatial'})
        isosurface_3d = np.abs(isosurface_3d) > threshold
        
        # Создание визуализации
        visualization = self._plot_3d_isosurface(isosurface_3d, 
                                               f"7D Isosurface - threshold={threshold}")
        
        return {
            'isosurface': isosurface,
            'isosurface_3d': isosurface_3d,
            'visualization': visualization,
            'threshold': threshold
        }
    
    def _create_streamline_visualization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание визуализации линий потока.
        
        Physical Meaning:
            Создает линии потока для 7D фазового поля,
            показывающие направление градиента фазы.
        """
        # Вычисление градиента фазы
        phase_gradient = self._compute_phase_gradient_7d()
        
        # Создание линий потока
        streamlines = self._trace_streamlines_7d(phase_gradient, params)
        
        # Проекция на 3D для визуализации
        streamlines_3d = self._project_streamlines_3d(streamlines)
        
        # Создание визуализации
        visualization = self._plot_3d_streamlines(streamlines_3d, 
                                                "7D Phase Streamlines")
        
        return {
            'streamlines': streamlines,
            'streamlines_3d': streamlines_3d,
            'visualization': visualization,
            'phase_gradient': phase_gradient
        }
    
    def _compute_phase_gradient_7d(self) -> np.ndarray:
        """
        Вычисление градиента фазы в 7D.
        
        Physical Meaning:
            Вычисляет градиент фазы φ(x,y,z,ψ,χ,ζ) в 7D пространстве.
        """
        # Вычисление фазы
        phase = np.angle(self.field_7d)
        
        # Вычисление градиента по всем координатам
        phase_gradient = np.gradient(phase)
        
        return phase_gradient
    
    def _trace_streamlines_7d(self, phase_gradient: np.ndarray, 
                            params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Трассировка линий потока в 7D.
        
        Physical Meaning:
            Трассирует линии потока в 7D пространстве,
            следуя направлению градиента фазы.
        """
        # Параметры трассировки
        n_streamlines = params.get('n_streamlines', 100)
        max_length = params.get('max_length', 1000)
        
        # Начальные точки для линий потока
        start_points = self._generate_start_points(n_streamlines)
        
        # Трассировка линий потока
        streamlines = []
        for start_point in start_points:
            streamline = self._trace_single_streamline(start_point, phase_gradient, max_length)
            streamlines.append(streamline)
        
        return streamlines
    
    def _generate_start_points(self, n_points: int) -> List[np.ndarray]:
        """
        Генерация начальных точек для линий потока.
        
        Physical Meaning:
            Генерирует случайные начальные точки в 7D пространстве
            для трассировки линий потока.
        """
        start_points = []
        for _ in range(n_points):
            # Случайная точка в 7D пространстве
            point = np.random.rand(7) * self.domain.N
            start_points.append(point)
        
        return start_points
    
    def _trace_single_streamline(self, start_point: np.ndarray, 
                               phase_gradient: np.ndarray, 
                               max_length: int) -> np.ndarray:
        """
        Трассировка одной линии потока.
        
        Physical Meaning:
            Трассирует одну линию потока, начиная с заданной точки
            и следуя направлению градиента фазы.
        """
        streamline = [start_point.copy()]
        current_point = start_point.copy()
        
        for _ in range(max_length):
            # Вычисление направления в текущей точке
            direction = self._get_direction_at_point(current_point, phase_gradient)
            
            # Обновление позиции
            current_point += direction * 0.1  # Шаг интегрирования
            
            # Проверка границ
            if self._is_out_of_bounds(current_point):
                break
            
            streamline.append(current_point.copy())
        
        return np.array(streamline)
    
    def _get_direction_at_point(self, point: np.ndarray, 
                              phase_gradient: np.ndarray) -> np.ndarray:
        """
        Получение направления в заданной точке.
        
        Physical Meaning:
            Вычисляет направление градиента фазы в заданной точке
            7D пространства.
        """
        # Интерполяция градиента в точке
        direction = np.zeros(7)
        for i in range(7):
            # Упрощенная интерполяция - в реальности нужна более точная
            idx = int(point[i]) % phase_gradient[i].shape[0]
            direction[i] = phase_gradient[i].flat[idx]
        
        # Нормализация
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        
        return direction
    
    def _is_out_of_bounds(self, point: np.ndarray) -> bool:
        """
        Проверка выхода за границы области.
        
        Physical Meaning:
            Проверяет, находится ли точка в пределах
            расчетной области.
        """
        return np.any(point < 0) or np.any(point >= self.domain.N)
    
    def _project_streamlines_3d(self, streamlines: List[np.ndarray]) -> List[np.ndarray]:
        """
        Проекция линий потока на 3D.
        
        Physical Meaning:
            Проецирует 7D линии потока на 3D пространство
            для визуализации.
        """
        streamlines_3d = []
        for streamline in streamlines:
            # Проекция на первые 3 координаты (x, y, z)
            streamline_3d = streamline[:, :3]
            streamlines_3d.append(streamline_3d)
        
        return streamlines_3d
    
    def _plot_3d_field(self, field_3d: np.ndarray, title: str) -> None:
        """
        Построение 3D поля.
        
        Physical Meaning:
            Создает 3D визуализацию поля с использованием
            изоповерхностей и объемного рендеринга.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Создание сетки
        x, y, z = np.meshgrid(np.linspace(0, 1, field_3d.shape[0]),
                             np.linspace(0, 1, field_3d.shape[1]),
                             np.linspace(0, 1, field_3d.shape[2]))
        
        # Визуализация изоповерхности
        threshold = np.max(np.abs(field_3d)) * 0.5
        ax.plot_surface(x[:, :, field_3d.shape[2]//2], 
                       y[:, :, field_3d.shape[2]//2], 
                       np.abs(field_3d[:, :, field_3d.shape[2]//2]),
                       cmap='viridis', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()
    
    def _plot_3d_isosurface(self, isosurface: np.ndarray, title: str) -> None:
        """
        Построение 3D изоповерхности.
        
        Physical Meaning:
            Создает 3D визуализацию изоповерхности
            с использованием объемного рендеринга.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Создание сетки
        x, y, z = np.meshgrid(np.linspace(0, 1, isosurface.shape[0]),
                             np.linspace(0, 1, isosurface.shape[1]),
                             np.linspace(0, 1, isosurface.shape[2]))
        
        # Визуализация изоповерхности
        ax.voxels(isosurface, alpha=0.7, color='blue')
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()
    
    def _plot_3d_streamlines(self, streamlines: List[np.ndarray], title: str) -> None:
        """
        Построение 3D линий потока.
        
        Physical Meaning:
            Создает 3D визуализацию линий потока
            с использованием трассировки.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Построение линий потока
        for streamline in streamlines:
            if len(streamline) > 1:
                ax.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2], 
                       alpha=0.7, linewidth=1)
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()
```

### 3.2. Анимации (animations.py)

#### Временная эволюция
```python
def create_temporal_animation(time_series: List[np.ndarray], title: str, save_path: Optional[str] = None) -> None:
    """
    Create animation of temporal evolution.
    
    Physical Meaning:
        Shows the temporal evolution of the phase field, including
        mode dynamics, defect motion, and field oscillations.
        
    Mathematical Foundation:
        - Frame-by-frame visualization of a(x,t)
        - Proper time scaling and interpolation
        - Overlay of key physical quantities
        
    Args:
        time_series: List of field snapshots at different times
        title: Animation title with physical meaning
        save_path: Optional path for saving the animation
    """
```

### 3.3. Интерактивные графики (interactive.py)

#### Интерактивный анализ
```python
def create_interactive_plot(data: Dict[str, np.ndarray]) -> None:
    """
    Create interactive analysis plot.
    
    Physical Meaning:
        Provides interactive tools for exploring phase field data,
        including zooming, panning, and parameter adjustment.
        
    Mathematical Foundation:
        - Interactive widgets for parameter control
        - Real-time updates of visualizations
        - Export capabilities for analysis results
        
    Args:
        data: Dictionary containing field data and analysis results
    """
```

## 4. Детальные требования к отчетности

### 4.1. Генерация отчетов (generator.py)

#### Отчет по эксперименту
```python
def generate_experiment_report(experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive experiment report.
    
    Physical Meaning:
        Creates detailed report for a single experiment, including
        all analysis results, visualizations, and theoretical comparisons.
        
    Mathematical Foundation:
        - Statistical analysis results
        - Quality metrics and convergence data
        - Theoretical comparison and error analysis
        - Visualization references and captions
        
    Args:
        experiment_data: Complete experiment data including results
        
    Returns:
        Structured report dictionary
    """
```

#### Сводный отчет
```python
def generate_summary_report(all_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary report for all experiments.
    
    Physical Meaning:
        Creates comprehensive summary of all experiment levels,
        providing overview of results and cross-level comparisons.
        
    Mathematical Foundation:
        - Aggregated statistics across all levels
        - Success rates and quality metrics
        - Comparative analysis between levels
        - Overall project status and recommendations
        
    Args:
        all_experiments: List of all experiment results
        
    Returns:
        Summary report dictionary
    """
```

### 4.2. Шаблоны отчетов (templates.py)

#### HTML шаблон
```python
HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>7D Phase Field Analysis Report</title>
    <style>
        /* Professional styling for scientific reports */
        body { font-family: 'Times New Roman', serif; }
        .header { background-color: #2c3e50; color: white; padding: 20px; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #ecf0f1; }
        .plot { text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>7D Phase Field Theory Analysis Report</h1>
        <p>Generated: {{timestamp}}</p>
    </div>
    
    <div class="section">
        <h2>Experiment Overview</h2>
        <p><strong>Level:</strong> {{level}}</p>
        <p><strong>Parameters:</strong> {{parameters}}</p>
        <p><strong>Status:</strong> {{status}}</p>
    </div>
    
    <div class="section">
        <h2>Analysis Results</h2>
        <div class="metric">
            <h3>Statistical Metrics</h3>
            <p>Mean: {{stats.mean}}</p>
            <p>Std: {{stats.std}}</p>
            <p>Skewness: {{stats.skewness}}</p>
        </div>
        <!-- Additional metrics and visualizations -->
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        <div class="plot">
            <img src="{{field_2d_plot}}" alt="2D Field Visualization">
            <p>2D Field Visualization</p>
        </div>
        <!-- Additional plots -->
    </div>
</body>
</html>
"""
```

#### PDF шаблон
```python
PDF_REPORT_TEMPLATE = """
# 7D Phase Field Theory Analysis Report

## Experiment Information
- **Level:** {{level}}
- **Timestamp:** {{timestamp}}
- **Parameters:** {{parameters}}
- **Status:** {{status}}

## Analysis Results

### Statistical Analysis
- Mean: {{stats.mean}}
- Standard Deviation: {{stats.std}}
- Skewness: {{stats.skewness}}
- Kurtosis: {{stats.kurtosis}}

### Quality Metrics
- Convergence: {{quality.convergence}}
- Stability: {{quality.stability}}
- Accuracy: {{quality.accuracy}}

### Theoretical Comparison
- Relative Error: {{comparison.relative_error}}
- Correlation: {{comparison.correlation}}
- R-squared: {{comparison.r_squared}}

## Visualizations
[Include plots and figures with proper captions]

## Conclusions
[Summary of results and recommendations]
"""
```

## 5. Конфигурация системы

### 5.1. Конфигурация анализа (configs/analysis_config.json)
```json
{
    "statistics": {
        "compute_basic": true,
        "compute_spectral": true,
        "compute_correlations": true,
        "radial_analysis": {
            "rmin_cells": 4,
            "rmax_fraction_of_L": 0.25,
            "smoothing": "savitzky_golay",
            "smoothing_window": 5
        }
    },
    "quality_metrics": {
        "convergence_threshold": 1e-6,
        "stability_threshold": 1e-3,
        "accuracy_threshold": 0.05,
        "energy_balance_threshold": 0.03
    },
    "theoretical_comparison": {
        "power_law_fit": {
            "min_decades": 1.5,
            "robust_fitting": true,
            "outlier_threshold": 3.0
        },
        "mode_analysis": {
            "peak_prominence": 10.0,
            "quality_factor_min": 1.0
        }
    },
    "visualization": {
        "backend": "matplotlib",
        "style": "scientific",
        "dpi": 300,
        "format": "png",
        "animation_fps": 10,
        "color_maps": {
            "amplitude": "viridis",
            "phase": "hsv",
            "spectrum": "plasma"
        }
    },
    "reporting": {
        "template": "scientific",
        "output_format": ["pdf", "html"],
        "include_visualizations": true,
        "include_raw_data": false,
        "language": "english"
    }
}
```

### 5.2. Конфигурация визуализации (configs/visualization_config.json)
```json
{
    "plots": {
        "figure_size": [12, 8],
        "font_size": 12,
        "line_width": 2,
        "marker_size": 6,
        "grid": true,
        "legend": true
    },
    "animations": {
        "fps": 10,
        "bitrate": 1800,
        "codec": "h264",
        "format": "mp4"
    },
    "export": {
        "formats": ["png", "pdf", "svg"],
        "dpi": 300,
        "bbox_inches": "tight",
        "transparent": false
    },
    "interactive": {
        "backend": "plotly",
        "theme": "plotly_white",
        "responsive": true
    }
}
```

## 6. Алгоритмы анализа

### 6.1. Радиальный анализ
```python
def compute_radial_analysis(field: np.ndarray, center: Tuple[int, int, int], 
                          domain: Domain) -> Dict[str, np.ndarray]:
    """
    Comprehensive radial analysis of phase field.
    
    Physical Meaning:
        Analyzes the radial distribution of field amplitude around the defect,
        providing the fundamental A(r) profile and its derivatives for
        comparison with theoretical predictions.
        
    Mathematical Foundation:
        1. Radial averaging: A(r) = mean(|a(x)| : |x-c| ∈ [r, r+dr])
        2. Smoothing: Apply Savitzky-Golay filter for derivatives
        3. Zone detection: Identify core, tail, and transition zones
        4. Power law fitting: A(r) ~ r^p in tail region
        
    Args:
        field: 3D complex field array
        center: Center coordinates for radial analysis
        domain: Domain configuration
        
    Returns:
        Dict with r, A(r), A'(r), A''(r), zones, and fit parameters
    """
    # Implementation details...
```

### 6.2. Спектральный анализ
```python
def compute_spectral_analysis(field: np.ndarray, domain: Domain) -> Dict[str, np.ndarray]:
    """
    Comprehensive spectral analysis in k-space.
    
    Physical Meaning:
        Analyzes the frequency content of the phase field, identifying
        resonant modes and their properties for comparison with theoretical
        predictions.
        
    Mathematical Foundation:
        1. FFT: â(k) = FFT(a(x))
        2. Power spectrum: P(k) = |â(k)|²
        3. Shell averaging: P_shell(k) = mean(P(k) : |k| ∈ [k, k+dk])
        4. Peak finding: Identify resonant frequencies
        5. Quality factors: Q = ω/Δω
        
    Args:
        field: 3D complex field array
        domain: Domain configuration
        
    Returns:
        Dict with k values, power spectrum, peaks, and Q-factors
    """
    # Implementation details...
```

### 6.3. Анализ качества
```python
def compute_quality_metrics(results: Dict[str, Any]) -> Dict[str, bool]:
    """
    Compute comprehensive quality metrics.
    
    Physical Meaning:
        Evaluates the quality of numerical results, including convergence,
        stability, and accuracy metrics.
        
    Mathematical Foundation:
        1. Convergence: Error reduction with mesh refinement
        2. Stability: Energy conservation and passivity
        3. Accuracy: Comparison with analytical solutions
        4. Completeness: Coverage of all required metrics
        
    Args:
        results: Dictionary containing all experiment results
        
    Returns:
        Dict with quality flags and error estimates
    """
    # Implementation details...
```

## 7. Критерии готовности

### 7.1. Функциональные требования
- [ ] Реализована система статистического анализа
- [ ] Реализована система спектрального анализа
- [ ] Реализована система радиального анализа
- [ ] Реализована система анализа качества
- [ ] Реализована система сравнения с теорией
- [ ] Реализована система 2D/3D визуализации
- [ ] Реализована система анимаций
- [ ] Реализована система интерактивных графиков
- [ ] Реализована система генерации отчетов
- [ ] Реализована система экспорта в различные форматы

### 7.2. Качественные требования
- [ ] Все алгоритмы анализа работают корректно
- [ ] Визуализация создает качественные графики
- [ ] Отчеты генерируются автоматически
- [ ] Конфигурация системы настроена
- [ ] Документация написана на английском языке
- [ ] Примеры использования созданы
- [ ] Тесты покрывают все основные функции
- [ ] Производительность соответствует требованиям

### 7.3. Интеграционные требования
- [ ] Система интегрирована с существующими модулями
- [ ] Поддерживает все уровни экспериментов (A-G)
- [ ] Совместима с форматами данных проекта
- [ ] Поддерживает конфигурационные файлы
- [ ] Интегрирована с системой тестирования

## 8. Следующие шаги

После завершения Step 11 система анализа и визуализации будет готова для интеграции с автоматизированной системой тестирования (Step 12), обеспечивая полный цикл разработки и валидации 7D фазового поля.
