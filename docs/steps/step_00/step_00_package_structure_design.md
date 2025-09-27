# Step 00: Проектирование структуры пакетов

## Цель
Спроектировать оптимальную структуру пакетов на основе анализа объектной структуры всех шагов и ТЗ, выделив общие блоки и устранив дублирование.

## Принципы проектирования

### 1. **ВБП-центричность**
- ВБП (Высокочастотное Базовое Поле) - центральный каркас всей системы
- Все остальные компоненты работают через ВБП-интерфейсы

### 2. **Разделение ответственности**
- Каждый пакет имеет четко определенную ответственность
- Минимальные зависимости между пакетами
- Четкие интерфейсы между пакетами

### 3. **Иерархия наследования**
- Абстрактные базовые классы для общих интерфейсов
- Конкретные реализации в специализированных пакетах
- Полиморфизм через интерфейсы

### 4. **Композиция над наследованием**
- Использование композиции для объединения функциональности
- Агрегация для слабых связей
- Композиция для сильных связей

## Структура пакетов

### **1. `bhlff.core` - Ядро системы**

#### **1.1. `bhlff.core.bvp` - ВБП-каркас**
```python
# Центральный каркас системы
class BVPCore:
    """Ядро ВБП - центральный компонент системы"""
    
class QuenchDetector:
    """Детектор квенчей в ВБП"""
    
class BVPInterface:
    """Интерфейсы ВБП с другими компонентами"""
    
class BVPEnvelopeSolver:
    """Решатель уравнения огибающей ВБП"""
```

#### **1.2. `bhlff.core.domain` - Вычислительная область**
```python
class Domain:
    """Вычислительная область с сеткой"""
    
class Field:
    """Фазовое поле на области"""
    
class Grid:
    """Сетка для вычислений"""
    
class BoundaryConditions:
    """Граничные условия"""
```

#### **1.3. `bhlff.core.operators` - Операторы**
```python
class OperatorRiesz:
    """Фракционный оператор Рисса"""
    
class FractionalLaplacian:
    """Фракционный лапласиан"""
    
class MemoryKernel:
    """Ядро памяти для нелокальных операций"""
```

#### **1.4. `bhlff.core.fft` - FFT операции**
```python
class FFTBackend:
    """Бэкенд для FFT операций"""
    
class SpectralOperations:
    """Спектральные операции"""
    
class FFTPlan:
    """План FFT для оптимизации"""
```

#### **1.5. `bhlff.core.sources` - Источники**
```python
class Source(ABC):
    """Абстрактный базовый класс для источников"""
    
class BVPSource(Source):
    """ВБП-модуляционный источник"""
    
class QuenchSource(Source):
    """Источник квенчей"""
    
class HarmonicSource(Source):
    """Гармонический источник (для совместимости)"""
```

### **2. `bhlff.solvers` - Решатели**

#### **2.1. `bhlff.solvers.base` - Базовые решатели**
```python
class AbstractSolver(ABC):
    """Абстрактный базовый класс для решателей"""
    
class StationarySolver(AbstractSolver):
    """Стационарный решатель"""
    
class TimeSolver(AbstractSolver):
    """Временной решатель"""
```

#### **2.2. `bhlff.solvers.integrators` - Интеграторы**
```python
class TimeIntegrator(ABC):
    """Абстрактный базовый класс для интеграторов"""
    
class BVPModulationIntegrator(TimeIntegrator):
    """ВБП-модуляционный интегратор"""
    
class CrankNicolsonIntegrator(TimeIntegrator):
    """CN интегратор"""
    
class AdaptiveIntegrator(TimeIntegrator):
    """Адаптивный интегратор"""
```

#### **2.3. `bhlff.solvers.spectral` - Спектральные решатели**
```python
class FFTSolver3D:
    """3D FFT решатель"""
    
class SpectralSolver:
    """Спектральный решатель"""
    
class FrequencySweep:
    """Развертка по частотам"""
```

### **3. `bhlff.geometry` - Геометрия**

#### **3.1. `bhlff.geometry.layers` - Слои**
```python
class SphericalLayer:
    """Сферический слой"""
    
class LayerStack:
    """Стек слоев"""
    
class ReferenceBall:
    """Референсный шар"""
```

#### **3.2. `bhlff.geometry.boundaries` - Границы**
```python
class Boundary:
    """Абстрактная граница"""
    
class SphericalBoundary(Boundary):
    """Сферическая граница"""
    
class PeriodicBoundary(Boundary):
    """Периодическая граница"""
```

### **4. `bhlff.analysis` - Анализ**

#### **4.1. `bhlff.analysis.metrics` - Метрики**
```python
class RadialAverager:
    """Радиальное усреднение"""
    
class TailSlopeEstimator:
    """Оценка степенного хвоста"""
    
class PhaseWinding:
    """Вычисление обмоток фазы"""
    
class ZoneDetector:
    """Детектор зон (ядро/переход/хвост)"""
```

#### **4.2. `bhlff.analysis.envelope` - Анализ огибающей**
```python
class EnvelopeProxies:
    """Прокси огибающей"""
    
class FluxRadial:
    """Радиальный поток"""
    
class HotZonesMask:
    """Маска горячих зон"""
    
class ChiralityMetrics:
    """Метрики хиральности"""
```

#### **4.3. `bhlff.analysis.spectral` - Спектральный анализ**
```python
class SpectralAnalyzer:
    """Спектральный анализатор"""
    
class PeaksFinder:
    """Поиск пиков в спектре"""
    
class AdmittanceEstimator:
    """Оценка адмиттанса"""
```

### **5. `bhlff.transmission` - Передача (ABCD)**

#### **5.1. `bhlff.transmission.abcd` - ABCD матрицы**
```python
class LayerTransferMatrix:
    """Матрица передачи слоя"""
    
class TransferChain:
    """Цепочка передачи"""
    
class ABCDMatrix:
    """ABCD матрица"""
```

#### **5.2. `bhlff.transmission.impedance` - Импеданс**
```python
class ImpedanceCalculator:
    """Калькулятор импеданса"""
    
class AdmittanceCalculator:
    """Калькулятор адмиттанса"""
    
class ReflectionTransmission:
    """Коэффициенты отражения и передачи"""
```

### **6. `bhlff.windows` - Окна и каркас**

#### **6.1. `bhlff.windows.selection` - Селекция окон**
```python
class WindowSelector:
    """Селектор окон"""
    
class EMWindow:
    """EM окно"""
    
class StrongWindow:
    """Сильное окно"""
    
class WeakWindow:
    """Слабое окно"""
```

#### **6.2. `bhlff.windows.skeleton` - Каркас**
```python
class Skeleton:
    """Каркас системы"""
    
class OverlayManager:
    """Менеджер наложений"""
    
class CarcassBuilder:
    """Построитель каркаса"""
```

### **7. `bhlff.models` - Модели уровней**

#### **7.1. `bhlff.models.solitons` - Солитоны**
```python
class SolitonModel(ABC):
    """Абстрактная модель солитона"""
    
class BaryonSoliton(SolitonModel):
    """Барионный солитон"""
    
class SkyrmionSoliton(SolitonModel):
    """Скермионный солитон"""
```

#### **7.2. `bhlff.models.defects` - Дефекты**
```python
class DefectModel(ABC):
    """Абстрактная модель дефекта"""
    
class VortexDefect(DefectModel):
    """Вихревой дефект"""
    
class MultiDefectSystem(DefectModel):
    """Много-дефектная система"""
```

#### **7.3. `bhlff.models.levels` - Модели по уровням**
```python
# Уровень A
class LevelAModel:
    """Модель уровня A"""
    
# Уровень B
class LevelBModel:
    """Модель уровня B"""
    
# Уровень C
class LevelCModel:
    """Модель уровня C"""
    
# Уровень D
class LevelDModel:
    """Модель уровня D"""
    
# Уровень E
class LevelEModel:
    """Модель уровня E"""
    
# Уровень F
class LevelFModel:
    """Модель уровня F"""
    
# Уровень G
class LevelGModel:
    """Модель уровня G"""
```

### **8. `bhlff.dynamics` - Динамика**

#### **8.1. `bhlff.dynamics.traps` - Ловушки**
```python
class Trap:
    """Ловушка"""
    
class Drive:
    """Привод"""
    
class Kick:
    """Толчок"""
```

#### **8.2. `bhlff.dynamics.tracking` - Отслеживание**
```python
class Tracker:
    """Трекер движения"""
    
class MobilityGyro:
    """Мобильность-гироскоп"""
    
class MassEstimators:
    """Оценщики массы"""
```

#### **8.3. `bhlff.dynamics.collisions` - Столкновения**
```python
class Collisions:
    """Столкновения"""
    
class EnergyBalance:
    """Энергетический баланс"""
```

### **9. `bhlff.inversion` - Инверсия**

#### **9.1. `bhlff.inversion.driver` - Драйвер инверсии**
```python
class InverseDriver:
    """Драйвер инверсии"""
    
class LossMetrics:
    """Метрики потерь"""
    
class PriorsRegularizers:
    """Регуляризаторы априори"""
```

#### **9.2. `bhlff.inversion.forward` - Прямое моделирование**
```python
class ForwardWrappers:
    """Обертки прямого моделирования"""
    
class Identifiability:
    """Идентифицируемость"""
    
class PostpredChecks:
    """Пост-предсказательные проверки"""
```

#### **9.3. `bhlff.inversion.particles` - Частицы**
```python
class ParticleTemplate:
    """Шаблон частицы"""
    
class ParticlePassport:
    """Паспорт частицы"""
    
class ParticleInversion:
    """Инверсия частиц"""
```

### **10. `bhlff.experiments` - Эксперименты**

#### **10.1. `bhlff.experiments.cases` - Случаи экспериментов**
```python
class ExperimentCase:
    """Случай эксперимента"""
    
class AcceptanceCriteria:
    """Критерии принятия"""
    
class SweepManager:
    """Менеджер разверток"""
```

#### **10.2. `bhlff.experiments.artifacts` - Артефакты**
```python
class Artifacts:
    """Артефакты эксперимента"""
    
class Report:
    """Отчет эксперимента"""
    
class DataStorage:
    """Хранилище данных"""
```

### **11. `bhlff.testing` - Тестирование**

#### **11.1. `bhlff.testing.validators` - Валидаторы**
```python
class TestValidator:
    """Валидатор тестов"""
    
class AcceptanceValidator:
    """Валидатор принятия"""
    
class QualityValidator:
    """Валидатор качества"""
```

#### **11.2. `bhlff.testing.analyzers` - Анализаторы тестов**
```python
class SobolAnalyzer:
    """Анализатор Соболя"""
    
class RobustnessTester:
    """Тестер робастности"""
    
class DiscretizationAnalyzer:
    """Анализатор дискретизации"""
```

### **12. `bhlff.visualization` - Визуализация**

#### **12.1. `bhlff.visualization.plots` - Графики**
```python
class DataVisualizer:
    """Визуализатор данных"""
    
class FieldPlotter:
    """Построитель графиков полей"""
    
class AnimationGenerator:
    """Генератор анимаций"""
```

#### **12.2. `bhlff.visualization.reports` - Отчеты**
```python
class ReportGenerator:
    """Генератор отчетов"""
    
class TemplateManager:
    """Менеджер шаблонов"""
    
class ExportManager:
    """Менеджер экспорта"""
```

## Зависимости между пакетами

### **Уровень 1 (базовый)**
- `bhlff.core` - не зависит ни от чего

### **Уровень 2 (основной)**
- `bhlff.solvers` → `bhlff.core`
- `bhlff.geometry` → `bhlff.core`
- `bhlff.analysis` → `bhlff.core`

### **Уровень 3 (специализированный)**
- `bhlff.transmission` → `bhlff.solvers`, `bhlff.analysis`
- `bhlff.windows` → `bhlff.analysis`, `bhlff.transmission`
- `bhlff.models` → `bhlff.solvers`, `bhlff.analysis`

### **Уровень 4 (прикладной)**
- `bhlff.dynamics` → `bhlff.solvers`, `bhlff.models`
- `bhlff.inversion` → `bhlff.experiments`, `bhlff.models`
- `bhlff.experiments` → `bhlff.solvers`, `bhlff.analysis`, `bhlff.models`

### **Уровень 5 (инструментальный)**
- `bhlff.testing` → `bhlff.experiments`, `bhlff.models`
- `bhlff.visualization` → `bhlff.analysis`, `bhlff.experiments`

## Преимущества новой структуры

### **1. Четкое разделение ответственности**
- Каждый пакет имеет определенную область ответственности
- Минимальные пересечения между пакетами
- Легко понять, где искать нужную функциональность

### **2. Устранение дублирования**
- Общие компоненты вынесены в базовые пакеты
- Специализированные компоненты в соответствующих пакетах
- Переиспользование кода через наследование и композицию

### **3. Модульность**
- Пакеты можно разрабатывать независимо
- Легко добавлять новые компоненты
- Простое тестирование отдельных модулей

### **4. Масштабируемость**
- Легко добавлять новые уровни моделей
- Простое расширение функциональности
- Гибкая архитектура для будущих изменений

### **5. ВБП-центричность**
- ВБП остается центральным каркасом
- Все компоненты работают через ВБП-интерфейсы
- Согласованность с теорией

## Критерии готовности

- [ ] Создана структура всех пакетов
- [ ] Определены интерфейсы между пакетами
- [ ] Устранено дублирование классов
- [ ] Создана иерархия наследования
- [ ] Реализованы абстрактные базовые классы
- [ ] Настроены зависимости между пакетами
- [ ] Создана документация по пакетам
- [ ] Написаны тесты для каждого пакета

## Заключение

Новая структура пакетов обеспечивает:
- **Четкую организацию** кода по функциональности
- **Устранение дублирования** через общие базовые классы
- **Модульность** и независимость разработки
- **ВБП-центричность** как центральный каркас системы
- **Масштабируемость** для будущих расширений

Эта структура служит основой для рефакторинга существующего кода и создания новых компонентов в соответствии с принципами ВБП-теории.

