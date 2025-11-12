# Список доработок для улучшения блочной обработки, векторизации, CUDA и защиты памяти

## Критические проблемы в текущей реализации

### 1. Тесты не используют готовые генераторы и решатели

**Проблема:**
- Тест `test_A11_scale_length.py` создаёт источники вручную вместо использования `BVPSource` и `BVPSourceGenerators`
- Тест не использует готовые решатели `FFTSolver7DBasic` или `FFTSolver7DAdvanced`
- Тест не использует `UnifiedSpectralOperations` с автоматической блочной обработкой

**Требуется:**
- Переписать все тесты для использования готовых генераторов полей (`BVPSource`, `BVPSourceGenerators`)
- Использовать готовые решатели (`FFTSolver7DBasic`, `FFTSolver7DAdvanced`) вместо ручной реализации
- Использовать `FieldArray` для автоматического управления памятью и swap

**Файлы для исправления:**
- `tests/unit/test_level_a/test_A11_scale_length.py`
- `tests/unit/test_level_a/test_A12_units_invariance.py`
- Другие тесты Level A, которые создают поля вручную

---

## 2. Блочная обработка

### 2.1. Автоматический выбор размера блока (80% GPU памяти)

**Текущее состояние:**
- Есть реализация в `EnhancedBlockProcessorUtils.calculate_optimal_block_size()`
- Есть реализация в `BlockConfig.compute_optimal_block_size()`
- Используется 80% GPU памяти, но не везде

**Проблемы:**
1. **Несогласованность в разных модулях:**
   - `EnhancedBlockProcessorUtils` использует `memory_fraction=0.8`
   - `BlockConfig` использует `gpu_memory_ratio=0.8`
   - `UnifiedSpectralOperations` использует `_gpu_memory_ratio=0.8`
   - Нет единого места для настройки этого параметра

2. **Не все компоненты используют автоматический расчёт:**
   - `OptimizedBlockProcessor._calculate_optimal_block_size()` использует только CPU память
   - `SimpleBlockProcessor` не имеет автоматического расчёта
   - Некоторые тесты используют фиксированные размеры блоков

3. **Отсутствие динамической адаптации:**
   - Размер блока вычисляется один раз при инициализации
   - Не адаптируется при изменении доступной памяти
   - Не учитывает текущую загрузку GPU другими процессами

**Требуется:**
1. Создать единый класс `OptimalBlockSizeCalculator` с единым API:
   ```python
   class OptimalBlockSizeCalculator:
       def __init__(self, gpu_memory_ratio: float = 0.8):
           self.gpu_memory_ratio = gpu_memory_ratio
       
       def calculate_for_7d(
           self, 
           domain_shape: Tuple[int, ...],
           dtype: np.dtype = np.complex128,
           overhead_factor: float = 5.0
       ) -> Tuple[int, ...]:
           """Вычисляет оптимальный размер блока для 7D поля."""
   ```

2. Интегрировать во все компоненты:
   - `EnhancedBlockProcessor`
   - `OptimizedBlockProcessor`
   - `SimpleBlockProcessor`
   - `BlockConfig`
   - `UnifiedSpectralOperations`

3. Добавить динамическую адаптацию:
   - Пересчитывать размер блока перед каждой операцией
   - Учитывать текущую загрузку GPU
   - Логировать изменения размера блока

**Файлы для доработки:**
- `bhlff/core/domain/enhanced_block_processing_utils.py`
- `bhlff/core/domain/optimized_block_processor.py`
- `bhlff/core/domain/simple_block_processor.py`
- `bhlff/core/sources/block_config.py`
- `bhlff/core/fft/unified/facade.py`
- Создать: `bhlff/core/domain/optimal_block_size_calculator.py`

### 2.2. Векторизация блочной обработки

**Текущее состояние:**
- Есть `VectorizedBlockProcessor` с базовой векторизацией
- Есть `GPUBlockProcessor` с CUDA векторизацией
- Не все операции векторизованы

**Проблемы:**
1. **Неполная векторизация:**
   - Извлечение блоков (`extract_block`) не векторизовано
   - Объединение блоков (`merge_blocks`) не векторизовано
   - Проверка границ блоков выполняется в цикле

2. **Отсутствие батчинга:**
   - Блоки обрабатываются по одному
   - Нет группировки блоков для параллельной обработки
   - Не используется `cupy.RawKernel` для кастомных операций

3. **Неоптимальное использование GPU:**
   - Множественные передачи данных CPU↔GPU
   - Нет использования CUDA streams для параллельной обработки
   - Нет использования shared memory для кэширования

**Требуется:**
1. Векторизовать извлечение и объединение блоков:
   ```python
   def extract_blocks_vectorized(
       self, 
       field: cp.ndarray, 
       block_indices: List[Tuple[int, ...]]
   ) -> cp.ndarray:
       """Векторизованное извлечение нескольких блоков за раз."""
       # Использовать cupy advanced indexing
       # Использовать cupy.RawKernel для кастомных операций
   ```

2. Реализовать батчинг:
   ```python
   def process_blocks_batched(
       self,
       blocks: List[cp.ndarray],
       operation: str,
       batch_size: int = None  # Автоматически на основе GPU памяти
   ) -> List[cp.ndarray]:
       """Обработка блоков батчами для оптимального использования GPU."""
   ```

3. Использовать CUDA streams:
   ```python
   def process_blocks_streamed(
       self,
       blocks: List[cp.ndarray],
       operation: str
   ) -> List[cp.ndarray]:
       """Параллельная обработка блоков с использованием CUDA streams."""
       streams = [cp.cuda.Stream() for _ in range(num_streams)]
       # Обработка блоков параллельно в разных streams
   ```

**Файлы для доработки:**
- `bhlff/core/domain/vectorized_block_processor.py`
- `bhlff/core/domain/enhanced_block_processing/gpu_block_processor.py`
- Создать: `bhlff/core/domain/vectorized_block_operations.py`
- Создать: `bhlff/core/domain/cuda_block_kernels.py`

### 2.3. Оптимизация для 7D структуры

**Проблемы:**
1. **Не учитывается 7D структура M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ:**
   - Блоки разбиваются равномерно по всем измерениям
   - Не учитывается, что пространственные (x,y,z) и фазовые (φ₁,φ₂,φ₃) измерения имеют разную природу
   - Временное измерение (t) обрабатывается так же, как пространственные

2. **Неоптимальное тилирование:**
   - Используется кубическое тилирование для всех измерений
   - Не учитывается, что для фазовых измерений может быть оптимальнее другое тилирование

**Требуется:**
1. Реализовать 7D-специфичное тилирование:
   ```python
   def compute_7d_block_tiling(
       self,
       domain_shape: Tuple[int, int, int, int, int, int, int],
       available_memory: int
   ) -> Tuple[int, int, int, int, int, int, int]:
       """
       Вычисляет оптимальное тилирование для 7D структуры.
       
       Учитывает:
       - Пространственные измерения (0,1,2): большие блоки
       - Фазовые измерения (3,4,5): средние блоки
       - Временное измерение (6): маленькие блоки или полное
       """
   ```

2. Оптимизировать порядок обработки:
   - Сначала обрабатывать пространственные измерения
   - Затем фазовые измерения
   - Временное измерение обрабатывать последним или полностью

**Файлы для доработки:**
- `bhlff/core/domain/enhanced_block_processing_utils.py`
- `bhlff/core/bvp/phase_vector/electroweak_block_sizing.py`
- Создать: `bhlff/core/domain/7d_block_tiling.py`

---

## 3. Векторизация

### 3.1. Векторизация операций над полями

**Проблемы:**
1. **Использование циклов вместо векторизации:**
   - В `test_A11_scale_length.py` используется цикл для сравнения мод
   - Многие операции выполняются поэлементно в циклах
   - Не используется `numpy.vectorize` или `cupy.ElementwiseKernel`

2. **Отсутствие векторизации для сложных операций:**
   - Вычисление спектральных коэффициентов выполняется поэлементно
   - Сравнение решений выполняется в циклах
   - Интерполяция выполняется для каждого элемента отдельно

**Требуется:**
1. Векторизовать все операции над массивами:
   ```python
   # Вместо циклов:
   for i in range(N):
       for j in range(N):
           for k in range(N):
               result[i, j, k] = operation(field[i, j, k])
   
   # Использовать векторизацию:
   result = operation(field)  # Автоматическая векторизация NumPy/CuPy
   ```

2. Использовать `cupy.ElementwiseKernel` для кастомных операций:
   ```python
   custom_kernel = cp.ElementwiseKernel(
       'T x, T y',
       'T z',
       'z = x * y + sin(x)',
       'custom_operation'
   )
   result = custom_kernel(field1, field2)
   ```

**Файлы для доработки:**
- `tests/unit/test_level_a/test_A11_scale_length.py`
- `tests/unit/test_level_a/test_A12_units_invariance.py`
- Все тесты, использующие циклы для обработки массивов

### 3.2. Векторизация спектральных операций

**Проблемы:**
1. **FFT операции не полностью векторизованы:**
   - Используется `fftn` для всего поля, что может быть неоптимально для больших полей
   - Нет батчинга FFT операций для нескольких блоков

2. **Спектральные коэффициенты вычисляются поэлементно:**
   - В `FFTSolver7DBasicCoefficientsMixin` коэффициенты вычисляются для всего поля сразу
   - Для больших полей это может привести к переполнению памяти

**Требуется:**
1. Реализовать батчинг FFT:
   ```python
   def batched_fftn(
       self,
       fields: List[np.ndarray],
       normalization: str = "ortho"
   ) -> List[np.ndarray]:
       """Выполняет FFT для нескольких полей батчами."""
       # Использовать cupy.fft для батчинга
   ```

2. Векторизовать вычисление спектральных коэффициентов:
   ```python
   def compute_spectral_coefficients_vectorized(
       self,
       k_arrays: List[np.ndarray],
       mu: float,
       beta: float,
       lam: float
   ) -> np.ndarray:
       """Векторизованное вычисление спектральных коэффициентов."""
       # Использовать векторизацию NumPy/CuPy
   ```

**Файлы для доработки:**
- `bhlff/core/fft/fft_solver_7d_basic/fft_solver_7d_basic_coefficients.py`
- `bhlff/core/fft/unified/facade.py`
- Создать: `bhlff/core/fft/batched_fft_operations.py`

---

## 4. Интеграция с CUDA

### 4.1. Единообразное использование CUDA

**Проблемы:**
1. **Разные способы проверки доступности CUDA:**
   - В одних местах: `CUDA_AVAILABLE`
   - В других: `cp is not None`
   - В третьих: `cuda_available` флаг

2. **Не везде используется CUDA по умолчанию:**
   - Некоторые компоненты требуют явного указания `use_cuda=True`
   - Нет единой политики: когда использовать CUDA, когда CPU

3. **Отсутствие fallback механизмов:**
   - При ошибке CUDA операции падают с исключением
   - Нет автоматического fallback на CPU

**Требуется:**
1. Создать единый модуль для работы с CUDA:
   ```python
   # bhlff/utils/cuda_backend.py
   class CUDABackend:
       @staticmethod
       def is_available() -> bool:
           """Проверяет доступность CUDA."""
       
       @staticmethod
       def get_backend():
           """Возвращает cupy или numpy в зависимости от доступности."""
       
       @staticmethod
       def require_cuda():
           """Требует CUDA, выбрасывает исключение если недоступна."""
   ```

2. Реализовать автоматический fallback:
   ```python
   def process_with_fallback(
       self,
       operation: Callable,
       *args,
       **kwargs
   ):
       """Выполняет операцию с автоматическим fallback на CPU."""
       try:
           return operation(*args, use_cuda=True, **kwargs)
       except (RuntimeError, MemoryError) as e:
           logger.warning(f"CUDA failed: {e}, falling back to CPU")
           return operation(*args, use_cuda=False, **kwargs)
   ```

**Файлы для доработки:**
- Создать: `bhlff/utils/cuda_backend.py`
- Обновить все модули для использования единого API
- `bhlff/core/fft/unified/facade.py`
- `bhlff/core/domain/enhanced_block_processor.py`

### 4.2. Оптимизация CUDA операций

**Проблемы:**
1. **Неоптимальное использование GPU памяти:**
   - Множественные копии данных на GPU
   - Не используется pinned memory для быстрых передач
   - Нет использования unified memory

2. **Отсутствие асинхронных операций:**
   - Все операции синхронные
   - Нет перекрытия вычислений и передач данных
   - Не используются CUDA streams для параллелизма

3. **Неоптимальные CUDA kernels:**
   - Используются общие kernels вместо специализированных
   - Нет использования shared memory
   - Нет использования tensor cores (если доступны)

**Требуется:**
1. Использовать pinned memory:
   ```python
   def allocate_pinned_memory(self, shape: Tuple[int, ...], dtype: np.dtype):
       """Выделяет pinned memory для быстрых передач CPU↔GPU."""
       return cp.cuda.alloc_pinned_memory(shape, dtype)
   ```

2. Реализовать асинхронные операции:
   ```python
   def async_transfer_and_compute(
       self,
       field_cpu: np.ndarray,
       operation: Callable
   ) -> cp.ndarray:
       """Асинхронная передача и вычисление с перекрытием."""
       stream = cp.cuda.Stream()
       with stream:
           field_gpu = cp.asarray(field_cpu)
           result = operation(field_gpu)
       stream.synchronize()
       return result
   ```

3. Оптимизировать CUDA kernels:
   ```python
   # Использовать shared memory для кэширования
   @cp.RawKernel(r'''
   extern "C" __global__
   void optimized_7d_operation(
       const double* input,
       double* output,
       int N
   ) {
       __shared__ double cache[256];
       // Использование shared memory
   }
   ''', 'optimized_7d_operation')
   ```

**Файлы для доработки:**
- `bhlff/core/domain/enhanced_block_processing/gpu_block_operations.py`
- `bhlff/core/bvp/bvp_core/bvp_cuda_block/bvp_cuda_block_operations.py`
- Создать: `bhlff/core/cuda/optimized_kernels.py`
- Создать: `bhlff/core/cuda/memory_management.py`

---

## 5. Защита от переполнения памяти

### 5.1. Защита GPU памяти

**Текущее состояние:**
- Есть проверки в `BlockedFieldGenerator.get_block_by_indices()`
- Есть проверки в `BlockGenerator.generate_block()`
- Есть проверки в `EnhancedBlockProcessor.optimize_for_field()`

**Проблемы:**
1. **Неполная защита:**
   - Проверки выполняются только перед операциями
   - Нет проверки во время выполнения операций
   - Нет защиты от фрагментации памяти

2. **Отсутствие мониторинга:**
   - Нет отслеживания использования памяти в реальном времени
   - Нет предупреждений при приближении к лимиту
   - Нет автоматического освобождения памяти

3. **Не используется memory pool:**
   - Каждая операция выделяет память заново
   - Нет переиспользования памяти
   - Нет управления памятью на уровне приложения

**Требуется:**
1. Реализовать мониторинг GPU памяти:
   ```python
   class GPUMemoryMonitor:
       def __init__(self, warning_threshold: float = 0.75, critical_threshold: float = 0.9):
           self.warning_threshold = warning_threshold
           self.critical_threshold = critical_threshold
       
       def check_memory(self) -> Dict[str, Any]:
           """Проверяет использование GPU памяти."""
           mem_info = cp.cuda.runtime.memGetInfo()
           free = mem_info[0]
           total = mem_info[1]
           used = total - free
           usage_ratio = used / total
           
           if usage_ratio > self.critical_threshold:
               raise MemoryError(f"GPU memory critical: {usage_ratio:.1%} used")
           elif usage_ratio > self.warning_threshold:
               logger.warning(f"GPU memory high: {usage_ratio:.1%} used")
           
           return {
               "free": free,
               "total": total,
               "used": used,
               "usage_ratio": usage_ratio
           }
   ```

2. Использовать memory pool:
   ```python
   class GPUMemoryPool:
       def __init__(self, max_memory_ratio: float = 0.8):
           self.pool = cp.get_default_memory_pool()
           self.max_memory = self._get_max_memory() * max_memory_ratio
       
       def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> cp.ndarray:
           """Выделяет память из pool с проверкой лимита."""
           required = np.prod(shape) * np.dtype(dtype).itemsize
           if required > self.max_memory:
               raise MemoryError(f"Required {required} exceeds limit {self.max_memory}")
           return self.pool.malloc(shape, dtype)
   ```

3. Реализовать автоматическое освобождение:
   ```python
   @contextmanager
   def gpu_memory_context(max_usage_ratio: float = 0.8):
       """Контекстный менеджер для автоматического управления GPU памятью."""
       monitor = GPUMemoryMonitor()
       try:
           monitor.check_memory()
           yield
       finally:
           cp.get_default_memory_pool().free_all_blocks()
           cp.get_default_pinned_memory_pool().free_all_blocks()
   ```

**Файлы для доработки:**
- Создать: `bhlff/utils/gpu_memory_monitor.py`
- Создать: `bhlff/utils/gpu_memory_pool.py`
- Обновить: `bhlff/core/sources/blocked_field_generator.py`
- Обновить: `bhlff/core/domain/enhanced_block_processor.py`

### 5.2. Защита CPU памяти

**Проблемы:**
1. **Отсутствие защиты CPU памяти:**
   - Нет проверок перед выделением больших массивов
   - Нет защиты от OOM на CPU
   - Нет использования swap для больших массивов

2. **Не используется FieldArray для всех полей:**
   - Многие тесты создают массивы напрямую через `np.array()`
   - Не используется автоматический swap через `FieldArray`
   - Нет прозрачного управления памятью

**Требуется:**
1. Использовать `FieldArray` везде:
   ```python
   # Вместо:
   field = np.array(data)
   
   # Использовать:
   from bhlff.core.arrays import FieldArray
   field = FieldArray(array=data)  # Автоматический swap при превышении лимита
   ```

2. Реализовать мониторинг CPU памяти:
   ```python
   class CPUMemoryMonitor:
       def check_memory(self, required_bytes: int) -> bool:
           """Проверяет, достаточно ли CPU памяти."""
           available = psutil.virtual_memory().available
           if required_bytes > available * 0.8:  # 80% лимит
               return False
           return True
   ```

3. Использовать memory-mapped arrays для больших данных:
   ```python
   def create_large_array(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
       """Создаёт большой массив с использованием memory mapping."""
       if np.prod(shape) * np.dtype(dtype).itemsize > LARGE_ARRAY_THRESHOLD:
           return np.memmap(temp_file, dtype=dtype, shape=shape, mode='w+')
       else:
           return np.zeros(shape, dtype=dtype)
   ```

**Файлы для доработки:**
- Создать: `bhlff/utils/cpu_memory_monitor.py`
- Обновить все тесты для использования `FieldArray`
- Обновить генераторы для использования `FieldArray`

### 5.3. Защита от фрагментации памяти

**Проблемы:**
1. **Фрагментация GPU памяти:**
   - Множественные выделения и освобождения
   - Нет дефрагментации
   - Может привести к OOM даже при достаточном свободном месте

2. **Фрагментация CPU памяти:**
   - Аналогичная проблема на CPU
   - Особенно критично для долгих вычислений

**Требуется:**
1. Реализовать дефрагментацию GPU памяти:
   ```python
   def defragment_gpu_memory():
       """Дефрагментирует GPU память."""
       pool = cp.get_default_memory_pool()
       pool.free_all_blocks()
       # Принудительная синхронизация для освобождения
       cp.cuda.Stream.null.synchronize()
   ```

2. Использовать предварительное выделение:
   ```python
   class PreallocatedMemoryPool:
       def __init__(self, total_size: int):
           self.pool = cp.get_default_memory_pool()
           # Предварительно выделить большой блок
           self.preallocated = cp.zeros(total_size, dtype=cp.complex128)
       
       def get_chunk(self, size: int) -> cp.ndarray:
           """Возвращает кусок из предварительно выделенного блока."""
           # Использовать views вместо новых выделений
   ```

**Файлы для доработки:**
- Создать: `bhlff/utils/memory_defragmentation.py`
- Обновить: `bhlff/core/domain/enhanced_block_processor.py`

---

## Приоритеты доработок

### Критично (P0):
1. Переписать тесты для использования готовых генераторов и решателей
2. Создать единый `OptimalBlockSizeCalculator`
3. Реализовать `GPUMemoryMonitor` и `CPUMemoryMonitor`
4. Использовать `FieldArray` везде

### Высокий приоритет (P1):
1. Векторизовать все операции над массивами
2. Реализовать батчинг для FFT и других операций
3. Использовать CUDA streams для параллелизма
4. Реализовать автоматический fallback CPU↔GPU

### Средний приоритет (P2):
1. Оптимизировать CUDA kernels
2. Реализовать 7D-специфичное тилирование
3. Использовать pinned memory
4. Реализовать дефрагментацию памяти

### Низкий приоритет (P3):
1. Использовать tensor cores (если доступны)
2. Реализовать unified memory
3. Оптимизировать порядок обработки измерений

---

## Метрики успеха

После реализации доработок ожидается:
1. **Производительность:**
   - Ускорение операций на GPU в 2-5 раз
   - Снижение времени обработки больших полей на 50-80%
   - Улучшение использования GPU до 80-90%

2. **Надёжность:**
   - Отсутствие OOM ошибок при работе с большими полями
   - Автоматический fallback при проблемах с GPU
   - Стабильная работа при длительных вычислениях

3. **Удобство использования:**
   - Единый API для всех компонентов
   - Автоматическая оптимизация без ручной настройки
   - Прозрачное управление памятью

