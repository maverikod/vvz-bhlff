"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Detailed action plan for resolving GPU memory overflow in Level A resolvers.

This plan enumerates concrete steps across source generators, solvers, and
analyzers to ensure block processing, vectorization, CUDA enforcement, and swap
integration are consistently applied, preventing memory saturation during Level
A regression tests.
"""

# Plan: Enforce Blocked CUDA Workflows for Level A Components

## 1. Source Generators (`bhlff/core/sources/bvp_source_generators_basic.py`)
- **Methods**: `generate_plane_wave_source`, `generate_gaussian_source`, `generate_distributed_source`.
- **Goal**: Replace direct CuPy mesh-grid allocation with block-aware generation.
  - Integrate `OptimalBlockSizeCalculator.calculate_for_7d()` to determine per-axis tiling before allocating arrays.
  - Use `BlockedFieldGenerator` or `block_7d_expansion.expand_spatial_to_7d` with block streaming to avoid building full 7D tensors at once.
  - Return `FieldArray` objects allocated through the swap manager (`FieldArray(shape=..., swap_threshold_gb=...)`) so large sources immediately opt-in to swap.
  - Enforce CUDA presence via a shared helper (`CUDABackend.require_cuda()`), raising `RuntimeError` if unavailable.
  - Batch generate phase/time expansions using `vectorized_block_operations` utilities instead of nested loops (each batch limited to 80% GPU memory; use CUDA streams to overlap phase and time tiles).

## 2. Block Configuration & Utilities
- **Files**: `bhlff/core/domain/optimal_block_size_calculator.py`, `bhlff/core/sources/block_config.py`.
- **Goal**: Ensure a single configuration source for 80% GPU memory usage.
  - Expose a helper (e.g., `get_default_block_calculator()`) returning a configured calculator with `gpu_memory_ratio=0.8`.
  - Update `BlockConfig.compute_optimal_block_size()` to delegate to the calculator and cache results per domain/dtype.
  - Propagate calculator instances to generators/solvers via constructor injection or lazy singleton.
  - Document which downstream modules can consume calculator-provided tilings for batching (sources, analyzers) vs. those that must rely on contiguous streaming (FFT kernels).

## 3. Field Creation & Swap (`bhlff/core/arrays/field_array.py`, `bhlff/core/sources/blocked_field_generator.py`)
- **Goal**: Guarantee all large arrays funnel through swap-aware pathways.
  - Add convenience factory `FieldArray.from_block_generator(block_generator, swap_threshold_gb=None)` to materialize fields chunk-by-chunk.
  - Extend `BlockedFieldGenerator.get_block_by_indices()` to accept GPU tensors directly and stream them through `cp.cuda.Stream` for minimal peak memory.
  - Update documentation/tests to ensure any manual `np.zeros`/`cp.zeros` is replaced with these factories.
  - Where algorithms tolerate batched evaluation (e.g., amplitude modulation, weighting), surface `FieldArray.iter_batches(block_shape)` to provide batched views while keeping swap semantics intact.

## 4. FFT Solver Path (`bhlff/core/fft/fft_solver_7d_basic/*.py`, `bhlff/core/fft/unified/*.py`)
- **Methods**: `FFTSolver7DBasicSolveMixin.solve_stationary`, `UnifiedSpectralOperations.forward_fft`, `forward_fft_blocked`.
- **Goals**:
  - Before FFT, convert incoming `FieldArray` to a `BlockedField` representation if the total size exceeds calculator limits.
  - **Preserve logical array integrity:** FFT kernels must see one continuous tensor. Implement a streaming loader that sequentially feeds blocks to the same FFT call (no splitting into multiple FFTs) by paging data through pinned buffers and CUDA streams while maintaining the single-kernel view.
  - Ensure `_ops.forward_fft` receives iterables of GPU blocks as a streaming source, not as independent FFT jobs—use `cuda_stream_block_processor` to stage sub-blocks while emulating a contiguous field.
  - Add explicit swap hooks so each block is fetched from disk just-in-time, processed, and released.
  - Harden CUDA enforcement: `UnifiedSpectralOperations` must call `CUDABackend.require_cuda()` when initialized with `use_cuda=True`, failing fast otherwise.
  - Introduce log statements capturing block sizes, memory ratios, and swap status for diagnostics.

### 4.1 Batched Auxiliary Operations
- Batched conjugate-gradient preconditioning, residual normalization, and spectrum analysis can operate on independent channel groups. Update helper routines (`FFTSolver7DBasicSolveMixin._apply_preconditioner_batches`, `UnifiedSpectralOperations.compute_power_spectrum`) to:
  - Accept `batch_iterator` yielding up to 80% GPU memory chunks per batch.
  - Use CUDA streams per batch to overlap transfers with computation.
  - Record batch identifiers in logs for traceability.

## 5. Level A Analyzers & Tests
- **Files**: `tests/unit/test_level_a/test_A01_plane_wave*.py`, `test_A02_multi_plane*.py`, `test_A05_residual_energy*.py`, `test_A11_scale_length.py`.
- **Goals**:
  - Replace direct creation of `np.zeros`/`cp.zeros` in helper functions with calls to the refactored generators returning swap-managed `FieldArray`.
  - Where tests still reshape arrays, ensure they operate on views (`FieldArray.array[:, :, :, 0,0,0,0]`) without duplicating entire tensors.
  - Add assertions that `FieldArray.is_swapped` becomes `True` for large cases to confirm swap usage.
  - Leverage `GPUMemoryMonitor` fixtures to verify runtime stays within 80% memory usage.
  - Add explicit coverage for batched analyzer utilities: tests should drive `analyzer.process_batches()` with synthetic data to prove batching preserves physics.

## 6. Monitoring & Diagnostics
- **Files**: `bhlff/utils/gpu_memory_monitor.py`, `bhlff/utils/cuda_backend.py`.
- **Goals**:
  - Hook monitors into generator/solver contexts to log warnings before block creation.
  - Ensure `CUDABackend.fft` and other kernels emit measured block sizes and stream counts to aid debugging.
  - Add regression tests (new module `tests/unit/test_core/test_blocked_cuda_memory.py`) that simulate low-memory GPUs by patching `gpu_memory_ratio` and confirm block resizing occurs.
  - Extend monitors to record batch identifiers and stream concurrency so we can verify batching only runs where algorithms allow independence.

## 7. Documentation & Compliance
- **Docs**: Update `IMPROVEMENTS_BLOCK_CUDA_MEMORY.md` with a “Status: in progress” note referencing this plan.
- **Coding Standards**: After implementation, rerun `code_mapper` to verify no file exceeds 400 lines and no docstring gaps remain.

---

**Exit Criteria**:
1. All Level A tests complete without `cupy.cuda.memory.OutOfMemoryError`.
2. `code_analysis/code_issues.yaml` no longer lists Level A files under `files_too_large`.
3. Logs show block sizes derived from `OptimalBlockSizeCalculator` and swap usage for large arrays.

