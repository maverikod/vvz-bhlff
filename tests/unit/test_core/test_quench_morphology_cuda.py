"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Unit tests for CUDA-based quench morphology operations.

This module contains comprehensive unit tests for the QuenchMorphologyCUDA class,
validating all morphological operations including erosion, dilation, opening,
closing, and connected component labeling in 7D space-time.

Physical Meaning:
    Tests validate morphological operations used for noise filtering and
    connected component analysis in quench detection, ensuring proper
    identification of coherent quench structures in 7D space-time.

Mathematical Foundation:
    Tests validate:
    - Binary erosion: E(A) = {x | B_x ⊆ A}
    - Binary dilation: D(A) = {x | B_x ∩ A ≠ ∅}
    - Binary opening: O(A) = D(E(A))
    - Binary closing: C(A) = E(D(A))
    - Connected component labeling via flood-fill algorithm

Example:
    >>> pytest tests/unit/test_core/test_quench_morphology_cuda.py -v
"""

import numpy as np
import pytest
from typing import Tuple, Dict

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from bhlff.core.bvp.quench_morphology.quench_morphology_cuda import QuenchMorphologyCUDA
from bhlff.core.bvp.quench_morphology.quench_morphology_cpu import QuenchMorphologyCPU


@pytest.fixture
def cuda_processor():
    """Create CUDA processor instance."""
    return QuenchMorphologyCUDA()


@pytest.fixture
def cpu_processor():
    """Create CPU processor instance for fallback testing."""
    return QuenchMorphologyCPU()


@pytest.fixture
def small_7d_mask() -> np.ndarray:
    """
    Create a small 7D binary mask for testing.

    Physical Meaning:
        Creates a test mask with known quench regions for validating
        morphological operations in 7D space-time.

    Returns:
        np.ndarray: Binary mask of shape (5, 5, 5, 3, 3, 3, 3).
    """
    mask = np.zeros((5, 5, 5, 3, 3, 3, 3), dtype=bool)
    # Create a small connected region
    mask[2, 2, 2, 1, 1, 1, 1] = True
    mask[2, 2, 2, 1, 1, 1, 2] = True
    mask[2, 2, 2, 1, 1, 2, 1] = True
    return mask


@pytest.fixture
def noise_7d_mask() -> np.ndarray:
    """
    Create a 7D mask with noise for testing filtering operations.

    Physical Meaning:
        Creates a test mask with both signal (connected regions) and
        noise (isolated pixels) to validate noise filtering operations.

    Returns:
        np.ndarray: Binary mask with noise.
    """
    mask = np.zeros((7, 7, 7, 3, 3, 3, 3), dtype=bool)
    # Create a connected region
    mask[3:5, 3:5, 3:5, 1, 1, 1, 1] = True
    # Add isolated noise pixels
    mask[0, 0, 0, 0, 0, 0, 0] = True
    mask[6, 6, 6, 2, 2, 2, 2] = True
    return mask


@pytest.fixture
def gap_7d_mask() -> np.ndarray:
    """
    Create a 7D mask with gaps for testing closing operations.

    Physical Meaning:
        Creates a test mask with gaps in connected regions to validate
        gap-filling operations (closing).

    Returns:
        np.ndarray: Binary mask with gaps.
    """
    mask = np.zeros((7, 7, 7, 3, 3, 3, 3), dtype=bool)
    # Create region with gap
    mask[3, 3, 3, 1, 1, 1, 1] = True
    mask[3, 3, 3, 1, 1, 1, 2] = True
    # Gap at [3, 3, 3, 1, 1, 1, 0]
    mask[3, 3, 3, 1, 1, 1, 0] = False
    return mask


class TestQuenchMorphologyCUDA:
    """
    Unit tests for QuenchMorphologyCUDA.

    Physical Meaning:
        Tests validate CUDA-accelerated morphological operations for
        quench detection, ensuring correct noise filtering and component
        identification in 7D space-time.
    """

    def test_initialization(self, cuda_processor):
        """
        Test CUDA processor initialization.

        Physical Meaning:
            Validates that the processor is correctly initialized and
            ready for morphological operations.
        """
        assert cuda_processor is not None
        assert isinstance(cuda_processor, QuenchMorphologyCUDA)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_apply_morphological_operations_cuda_basic(
        self, cuda_processor, cpu_processor, small_7d_mask
    ):
        """
        Test basic morphological operations on CUDA.

        Physical Meaning:
            Validates that morphological operations correctly filter
            noise and preserve signal regions in 7D space-time.

        Mathematical Foundation:
            Tests binary opening followed by closing operations.
        """
        # Transfer to GPU
        mask_gpu = cp.asarray(small_7d_mask)

        # Apply morphological operations
        result_gpu = cuda_processor.apply_morphological_operations_cuda(
            mask_gpu, True, cpu_processor
        )

        # Verify result is on GPU
        assert hasattr(result_gpu, "get") or isinstance(result_gpu, cp.ndarray)

        # Transfer back to CPU for validation
        result_cpu = cp.asnumpy(result_gpu)

        # Verify shape preserved
        assert result_cpu.shape == small_7d_mask.shape

        # Verify binary mask
        assert np.all((result_cpu == 0) | (result_cpu == 1))

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_apply_morphological_operations_cuda_noise_filtering(
        self, cuda_processor, cpu_processor, noise_7d_mask
    ):
        """
        Test noise filtering with morphological operations.

        Physical Meaning:
            Validates that morphological operations remove isolated
            noise pixels while preserving connected signal regions.

        Mathematical Foundation:
            Opening removes small noise, closing fills small gaps.
        """
        # Transfer to GPU
        mask_gpu = cp.asarray(noise_7d_mask)

        # Apply morphological operations
        result_gpu = cuda_processor.apply_morphological_operations_cuda(
            mask_gpu, True, cpu_processor
        )

        # Transfer back to CPU
        result_cpu = cp.asnumpy(result_gpu)

        # Verify noise is removed (isolated pixels should be gone)
        # The connected region should be preserved
        original_connected = np.sum(noise_7d_mask[3:5, 3:5, 3:5, 1, 1, 1, 1])
        result_connected = np.sum(result_cpu[3:5, 3:5, 3:5, 1, 1, 1, 1])

        # Connected region should be preserved or slightly modified
        assert result_connected > 0, "Connected region should be preserved"

        # Isolated noise should be removed
        isolated_noise_removed = (
            result_cpu[0, 0, 0, 0, 0, 0, 0] == 0
            and result_cpu[6, 6, 6, 2, 2, 2, 2] == 0
        )
        assert isolated_noise_removed, "Isolated noise pixels should be removed"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_apply_morphological_operations_cuda_fallback(
        self, cuda_processor, cpu_processor, small_7d_mask
    ):
        """
        Test fallback to CPU when CUDA is not available.

        Physical Meaning:
            Validates that the processor correctly falls back to CPU
            operations when CUDA is unavailable, ensuring robustness.

        Args:
            cuda_processor: CUDA processor instance.
            cpu_processor: CPU processor instance.
            small_7d_mask: Test mask.
        """
        # Test with CUDA disabled
        mask_gpu = cp.asarray(small_7d_mask) if CUDA_AVAILABLE else small_7d_mask

        result = cuda_processor.apply_morphological_operations_cuda(
            mask_gpu, False, cpu_processor
        )

        # Result should be valid
        if CUDA_AVAILABLE:
            result_cpu = cp.asnumpy(result)
        else:
            result_cpu = result

        assert result_cpu.shape == small_7d_mask.shape
        assert np.all((result_cpu == 0) | (result_cpu == 1))

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_erosion_cuda_vectorized(self, cuda_processor, small_7d_mask):
        """
        Test CUDA erosion operation.

        Physical Meaning:
            Validates that erosion correctly shrinks regions by removing
            pixels that don't have all neighbors in the structuring element.

        Mathematical Foundation:
            Erosion: E(A) = {x | B_x ⊆ A}
        """
        mask_gpu = cp.asarray(small_7d_mask)
        structure = cp.ones((3, 3, 3, 3, 3, 3, 3), dtype=cp.bool_)

        # Apply erosion
        eroded = cuda_processor._erosion_cuda_vectorized(mask_gpu, structure)

        # Verify result
        assert isinstance(eroded, cp.ndarray)
        assert eroded.shape == small_7d_mask.shape
        assert eroded.dtype == cp.bool_

        # Erosion should shrink or preserve regions
        eroded_cpu = cp.asnumpy(eroded)
        original_sum = np.sum(small_7d_mask)
        eroded_sum = np.sum(eroded_cpu)

        # Erosion should not increase the number of True pixels
        assert eroded_sum <= original_sum, "Erosion should not increase region size"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_dilation_cuda_vectorized(self, cuda_processor, small_7d_mask):
        """
        Test CUDA dilation operation.

        Physical Meaning:
            Validates that dilation correctly expands regions by adding
            pixels where at least one neighbor in the structuring element is True.

        Mathematical Foundation:
            Dilation: D(A) = {x | B_x ∩ A ≠ ∅}
        """
        mask_gpu = cp.asarray(small_7d_mask)
        structure = cp.ones((3, 3, 3, 3, 3, 3, 3), dtype=cp.bool_)

        # Apply dilation
        dilated = cuda_processor._dilation_cuda_vectorized(mask_gpu, structure)

        # Verify result
        assert isinstance(dilated, cp.ndarray)
        assert dilated.shape == small_7d_mask.shape
        assert dilated.dtype == cp.bool_

        # Dilation should expand or preserve regions
        dilated_cpu = cp.asnumpy(dilated)
        original_sum = np.sum(small_7d_mask)
        dilated_sum = np.sum(dilated_cpu)

        # Dilation should not decrease the number of True pixels
        assert dilated_sum >= original_sum, "Dilation should not decrease region size"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_binary_opening_cuda(self, cuda_processor, noise_7d_mask):
        """
        Test binary opening operation.

        Physical Meaning:
            Validates that opening (erosion followed by dilation) correctly
            removes small noise while preserving larger structures.

        Mathematical Foundation:
            Opening: O(A) = D(E(A))
        """
        mask_gpu = cp.asarray(noise_7d_mask)
        structure = cp.ones((3, 3, 3, 3, 3, 3, 3), dtype=cp.bool_)

        # Apply opening
        opened = cuda_processor._binary_opening_cuda(mask_gpu, structure)

        # Verify result
        assert isinstance(opened, cp.ndarray)
        assert opened.shape == noise_7d_mask.shape

        opened_cpu = cp.asnumpy(opened)

        # Opening should remove small noise
        # Isolated pixels should be removed
        assert (
            opened_cpu[0, 0, 0, 0, 0, 0, 0] == 0
        ), "Opening should remove isolated noise"
        assert (
            opened_cpu[6, 6, 6, 2, 2, 2, 2] == 0
        ), "Opening should remove isolated noise"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_binary_closing_cuda(self, cuda_processor, gap_7d_mask):
        """
        Test binary closing operation.

        Physical Meaning:
            Validates that closing (dilation followed by erosion) correctly
            fills small gaps while preserving region boundaries.

        Mathematical Foundation:
            Closing: C(A) = E(D(A))
        """
        mask_gpu = cp.asarray(gap_7d_mask)
        structure = cp.ones((3, 3, 3, 3, 3, 3, 3), dtype=cp.bool_)

        # Apply closing
        closed = cuda_processor._binary_closing_cuda(mask_gpu, structure)

        # Verify result
        assert isinstance(closed, cp.ndarray)
        assert closed.shape == gap_7d_mask.shape

        closed_cpu = cp.asnumpy(closed)

        # Closing should fill small gaps
        # The gap should be filled
        gap_filled = closed_cpu[3, 3, 3, 1, 1, 1, 0] == 1
        assert gap_filled, "Closing should fill small gaps"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_find_connected_components_cuda(
        self, cuda_processor, cpu_processor, small_7d_mask
    ):
        """
        Test connected component labeling on CUDA.

        Physical Meaning:
            Validates that connected component labeling correctly identifies
            and groups spatially/phase/temporally connected quench events.

        Mathematical Foundation:
            Uses flood-fill algorithm to identify connected regions.
        """
        mask_gpu = cp.asarray(small_7d_mask)

        # Find connected components
        components = cuda_processor.find_connected_components_cuda(
            mask_gpu, True, cpu_processor
        )

        # Verify result structure
        assert isinstance(components, dict)
        assert len(components) > 0, "Should find at least one component"

        # Verify each component
        for component_id, component_mask in components.items():
            assert isinstance(component_id, int)
            assert component_id > 0, "Component ID should be positive"
            assert isinstance(component_mask, np.ndarray)
            assert component_mask.shape == small_7d_mask.shape
            assert np.all((component_mask == 0) | (component_mask == 1))

            # Component should be non-empty
            assert np.sum(component_mask) > 0, "Component should be non-empty"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_find_connected_components_cuda_multiple(
        self, cuda_processor, cpu_processor
    ):
        """
        Test connected component labeling with multiple components.

        Physical Meaning:
            Validates that the algorithm correctly identifies multiple
            disconnected quench regions as separate components.

        Mathematical Foundation:
            Tests flood-fill algorithm on multiple seed points.
        """
        # Create mask with two disconnected regions
        mask = np.zeros((7, 7, 7, 3, 3, 3, 3), dtype=bool)
        # First region
        mask[2, 2, 2, 1, 1, 1, 1] = True
        mask[2, 2, 2, 1, 1, 1, 2] = True
        # Second region (disconnected)
        mask[5, 5, 5, 2, 2, 2, 2] = True
        mask[5, 5, 5, 2, 2, 2, 1] = True

        mask_gpu = cp.asarray(mask)

        # Find connected components
        components = cuda_processor.find_connected_components_cuda(
            mask_gpu, True, cpu_processor
        )

        # Should find at least 2 components
        assert len(components) >= 2, "Should find at least 2 disconnected components"

        # Verify components don't overlap
        component_sum = np.zeros_like(mask)
        for component_mask in components.values():
            component_sum += component_mask

        # Each pixel should belong to at most one component
        assert np.all(component_sum <= 1), "Components should not overlap"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_find_connected_components_cuda_empty(self, cuda_processor, cpu_processor):
        """
        Test connected component labeling with empty mask.

        Physical Meaning:
            Validates that the algorithm correctly handles empty masks
            without errors.

        Mathematical Foundation:
            Empty mask should return empty component dictionary.
        """
        # Create empty mask
        mask = np.zeros((5, 5, 5, 3, 3, 3, 3), dtype=bool)
        mask_gpu = cp.asarray(mask)

        # Find connected components
        components = cuda_processor.find_connected_components_cuda(
            mask_gpu, True, cpu_processor
        )

        # Should return empty dictionary
        assert isinstance(components, dict)
        assert len(components) == 0, "Empty mask should return no components"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_convolve_7d_cuda(self, cuda_processor):
        """
        Test 7D convolution operation.

        Physical Meaning:
            Validates that 7D convolution correctly applies the structuring
            element to compute weighted sums of neighbors.

        Mathematical Foundation:
            Convolution: (f * g)(x) = Σ f(y) * g(x - y)
        """
        # Create test mask and structure
        mask = np.ones((5, 5, 5, 3, 3, 3, 3), dtype=np.float32)
        structure = np.ones((3, 3, 3, 3, 3, 3, 3), dtype=np.float32)

        mask_gpu = cp.asarray(mask)
        structure_gpu = cp.asarray(structure)

        # Apply convolution
        convolved = cuda_processor._convolve_7d_cuda(mask_gpu, structure_gpu)

        # Verify result
        assert isinstance(convolved, cp.ndarray)
        assert convolved.shape == mask.shape

        convolved_cpu = cp.asnumpy(convolved)

        # For all-ones mask and structure, result should be constant
        # (except at boundaries)
        center_value = convolved_cpu[2, 2, 2, 1, 1, 1, 1]
        structure_size = np.prod(structure.shape)

        # Center should have value equal to structure size
        assert (
            abs(center_value - structure_size) < 1e-3
        ), f"Center value should equal structure size: {center_value} vs {structure_size}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_shift_7d_cuda(self, cuda_processor):
        """
        Test 7D array shifting operation.

        Physical Meaning:
            Validates that array shifting correctly translates arrays
            by specified offsets for convolution operations.

        Mathematical Foundation:
            Shift: S(A, o) = A(x - o) with zero padding at boundaries.
        """
        # Create test array
        array = np.zeros((5, 5, 5, 3, 3, 3, 3), dtype=np.float32)
        array[2, 2, 2, 1, 1, 1, 1] = 1.0

        array_gpu = cp.asarray(array)

        # Test shift by (1, 0, 0, 0, 0, 0, 0)
        offset = (1, 0, 0, 0, 0, 0, 0)
        shifted = cuda_processor._shift_7d_cuda(array_gpu, offset)

        shifted_cpu = cp.asnumpy(shifted)

        # Original value should be at new position
        assert shifted_cpu[3, 2, 2, 1, 1, 1, 1] == 1.0, "Value should shift correctly"
        assert (
            shifted_cpu[2, 2, 2, 1, 1, 1, 1] == 0.0
        ), "Original position should be zero"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_morphological_operations_idempotency(
        self, cuda_processor, cpu_processor, small_7d_mask
    ):
        """
        Test idempotency of morphological operations.

        Physical Meaning:
            Validates that applying operations multiple times produces
            stable results (idempotency property).

        Mathematical Foundation:
            Opening and closing are idempotent: O(O(A)) = O(A), C(C(A)) = C(A)
        """
        mask_gpu = cp.asarray(small_7d_mask)

        # Apply operations once
        result1 = cuda_processor.apply_morphological_operations_cuda(
            mask_gpu, True, cpu_processor
        )

        # Apply operations again
        result2 = cuda_processor.apply_morphological_operations_cuda(
            result1, True, cpu_processor
        )

        # Results should be identical (idempotency)
        result1_cpu = cp.asnumpy(result1)
        result2_cpu = cp.asnumpy(result2)

        assert np.array_equal(
            result1_cpu, result2_cpu
        ), "Morphological operations should be idempotent"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_morphological_operations_preserve_shape(
        self, cuda_processor, cpu_processor
    ):
        """
        Test that morphological operations preserve array shape.

        Physical Meaning:
            Validates that operations don't change array dimensions,
            maintaining 7D structure throughout processing.

        Mathematical Foundation:
            Morphological operations are shape-preserving transformations.
        """
        # Test with various shapes
        shapes = [
            (5, 5, 5, 3, 3, 3, 3),
            (7, 7, 7, 5, 5, 5, 5),
            (3, 3, 3, 3, 3, 3, 3),
        ]

        for shape in shapes:
            mask = np.random.rand(*shape) > 0.5
            mask_gpu = cp.asarray(mask)

            result = cuda_processor.apply_morphological_operations_cuda(
                mask_gpu, True, cpu_processor
            )

            result_cpu = cp.asnumpy(result)

            assert (
                result_cpu.shape == shape
            ), f"Shape should be preserved: {result_cpu.shape} vs {shape}"
