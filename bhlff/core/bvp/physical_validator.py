"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validator for BVP methods and results.

This module implements comprehensive physical validation for BVP methods,
ensuring that all results are consistent with the theoretical framework
and physical principles of the 7D phase field theory.
"""

# flake8: noqa: E501,E203

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod

# Memory protection handled at higher level


class PhysicalValidator(ABC):
    """
    Abstract base class for physical validation.

    Physical Meaning:
        Provides the foundation for physical validation of BVP methods
        and results according to the 7D phase field theory framework.
    """

    def __init__(self, domain_shape: Tuple[int, ...], parameters: Dict[str, Any]):
        """
        Initialize physical validator.

        Physical Meaning:
            Sets up the validator with domain information and physical
            parameters for comprehensive validation.

        Args:
            domain_shape (Tuple[int, ...]): Shape of the computational domain.
            parameters (Dict[str, Any]): Physical parameters for validation.
        """
        self.domain_shape = domain_shape
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)

        # Physical constraints
        self.physical_constraints = self._setup_physical_constraints()
        self.theoretical_bounds = self._setup_theoretical_bounds()

    def _setup_physical_constraints(self) -> Dict[str, Any]:
        """Setup physical constraints for validation."""
        return {
            "energy_conservation_tolerance": 1e-6,
            "causality_tolerance": 1e-8,
            "phase_coherence_minimum": 0.1,
            "amplitude_bounds": (1e-15, 1e12),
            "frequency_bounds": (1e-6, 1e15),
            "phase_bounds": (-2 * np.pi, 2 * np.pi),
            "gradient_bounds": (1e-20, 1e10),
        }

    def _setup_theoretical_bounds(self) -> Dict[str, Any]:
        """Setup theoretical bounds for validation."""
        return {
            "max_field_energy": 1e15,
            "max_phase_gradient": 1e8,
            "min_coherence_length": 1e-12,
            "max_coherence_length": 1e3,
            "temporal_causality_limit": 1e-6,
            "spatial_resolution_limit": 1e-15,
        }

    @abstractmethod
    def validate_physical_constraints(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate physical constraints.

        Physical Meaning:
            Validates that the result satisfies all physical constraints
            and theoretical requirements.
        """
        raise NotImplementedError(
            "Subclasses must implement validate_physical_constraints"
        )

    @abstractmethod
    def validate_theoretical_bounds(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate theoretical bounds.

        Physical Meaning:
            Validates that the result is within theoretical bounds
            and limits.
        """
        raise NotImplementedError(
            "Subclasses must implement validate_theoretical_bounds"
        )


class BVPPhysicalValidator(PhysicalValidator):
    """
    Physical validator for BVP methods and results.

    Physical Meaning:
        Validates that all BVP methods and results are consistent
        with the theoretical framework and physical principles of
        the 7D phase field theory.

    Mathematical Foundation:
        Implements validation according to:
        - Energy conservation: E_total = E_field + E_kinetic + E_potential
        - Causality: |∂φ/∂t| ≤ c (speed of light)
        - Phase coherence: |⟨exp(iφ)⟩| ≥ threshold
        - 7D structure preservation: dim(field) = 7
    """

    def __init__(self, domain_shape: Tuple[int, ...], parameters: Dict[str, Any]):
        """
        Initialize BVP physical validator.

        Physical Meaning:
            Sets up the validator with comprehensive physical constraints
            and theoretical bounds for BVP validation.

        Args:
            domain_shape (Tuple[int, ...]): Shape of the 7D computational domain.
            parameters (Dict[str, Any]): BVP parameters for validation.
        """
        super().__init__(domain_shape, parameters)

        # BVP-specific constraints
        self.bvp_constraints = self._setup_bvp_constraints()
        self.validation_metrics = {}

    def _setup_bvp_constraints(self) -> Dict[str, Any]:
        """Setup BVP-specific constraints."""
        return {
            "envelope_equation_tolerance": 1e-8,
            "nonlinear_coefficient_bounds": (1e-12, 1e6),
            "quench_threshold_bounds": (0.0, 1.0),
            "topological_charge_bounds": (-10.0, 10.0),
            "power_law_exponent_bounds": (0.0, 3.0),
            "phase_field_coherence": 0.5,
            "energy_density_bounds": (1e-15, 1e12),
        }

    def validate_physical_constraints(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate physical constraints for BVP results.

        Physical Meaning:
            Validates that the BVP result satisfies all physical constraints
            including energy conservation, causality, phase coherence, and
            7D structure preservation according to the theoretical framework.

        Mathematical Foundation:
            Checks:
            - Energy conservation: |E_final - E_initial| < tolerance
            - Causality: |∇φ| ≤ c (speed of light constraint)
            - Phase coherence: |⟨exp(iφ)⟩| ≥ minimum_coherence
            - 7D structure: field.ndim == 7 and field.shape == domain_shape

        Args:
            result (Dict[str, Any]): BVP result to validate.

        Returns:
            Dict[str, Any]: Comprehensive validation results including
                physical_constraints_valid, energy_conservation, causality,
                phase_coherence, structure_preservation, and detailed metrics.
        """
        self.logger.info("Starting physical constraints validation")

        validation_result = {
            "physical_constraints_valid": True,
            "constraint_violations": [],
            "constraint_warnings": [],
            "detailed_metrics": {},
        }

        try:
            # Extract result components
            field = result.get("field", None)
            energy = result.get("energy", None)
            phase = result.get("phase", None)
            metadata = result.get("metadata", {})

            if field is None:
                validation_result["constraint_violations"].append("Missing field data")
                validation_result["physical_constraints_valid"] = False
                return validation_result

            # 1. Energy conservation validation
            energy_validation = self._validate_energy_conservation(
                field, energy, metadata
            )
            validation_result["detailed_metrics"][
                "energy_conservation"
            ] = energy_validation

            if not energy_validation["valid"]:
                validation_result["constraint_violations"].extend(
                    energy_validation["violations"]
                )
                validation_result["physical_constraints_valid"] = False

            # 2. Causality validation
            causality_validation = self._validate_causality(field, metadata)
            validation_result["detailed_metrics"]["causality"] = causality_validation

            if not causality_validation["valid"]:
                validation_result["constraint_violations"].extend(
                    causality_validation["violations"]
                )
                validation_result["physical_constraints_valid"] = False

            # 3. Phase coherence validation
            coherence_validation = self._validate_phase_coherence(
                field, phase, metadata
            )
            validation_result["detailed_metrics"][
                "phase_coherence"
            ] = coherence_validation

            if not coherence_validation["valid"]:
                validation_result["constraint_warnings"].extend(
                    coherence_validation["warnings"]
                )

            # 4. 7D structure preservation validation
            structure_validation = self._validate_7d_structure(field, metadata)
            validation_result["detailed_metrics"][
                "structure_preservation"
            ] = structure_validation

            if not structure_validation["valid"]:
                validation_result["constraint_violations"].extend(
                    structure_validation["violations"]
                )
                validation_result["physical_constraints_valid"] = False

            # 5. Amplitude bounds validation
            amplitude_validation = self._validate_amplitude_bounds(field, metadata)
            validation_result["detailed_metrics"][
                "amplitude_bounds"
            ] = amplitude_validation

            if not amplitude_validation["valid"]:
                validation_result["constraint_violations"].extend(
                    amplitude_validation["violations"]
                )
                validation_result["physical_constraints_valid"] = False

            # 6. Gradient bounds validation
            gradient_validation = self._validate_gradient_bounds(field, metadata)
            validation_result["detailed_metrics"][
                "gradient_bounds"
            ] = gradient_validation

            if not gradient_validation["valid"]:
                validation_result["constraint_violations"].extend(
                    gradient_validation["violations"]
                )
                validation_result["physical_constraints_valid"] = False

        except Exception as e:
            self.logger.error(f"Physical constraints validation failed: {e}")
            validation_result["constraint_violations"].append(
                f"Validation error: {str(e)}"
            )
            validation_result["physical_constraints_valid"] = False

        self.logger.info(
            f"Physical constraints validation completed: {'PASSED' if validation_result['physical_constraints_valid'] else 'FAILED'}"
        )
        return validation_result

    def validate_theoretical_bounds(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate theoretical bounds for BVP results.

        Physical Meaning:
            Validates that the BVP result is within theoretical bounds
            and limits according to the 7D phase field theory framework.

        Mathematical Foundation:
            Checks:
            - Field energy: E_field ≤ E_max_theoretical
            - Phase gradients: |∇φ| ≤ ∇φ_max_theoretical
            - Coherence length: L_min ≤ L_coherence ≤ L_max
            - Temporal causality: Δt ≥ Δt_min_causal
            - Spatial resolution: Δx ≥ Δx_min_theoretical

        Args:
            result (Dict[str, Any]): BVP result to validate.

        Returns:
            Dict[str, Any]: Comprehensive theoretical validation results.
        """
        self.logger.info("Starting theoretical bounds validation")

        validation_result = {
            "theoretical_bounds_valid": True,
            "bound_violations": [],
            "bound_warnings": [],
            "detailed_metrics": {},
        }

        try:
            # Extract result components
            field = result.get("field", None)
            metadata = result.get("metadata", {})

            if field is None:
                validation_result["bound_violations"].append("Missing field data")
                validation_result["theoretical_bounds_valid"] = False
                return validation_result

            # 1. Field energy bounds validation
            energy_bounds_validation = self._validate_field_energy_bounds(
                field, metadata
            )
            validation_result["detailed_metrics"][
                "field_energy_bounds"
            ] = energy_bounds_validation

            if not energy_bounds_validation["valid"]:
                validation_result["bound_violations"].extend(
                    energy_bounds_validation["violations"]
                )
                validation_result["theoretical_bounds_valid"] = False

            # 2. Phase gradient bounds validation
            gradient_bounds_validation = self._validate_phase_gradient_bounds(
                field, metadata
            )
            validation_result["detailed_metrics"][
                "phase_gradient_bounds"
            ] = gradient_bounds_validation

            if not gradient_bounds_validation["valid"]:
                validation_result["bound_violations"].extend(
                    gradient_bounds_validation["violations"]
                )
                validation_result["theoretical_bounds_valid"] = False

            # 3. Coherence length bounds validation
            coherence_length_validation = self._validate_coherence_length_bounds(
                field, metadata
            )
            validation_result["detailed_metrics"][
                "coherence_length_bounds"
            ] = coherence_length_validation

            if not coherence_length_validation["valid"]:
                validation_result["bound_warnings"].extend(
                    coherence_length_validation["warnings"]
                )

            # 4. Temporal causality bounds validation
            temporal_causality_validation = self._validate_temporal_causality_bounds(
                field, metadata
            )
            validation_result["detailed_metrics"][
                "temporal_causality_bounds"
            ] = temporal_causality_validation

            if not temporal_causality_validation["valid"]:
                validation_result["bound_violations"].extend(
                    temporal_causality_validation["violations"]
                )
                validation_result["theoretical_bounds_valid"] = False

            # 5. Spatial resolution bounds validation
            spatial_resolution_validation = self._validate_spatial_resolution_bounds(
                field, metadata
            )
            validation_result["detailed_metrics"][
                "spatial_resolution_bounds"
            ] = spatial_resolution_validation

            if not spatial_resolution_validation["valid"]:
                validation_result["bound_warnings"].extend(
                    spatial_resolution_validation["warnings"]
                )

        except Exception as e:
            self.logger.error(f"Theoretical bounds validation failed: {e}")
            validation_result["bound_violations"].append(f"Validation error: {str(e)}")
            validation_result["theoretical_bounds_valid"] = False

        self.logger.info(
            f"Theoretical bounds validation completed: {'PASSED' if validation_result['theoretical_bounds_valid'] else 'FAILED'}"
        )
        return validation_result

    def _validate_energy_conservation(
        self,
        field: np.ndarray,
        energy: Optional[Dict[str, float]],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate energy conservation."""
        try:
            # Compute field energy
            field_energy = np.sum(np.abs(field) ** 2)

            # Check energy conservation if initial energy provided
            if energy is not None and "initial_energy" in energy:
                energy_balance = abs(field_energy - energy["initial_energy"])
                tolerance = self.physical_constraints["energy_conservation_tolerance"]

                return {
                    "valid": energy_balance < tolerance,
                    "field_energy": float(field_energy),
                    "energy_balance": float(energy_balance),
                    "tolerance": tolerance,
                    "violations": (
                        []
                        if energy_balance < tolerance
                        else [
                            f"Energy conservation violation: {energy_balance} > {tolerance}"
                        ]
                    ),
                }
            else:
                return {
                    "valid": True,
                    "field_energy": float(field_energy),
                    "energy_balance": None,
                    "tolerance": None,
                    "violations": [],
                }
        except Exception as e:
            return {
                "valid": False,
                "field_energy": None,
                "energy_balance": None,
                "tolerance": None,
                "violations": [f"Energy conservation validation error: {str(e)}"],
            }

    def _validate_causality(
        self, field: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate causality constraints."""
        try:
            # Check temporal gradients (causality)
            if field.ndim >= 7:  # Has time dimension
                temporal_gradients = np.gradient(field, axis=-1)  # Time axis
                max_temporal_gradient = np.max(np.abs(temporal_gradients))

                # Causality constraint: |∂φ/∂t| ≤ c (speed of light)
                speed_of_light = 299792458  # m/s
                causality_violation = max_temporal_gradient > speed_of_light

                return {
                    "valid": not causality_violation,
                    "max_temporal_gradient": float(max_temporal_gradient),
                    "speed_of_light": speed_of_light,
                    "violations": (
                        []
                        if not causality_violation
                        else [
                            f"Causality violation: {max_temporal_gradient} > {speed_of_light}"
                        ]
                    ),
                }
            else:
                return {
                    "valid": True,
                    "max_temporal_gradient": None,
                    "speed_of_light": None,
                    "violations": [],
                }
        except Exception as e:
            return {
                "valid": False,
                "max_temporal_gradient": None,
                "speed_of_light": None,
                "violations": [f"Causality validation error: {str(e)}"],
            }

    def _validate_phase_coherence(
        self, field: np.ndarray, phase: Optional[np.ndarray], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate phase coherence."""
        try:
            if phase is not None:
                # Compute phase coherence
                phase_coherence = np.abs(np.mean(np.exp(1j * phase)))
                min_coherence = self.physical_constraints["phase_coherence_minimum"]

                return {
                    "valid": phase_coherence >= min_coherence,
                    "phase_coherence": float(phase_coherence),
                    "min_coherence": min_coherence,
                    "warnings": (
                        []
                        if phase_coherence >= min_coherence
                        else [
                            f"Low phase coherence: {phase_coherence} < {min_coherence}"
                        ]
                    ),
                }
            else:
                # Extract phase from field
                field_phase = np.angle(field)
                phase_coherence = np.abs(np.mean(np.exp(1j * field_phase)))
                min_coherence = self.physical_constraints["phase_coherence_minimum"]

                return {
                    "valid": phase_coherence >= min_coherence,
                    "phase_coherence": float(phase_coherence),
                    "min_coherence": min_coherence,
                    "warnings": (
                        []
                        if phase_coherence >= min_coherence
                        else [
                            f"Low phase coherence: {phase_coherence} < {min_coherence}"
                        ]
                    ),
                }
        except Exception as e:
            return {
                "valid": False,
                "phase_coherence": None,
                "min_coherence": None,
                "warnings": [f"Phase coherence validation error: {str(e)}"],
            }

    def _validate_7d_structure(
        self, field: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate 7D structure preservation."""
        try:
            # Check 7D structure
            expected_ndim = 7
            structure_valid = (
                field.ndim == expected_ndim and field.shape == self.domain_shape
            )

            return {
                "valid": structure_valid,
                "field_ndim": field.ndim,
                "expected_ndim": expected_ndim,
                "field_shape": field.shape,
                "domain_shape": self.domain_shape,
                "violations": (
                    []
                    if structure_valid
                    else [
                        f"7D structure violation: ndim={field.ndim}, shape={field.shape}"
                    ]
                ),
            }
        except Exception as e:
            return {
                "valid": False,
                "field_ndim": None,
                "expected_ndim": None,
                "field_shape": None,
                "domain_shape": None,
                "violations": [f"7D structure validation error: {str(e)}"],
            }

    def _validate_amplitude_bounds(
        self, field: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate amplitude bounds."""
        try:
            field_amplitude = np.abs(field)
            max_amplitude = np.max(field_amplitude)
            min_amplitude = np.min(field_amplitude)

            amplitude_bounds = self.physical_constraints["amplitude_bounds"]
            amplitude_valid = (
                amplitude_bounds[0] <= min_amplitude <= amplitude_bounds[1]
                and amplitude_bounds[0] <= max_amplitude <= amplitude_bounds[1]
            )

            return {
                "valid": amplitude_valid,
                "max_amplitude": float(max_amplitude),
                "min_amplitude": float(min_amplitude),
                "amplitude_bounds": amplitude_bounds,
                "violations": (
                    []
                    if amplitude_valid
                    else [
                        f"Amplitude bounds violation: {min_amplitude}-{max_amplitude} not in {amplitude_bounds}"
                    ]
                ),
            }
        except Exception as e:
            return {
                "valid": False,
                "max_amplitude": None,
                "min_amplitude": None,
                "amplitude_bounds": None,
                "violations": [f"Amplitude bounds validation error: {str(e)}"],
            }

    def _validate_gradient_bounds(
        self, field: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate gradient bounds."""
        try:
            # Compute spatial gradients
            spatial_gradients = []
            for axis in range(min(3, field.ndim)):  # Spatial dimensions
                grad = np.gradient(field, axis=axis)
                spatial_gradients.append(np.abs(grad))

            if spatial_gradients:
                max_gradient = np.max([np.max(grad) for grad in spatial_gradients])
                gradient_bounds = self.physical_constraints["gradient_bounds"]
                gradient_valid = (
                    gradient_bounds[0] <= max_gradient <= gradient_bounds[1]
                )

                return {
                    "valid": gradient_valid,
                    "max_gradient": float(max_gradient),
                    "gradient_bounds": gradient_bounds,
                    "violations": (
                        []
                        if gradient_valid
                        else [
                            f"Gradient bounds violation: {max_gradient} not in {gradient_bounds}"
                        ]
                    ),
                }
            else:
                return {
                    "valid": True,
                    "max_gradient": None,
                    "gradient_bounds": None,
                    "violations": [],
                }
        except Exception as e:
            return {
                "valid": False,
                "max_gradient": None,
                "gradient_bounds": None,
                "violations": [f"Gradient bounds validation error: {str(e)}"],
            }

    def _validate_field_energy_bounds(
        self, field: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate field energy bounds."""
        try:
            field_energy = np.sum(np.abs(field) ** 2)
            max_energy = self.theoretical_bounds["max_field_energy"]
            energy_valid = field_energy <= max_energy

            return {
                "valid": energy_valid,
                "field_energy": float(field_energy),
                "max_energy": max_energy,
                "violations": (
                    []
                    if energy_valid
                    else [
                        f"Field energy exceeds theoretical maximum: {field_energy} > {max_energy}"
                    ]
                ),
            }
        except Exception as e:
            return {
                "valid": False,
                "field_energy": None,
                "max_energy": None,
                "violations": [f"Field energy bounds validation error: {str(e)}"],
            }

    def _validate_phase_gradient_bounds(
        self, field: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate phase gradient bounds."""
        try:
            # Extract phase and compute gradients
            field_phase = np.angle(field)
            max_phase_gradient = 0.0

            for axis in range(min(3, field.ndim)):  # Spatial dimensions
                phase_grad = np.gradient(field_phase, axis=axis)
                max_phase_gradient = max(max_phase_gradient, np.max(np.abs(phase_grad)))

            max_gradient_bound = self.theoretical_bounds["max_phase_gradient"]
            gradient_valid = max_phase_gradient <= max_gradient_bound

            return {
                "valid": gradient_valid,
                "max_phase_gradient": float(max_phase_gradient),
                "max_gradient_bound": max_gradient_bound,
                "violations": (
                    []
                    if gradient_valid
                    else [
                        f"Phase gradient exceeds theoretical maximum: {max_phase_gradient} > {max_gradient_bound}"
                    ]
                ),
            }
        except Exception as e:
            return {
                "valid": False,
                "max_phase_gradient": None,
                "max_gradient_bound": None,
                "violations": [f"Phase gradient bounds validation error: {str(e)}"],
            }

    def _validate_coherence_length_bounds(
        self, field: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate coherence length bounds."""
        try:
            # Estimate coherence length from field correlation
            field_amplitude = np.abs(field)
            coherence_length = self._estimate_coherence_length(field_amplitude)

            min_length = self.theoretical_bounds["min_coherence_length"]
            max_length = self.theoretical_bounds["max_coherence_length"]
            length_valid = min_length <= coherence_length <= max_length

            return {
                "valid": length_valid,
                "coherence_length": float(coherence_length),
                "min_length": min_length,
                "max_length": max_length,
                "warnings": (
                    []
                    if length_valid
                    else [
                        f"Coherence length outside theoretical bounds: {coherence_length} not in [{min_length}, {max_length}]"
                    ]
                ),
            }
        except Exception as e:
            return {
                "valid": False,
                "coherence_length": None,
                "min_length": None,
                "max_length": None,
                "warnings": [f"Coherence length bounds validation error: {str(e)}"],
            }

    def _validate_temporal_causality_bounds(
        self, field: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate temporal causality bounds."""
        try:
            if field.ndim >= 7:  # Has time dimension
                # Check temporal causality
                temporal_limit = self.theoretical_bounds["temporal_causality_limit"]

                # Simple causality check: no instantaneous changes
                time_axis = -1
                temporal_changes = np.abs(np.diff(field, axis=time_axis))
                max_temporal_change = np.max(temporal_changes)

                causality_valid = max_temporal_change <= temporal_limit

                return {
                    "valid": causality_valid,
                    "max_temporal_change": float(max_temporal_change),
                    "temporal_limit": temporal_limit,
                    "violations": (
                        []
                        if causality_valid
                        else [
                            f"Temporal causality violation: {max_temporal_change} > {temporal_limit}"
                        ]
                    ),
                }
            else:
                return {
                    "valid": True,
                    "max_temporal_change": None,
                    "temporal_limit": None,
                    "violations": [],
                }
        except Exception as e:
            return {
                "valid": False,
                "max_temporal_change": None,
                "temporal_limit": None,
                "violations": [f"Temporal causality bounds validation error: {str(e)}"],
            }

    def _validate_spatial_resolution_bounds(
        self, field: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate spatial resolution bounds."""
        try:
            # Check spatial resolution
            resolution_limit = self.theoretical_bounds["spatial_resolution_limit"]

            # Estimate resolution from field variations
            spatial_variations = []
            for axis in range(min(3, field.ndim)):  # Spatial dimensions
                field_slice = np.take(field, 0, axis=axis)  # Take first slice
                variations = np.abs(np.diff(field_slice))
                spatial_variations.extend(variations.flatten())

            if spatial_variations:
                min_variation = np.min(spatial_variations)
                resolution_valid = min_variation >= resolution_limit

                return {
                    "valid": resolution_valid,
                    "min_spatial_variation": float(min_variation),
                    "resolution_limit": resolution_limit,
                    "warnings": (
                        []
                        if resolution_valid
                        else [
                            f"Spatial resolution below theoretical limit: {min_variation} < {resolution_limit}"
                        ]
                    ),
                }
            else:
                return {
                    "valid": True,
                    "min_spatial_variation": None,
                    "resolution_limit": None,
                    "warnings": [],
                }
        except Exception as e:
            return {
                "valid": False,
                "min_spatial_variation": None,
                "resolution_limit": None,
                "warnings": [f"Spatial resolution bounds validation error: {str(e)}"],
            }

    def _estimate_coherence_length(self, field_amplitude: np.ndarray) -> float:
        """Estimate coherence length from field amplitude."""
        try:
            # Simple coherence length estimation from correlation
            if field_amplitude.size < 2:
                return 1e-6  # Default small length

            # Compute 1D correlation along first spatial axis
            axis = 0
            if field_amplitude.ndim > axis:
                field_1d = np.mean(
                    field_amplitude, axis=tuple(range(1, field_amplitude.ndim))
                )
                correlation = np.correlate(field_1d, field_1d, mode="full")
                correlation = correlation[len(correlation) // 2 :]

                # Find correlation length (where correlation drops to 1/e)
                max_correlation = correlation[0]
                target_correlation = max_correlation / np.e

                coherence_length = 1e-6  # Default
                for i, corr in enumerate(correlation):
                    if corr <= target_correlation:
                        coherence_length = i * 1e-6  # Scale factor
                        break

                return coherence_length
            else:
                return 1e-6
        except Exception:
            return 1e-6

    def get_validation_summary(
        self, physical_result: Dict[str, Any], theoretical_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get comprehensive validation summary.

        Physical Meaning:
            Provides a comprehensive summary of both physical and theoretical
            validation results for easy interpretation and reporting.

        Args:
            physical_result (Dict[str, Any]): Physical constraints validation result.
            theoretical_result (Dict[str, Any]): Theoretical bounds validation result.

        Returns:
            Dict[str, Any]: Comprehensive validation summary.
        """
        return {
            "overall_valid": (
                physical_result.get("physical_constraints_valid", False)
                and theoretical_result.get("theoretical_bounds_valid", False)
            ),
            "physical_validation": physical_result,
            "theoretical_validation": theoretical_result,
            "total_violations": (
                len(physical_result.get("constraint_violations", []))
                + len(theoretical_result.get("bound_violations", []))
            ),
            "total_warnings": (
                len(physical_result.get("constraint_warnings", []))
                + len(theoretical_result.get("bound_warnings", []))
            ),
            "validation_timestamp": np.datetime64("now").astype(str),
        }
