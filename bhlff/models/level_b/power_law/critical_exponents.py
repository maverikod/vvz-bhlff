"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Critical exponents analysis module for power law analysis.

This module implements critical exponent analysis for the 7D phase field theory,
including computation of all standard critical exponents and universality class determination.

Physical Meaning:
    Analyzes critical behavior of the BVP field using complete 7D critical
    exponent analysis according to the 7D phase field theory.

Mathematical Foundation:
    Implements full critical exponent analysis:
    - ν: correlation length exponent
    - β: order parameter exponent
    - γ: susceptibility exponent
    - δ: critical isotherm exponent
    - η: anomalous dimension
    - α: specific heat exponent
    - z: dynamic exponent
"""

import numpy as np
from typing import Dict, Any, List
import logging

from bhlff.core.bvp import BVPCore
from .estimators import (
    estimate_nu_from_correlation_length as _est_nu,
    estimate_beta_from_tail as _est_beta,
    estimate_chi_from_variance as _est_gamma,
)
from .scaling_functions import (
    compute_correlation_scaling_function as _corr_scaling,
    compute_susceptibility_scaling_function as _sus_scaling,
    compute_order_parameter_scaling_function as _op_scaling,
    identify_critical_regions as _identify_regions,
)
from .anomalous_dimension import compute_anomalous_dimension as _est_eta


class CriticalExponents:
    """
    Critical exponents analysis for BVP field.

    Physical Meaning:
        Computes the complete set of critical exponents for the 7D BVP field
        according to critical phenomena theory.
    """

    def __init__(self, bvp_core: BVPCore):
        """Initialize critical exponents analyzer."""
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def analyze_critical_behavior(self, envelope: np.ndarray) -> Dict[str, Any]:
        """
        Analyze critical behavior with full 7D critical exponents.

        Physical Meaning:
            Analyzes critical behavior of the BVP field using
            complete 7D critical exponent analysis according to
            the 7D phase field theory.
        """
        amplitude = np.abs(envelope)

        # Compute full set of critical exponents
        critical_exponents = self._compute_full_critical_exponents(amplitude)

        # Analyze critical regions
        critical_regions = self._identify_critical_regions(
            amplitude, critical_exponents
        )

        # Compute scaling dimension
        scaling_dimension = self._compute_7d_scaling_dimension(critical_exponents)

        # Determine universality class
        universality_class = self._determine_universality_class(critical_exponents)

        # Compute critical scaling functions
        critical_scaling = self._compute_critical_scaling_functions(
            amplitude, critical_exponents
        )

        return {
            "critical_exponents": critical_exponents,
            "critical_regions": critical_regions,
            "scaling_dimension": scaling_dimension,
            "universality_class": universality_class,
            "critical_scaling": critical_scaling,
        }

    def _compute_full_critical_exponents(
        self, amplitude: np.ndarray
    ) -> Dict[str, float]:
        """Compute full set of critical exponents."""
        # Compute correlation length exponent ν
        nu = self._compute_correlation_length_exponent(amplitude)

        # Compute order parameter exponent β
        beta = self._compute_order_parameter_exponent(amplitude)

        # Compute susceptibility exponent γ
        gamma = self._compute_susceptibility_exponent(amplitude)

        # Compute critical isotherm exponent δ
        delta = self._compute_critical_isotherm_exponent(amplitude)

        # Compute anomalous dimension η
        eta = self._compute_anomalous_dimension(amplitude)

        # Compute specific heat exponent α
        alpha = self._compute_specific_heat_exponent(amplitude)

        # Compute dynamic exponent z
        z = self._compute_dynamic_exponent(amplitude)

        return {
            "nu": float(nu),  # correlation length exponent
            "beta": float(beta),  # order parameter exponent
            "gamma": float(gamma),  # susceptibility exponent
            "delta": float(delta),  # critical isotherm exponent
            "eta": float(eta),  # anomalous dimension
            "alpha": float(alpha),  # specific heat exponent
            "z": float(z),  # dynamic exponent
        }

    def _compute_correlation_length_exponent(self, amplitude: np.ndarray) -> float:
        """Compute correlation length exponent ν."""
        return self.estimate_nu_from_correlation_length(amplitude)

    def _compute_order_parameter_exponent(self, amplitude: np.ndarray) -> float:
        """Compute order parameter exponent β."""
        return self.estimate_beta_from_tail(amplitude)

    def _compute_susceptibility_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute susceptibility exponent γ from actual susceptibility scaling.

        Physical Meaning:
            Computes susceptibility exponent γ from the scaling law
            χ ~ |A - A_c|^(-γ), where A is the order parameter and A_c
            is the critical point. This exponent characterizes how
            susceptibility diverges near the critical point.

        Mathematical Foundation:
            Susceptibility is defined as χ = ∂²F/∂h², where F is the
            free energy and h is the field. Near critical point,
            χ diverges as χ ~ |t|^(-γ) where t is the reduced control
            parameter. In BVP model, we use amplitude deviation from
            critical value as control parameter.

        Args:
            amplitude (np.ndarray): Field amplitude distribution

        Returns:
            float: Susceptibility exponent γ (bounded between 0.5 and 2.0)
        """
        return self.estimate_chi_from_variance(amplitude)

    # -------------------- Thin wrappers using helper modules --------------------
    def estimate_nu_from_correlation_length(self, amplitude: np.ndarray) -> float:
        return _est_nu(self.bvp_core, amplitude)

    def estimate_beta_from_tail(self, amplitude: np.ndarray) -> float:
        return _est_beta(amplitude)

    def estimate_chi_from_variance(self, amplitude: np.ndarray) -> float:
        return _est_gamma(amplitude)

    def _compute_critical_isotherm_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute critical isotherm exponent δ using scaling relation.

        Physical Meaning:
            Computes critical isotherm exponent δ from scaling relation
            δ = (γ + β) / β. This exponent characterizes the critical
            isotherm behavior M ~ H^(1/δ) at T = T_c.

        Mathematical Foundation:
            Uses scaling relation: δ = (γ + β) / β
            where:
            - γ: susceptibility exponent
            - β: order parameter exponent
            This follows from scaling theory in critical phenomena.

        Args:
            amplitude (np.ndarray): Field amplitude distribution.

        Returns:
            float: Critical isotherm exponent δ.

        Raises:
            ValueError: If β ≤ 0 or computed δ is not finite.
        """
        # Use scaling relation: δ = (γ + β) / β
        beta = self._compute_order_parameter_exponent(amplitude)
        gamma = self._compute_susceptibility_exponent(amplitude)

        # Validate β > 0 (no fixed fallback)
        if beta <= 0:
            raise ValueError(
                f"Cannot compute δ: β = {beta} ≤ 0. "
                f"Order parameter exponent must be positive."
            )

        delta = (gamma + beta) / beta

        # Validate result (no fixed fallback)
        if not np.isfinite(delta):
            raise ValueError(f"computed δ is not finite: {delta} (γ={gamma}, β={beta})")

        return float(delta)

    def _compute_anomalous_dimension(self, amplitude: np.ndarray) -> float:
        """Compute anomalous dimension η."""
        return _est_eta(self.bvp_core, amplitude)

    def _compute_specific_heat_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute specific heat exponent α using 7D scaling relation.

        Physical Meaning:
            Computes specific heat exponent α from scaling relation α = 2 - ν*d,
            where d=7 for 7D phase field theory. Specific heat diverges as
            C ~ |t|^{-α} near criticality.

        Mathematical Foundation:
            Uses scaling relation: α = 2 - ν*d where:
            - ν: correlation length exponent
            - d: space-time dimension (7 for 7D BVP theory)
            This follows from hyperscaling relation in critical phenomena.

        Args:
            amplitude (np.ndarray): Field amplitude distribution.

        Returns:
            float: Specific heat exponent α.

        Raises:
            ValueError: If computed α is not finite or violates scaling bounds.
        """
        # Use scaling relation: α = 2 - ν*d
        nu = self._compute_correlation_length_exponent(amplitude)
        # Explicit 7D dimension for 7D phase field theory
        d = 7

        alpha = 2 - nu * d

        # Validate result (no fixed fallback)
        if not np.isfinite(alpha):
            raise ValueError(f"computed α is not finite: {alpha} (ν={nu}, d={d})")

        return float(alpha)

    def _compute_dynamic_exponent(self, amplitude: np.ndarray) -> float:
        """
        Compute dynamic exponent z from block-wise correlation time scaling.

        Physical Meaning:
            Computes dynamic exponent z from the scaling of relaxation time
            τ ~ ξ^z, where ξ is correlation length. This characterizes
            critical slowing down near the critical point. Uses block-wise
            analysis to estimate z from temporal correlation structure.

        Mathematical Foundation:
            Dynamic exponent relates relaxation time to correlation length:
            τ ~ ξ^z. For BVP field, we estimate z from block-wise amplitude
            fluctuation correlations using robust regression on log-log scale.
            Estimates z from fitting log(τ) ~ z*log(ξ) across blocks.

        Args:
            amplitude (np.ndarray): Field amplitude distribution.

        Returns:
            float: Dynamic exponent z.

        Raises:
            ValueError: If insufficient block data or z is not finite.
        """
        from .correlation_analysis import CorrelationAnalysis
        from .robust_fit import robust_loglog_slope
        from .block_utils import iter_blocks
        from .cuda_estimator_utils import (
            get_cuda_backend,
            compute_block_statistics_cuda,
        )

        cuda_backend = get_cuda_backend()
        correlation_analyzer = CorrelationAnalysis(self.bvp_core)

        # Block-wise estimation of z from τ ~ ξ^z
        xi_vals: List[float] = []
        tau_vals: List[float] = []
        total_elems = amplitude.size

        # Adaptive minimum block size for 7D structure
        min_block_elems = max(128, min(262144, int(0.002 * total_elems)))

        block_count = 0
        for block in iter_blocks(amplitude):
            block_arr = amplitude[block]
            if block_arr.size < min_block_elems:
                continue

            try:
                # Compute block statistics using CUDA if available
                if cuda_backend is not None:
                    mean_amp, variance = compute_block_statistics_cuda(
                        block_arr, cuda_backend
                    )
                else:
                    mean_amp = float(np.mean(block_arr))
                    variance = float(np.var(block_arr))

                # Validate statistics
                if mean_amp <= 1e-12 or not np.isfinite(mean_amp):
                    continue
                if variance <= 0 or not np.isfinite(variance):
                    continue

                # Estimate relaxation time scale from fluctuation-to-mean ratio
                # τ ~ variance / (mean^2) for diffusive systems
                tau = variance / (mean_amp**2) if mean_amp > 0 else 1.0

                # Compute correlation length for this block
                corr_7d = correlation_analyzer._compute_7d_correlation_function(
                    block_arr
                )
                corr_lengths = correlation_analyzer._compute_7d_correlation_lengths(
                    corr_7d
                )

                if not corr_lengths:
                    continue

                # Average correlation length across 7 dimensions
                xi = float(np.mean(list(corr_lengths.values())))

                # Validate values
                if xi > 1e-12 and tau > 1e-12 and np.isfinite(xi) and np.isfinite(tau):
                    xi_vals.append(xi)
                    tau_vals.append(tau)
                    block_count += 1
            except Exception as e:
                self.logger.debug(f"Dynamic exponent computation failed for block: {e}")
                continue

        if len(xi_vals) < 3:
            raise ValueError(
                f"insufficient block data for z estimate: only {len(xi_vals)} blocks "
                f"with valid correlations (need ≥3)"
            )

        # Robust log-log fit: log(τ) ~ z*log(ξ)
        slope = robust_loglog_slope(np.asarray(xi_vals), np.asarray(tau_vals))
        z = slope

        # Validate result (no fixed fallback)
        if not np.isfinite(z):
            raise ValueError(
                f"computed z is not finite: {z} from {len(xi_vals)} blocks"
            )

        self.logger.info(
            f"Estimated z={z:.4f} from {block_count} blocks "
            f"(ξ range: [{min(xi_vals):.2e}, {max(xi_vals):.2e}], "
            f"τ range: [{min(tau_vals):.2e}, {max(tau_vals):.2e}])"
        )

        return float(z)

    def _identify_critical_regions(
        self, amplitude: np.ndarray, critical_exponents: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify critical regions with scaling analysis."""
        return _identify_regions(amplitude, critical_exponents)

    def _compute_7d_scaling_dimension(
        self, critical_exponents: Dict[str, float]
    ) -> float:
        """
        Compute effective 7D scaling dimension using scaling relation.

        Physical Meaning:
            Computes effective scaling dimension d_eff from scaling relation
            d_eff = 2 - α - β. This characterizes the effective dimension
            of the critical system in 7D phase field theory.

        Mathematical Foundation:
            Uses hyperscaling relation: d_eff = 2 - α - β
            where α and β are critical exponents. For 7D BVP theory,
            this gives the effective dimension of critical fluctuations.

        Args:
            critical_exponents (Dict[str, float]): Dictionary of critical exponents.

        Returns:
            float: Effective 7D scaling dimension.

        Raises:
            KeyError: If required exponents are missing.
            ValueError: If computed d_eff is not finite.
        """
        # Use scaling relation: d_eff = 2 - α - β
        # Require explicit values (no defaults)
        if "alpha" not in critical_exponents:
            raise KeyError("Missing 'alpha' exponent for scaling dimension computation")
        if "beta" not in critical_exponents:
            raise KeyError("Missing 'beta' exponent for scaling dimension computation")

        alpha = critical_exponents["alpha"]
        beta = critical_exponents["beta"]

        d_eff = 2 - alpha - beta

        # Validate result (no fixed bounds, but check finiteness)
        if not np.isfinite(d_eff):
            raise ValueError(
                f"computed d_eff is not finite: {d_eff} (α={alpha}, β={beta})"
            )

        return float(d_eff)

    def _determine_universality_class(
        self, critical_exponents: Dict[str, float]
    ) -> str:
        """Determine universality class from critical exponents."""
        # Compare with known universality classes
        nu = critical_exponents.get("nu", 0.5)
        beta = critical_exponents.get("beta", 0.5)
        gamma = critical_exponents.get("gamma", 1.0)
        # eta is not directly used for class determination here

        # Mean field values
        if abs(nu - 0.5) < 0.1 and abs(beta - 0.5) < 0.1 and abs(gamma - 1.0) < 0.1:
            return "mean_field"

        # Ising-like values
        elif abs(nu - 0.63) < 0.1 and abs(beta - 0.33) < 0.1:
            return "ising_3d"

        # XY-like values
        elif abs(nu - 0.67) < 0.1 and abs(beta - 0.35) < 0.1:
            return "xy_3d"

        # Custom 7D values
        else:
            return "custom_7d"

    def _compute_critical_scaling_functions(
        self, amplitude: np.ndarray, critical_exponents: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute critical scaling functions."""
        # Compute scaling functions
        scaling_functions = {
            "correlation_scaling": self._compute_correlation_scaling_function(
                amplitude, critical_exponents
            ),
            "susceptibility_scaling": self._compute_susceptibility_scaling_function(
                amplitude, critical_exponents
            ),
            "order_parameter_scaling": self._compute_order_parameter_scaling_function(
                amplitude, critical_exponents
            ),
        }

        return scaling_functions

    def _compute_correlation_scaling_function(
        self, amplitude: np.ndarray, critical_exponents: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute correlation scaling function."""
        return _corr_scaling(self.bvp_core, amplitude, critical_exponents)

    def _compute_susceptibility_scaling_function(
        self, amplitude: np.ndarray, critical_exponents: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute susceptibility scaling function."""
        return _sus_scaling(amplitude, critical_exponents)

    def _compute_order_parameter_scaling_function(
        self, amplitude: np.ndarray, critical_exponents: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute order parameter scaling function."""
        return _op_scaling(amplitude, critical_exponents)
