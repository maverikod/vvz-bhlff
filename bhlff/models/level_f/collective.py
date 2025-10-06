"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Collective excitations implementation for Level F models.

This module implements the CollectiveExcitations class for studying
collective excitations in multi-particle systems. It includes methods
for exciting the system with external fields, analyzing responses,
and computing dispersion relations.

Theoretical Background:
    Collective excitations in multi-particle systems are described by
    linear response theory. The system response to external fields
    reveals collective modes and their dispersion relations.
    
    The response function is given by:
    R(ω) = χ(ω) F(ω)
    where χ(ω) is the susceptibility and F(ω) is the external field.

Example:
    >>> excitations = CollectiveExcitations(system, excitation_params)
    >>> response = excitations.excite_system(external_field)
    >>> analysis = excitations.analyze_response(response)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..base.abstract_model import AbstractModel


class CollectiveExcitations(AbstractModel):
    """
    Collective excitations in multi-particle systems.

    Physical Meaning:
        Studies the response of multi-particle systems to
        external fields, identifying collective modes and
        their dispersion relations.

    Mathematical Foundation:
        Implements linear response theory for collective
        excitations in the effective potential framework.

    Attributes:
        system (MultiParticleSystem): Multi-particle system
        excitation_params (Dict[str, Any]): Excitation parameters
        frequency_range (Tuple[float, float]): Frequency range for analysis
        amplitude (float): Excitation amplitude
        excitation_type (str): Type of excitation
    """

    def __init__(
        self, system: "MultiParticleSystem", excitation_params: Dict[str, Any]
    ):
        """
        Initialize collective excitations model.

        Physical Meaning:
            Sets up the model for studying collective excitations
            in the multi-particle system.

        Args:
            system (MultiParticleSystem): Multi-particle system
            excitation_params (Dict): Parameters including:
                - frequency_range: [ω_min, ω_max]
                - amplitude: A (excitation amplitude)
                - type: "harmonic", "impulse", "sweep"
        """
        super().__init__(system.domain)
        self.system = system
        self.excitation_params = excitation_params

        # Extract parameters
        self.frequency_range = excitation_params.get("frequency_range", [0.1, 10.0])
        self.amplitude = excitation_params.get("amplitude", 0.1)
        self.excitation_type = excitation_params.get("type", "harmonic")
        self.duration = excitation_params.get("duration", 100.0)

        # Setup analysis parameters
        self._setup_analysis_parameters()

    def excite_system(self, external_field: np.ndarray) -> np.ndarray:
        """
        Excite the system with external field.

        Physical Meaning:
            Applies external field to the system and
            computes the response.

        Args:
            external_field (np.ndarray): External field F(x,t)

        Returns:
            np.ndarray: System response R(x,t)
        """
        if self.excitation_type == "harmonic":
            return self._harmonic_excitation(external_field)
        elif self.excitation_type == "impulse":
            return self._impulse_excitation(external_field)
        elif self.excitation_type == "sweep":
            return self._frequency_sweep_excitation(external_field)
        else:
            raise ValueError(f"Unknown excitation type: {self.excitation_type}")

    def analyze_response(self, response: np.ndarray) -> Dict[str, Any]:
        """
        Analyze system response to excitation.

        Physical Meaning:
            Extracts collective mode frequencies and
            amplitudes from the response.

        Args:
            response (np.ndarray): System response R(x,t)

        Returns:
            Dict containing:
                - frequencies: ω_n (collective frequencies)
                - amplitudes: A_n (mode amplitudes)
                - damping: γ_n (damping rates)
                - participation: p_n (particle participation)
        """
        # FFT analysis
        response_fft = np.fft.fft(response, axis=-1)
        frequencies = np.fft.fftfreq(response.shape[-1], self.dt)

        # Find spectral peaks
        peaks = self._find_spectral_peaks(np.abs(response_fft), frequencies)

        # Analyze damping
        damping_analysis = self._analyze_damping(response)

        # Compute participation ratios
        participation = self._compute_participation_ratios(response)

        # Quality factors
        quality_factors = self._compute_quality_factors(peaks, damping_analysis)

        return {
            "frequencies": frequencies,
            "peaks": peaks,
            "damping_analysis": damping_analysis,
            "participation": participation,
            "quality_factors": quality_factors,
            "spectrum": response_fft,
        }

    def compute_dispersion_relations(self) -> Dict[str, Any]:
        """
        Compute dispersion relations for collective modes.

        Physical Meaning:
            Calculates ω(k) relations for collective
            excitations in the system.

        Returns:
            Dict containing:
                - wave_vectors: k (wave vector magnitudes)
                - frequencies: ω(k) (dispersion relation)
                - group_velocities: v_g = dω/dk
                - phase_velocities: v_φ = ω/k
        """
        # Create wave vector grid
        k_values = np.linspace(0, self.k_max, self.n_k_points)

        # Compute frequencies for each k
        frequencies = []
        group_velocities = []
        phase_velocities = []

        for k in k_values:
            # Solve dispersion equation
            omega = self._solve_dispersion_equation(k)
            frequencies.append(omega)

            # Compute group velocity
            v_g = self._compute_group_velocity(k, omega)
            group_velocities.append(v_g)

            # Compute phase velocity
            v_phi = omega / k if k > 0 else 0
            phase_velocities.append(v_phi)

        # Fit dispersion relation
        dispersion_fit = self._fit_dispersion_relation(k_values, frequencies)

        return {
            "k_values": k_values,
            "frequencies": np.array(frequencies),
            "group_velocities": np.array(group_velocities),
            "phase_velocities": np.array(phase_velocities),
            "dispersion_fit": dispersion_fit,
        }

    def compute_susceptibility(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute susceptibility function χ(ω).

        Physical Meaning:
            Calculates the linear response susceptibility
            for collective excitations.

        Args:
            frequencies (np.ndarray): Frequency array

        Returns:
            np.ndarray: Susceptibility χ(ω)
        """
        # Get collective modes
        modes = self.system.find_collective_modes()
        mode_frequencies = modes["frequencies"]
        mode_amplitudes = modes["amplitudes"]

        # Compute susceptibility
        susceptibility = np.zeros_like(frequencies, dtype=complex)

        for i, (omega_n, A_n) in enumerate(zip(mode_frequencies, mode_amplitudes)):
            # Lorentzian response
            gamma = 0.1 * omega_n  # Damping rate
            susceptibility += A_n / (
                omega_n**2 - frequencies**2 - 1j * gamma * frequencies
            )

        return susceptibility

    def _setup_analysis_parameters(self) -> None:
        """
        Setup analysis parameters for collective excitations.

        Physical Meaning:
            Initializes parameters needed for analysis
            of collective excitations.
        """
        self.dt = 0.01  # Time step
        self.k_max = 10.0  # Maximum wave vector
        self.n_k_points = 100  # Number of k points
        self.peak_threshold = 0.1  # Peak detection threshold
        self.damping_threshold = 0.01  # Damping analysis threshold

    def _harmonic_excitation(self, external_field: np.ndarray) -> np.ndarray:
        """
        Apply harmonic excitation to the system.

        Physical Meaning:
            Applies harmonic external field and computes
            the steady-state response.
        """
        # Time array
        t = np.arange(0, self.duration, self.dt)

        # Harmonic excitation
        omega = np.mean(self.frequency_range)
        excitation = self.amplitude * np.sin(2 * np.pi * omega * t)

        # Apply to system
        response = self._apply_excitation(external_field, excitation)

        return response

    def _impulse_excitation(self, external_field: np.ndarray) -> np.ndarray:
        """
        Apply impulse excitation to the system.

        Physical Meaning:
            Applies impulse external field and computes
            the transient response.
        """
        # Time array
        t = np.arange(0, self.duration, self.dt)

        # Impulse excitation
        impulse_duration = 0.1
        excitation = np.zeros_like(t)
        mask = t < impulse_duration
        excitation[mask] = self.amplitude

        # Apply to system
        response = self._apply_excitation(external_field, excitation)

        return response

    def _frequency_sweep_excitation(self, external_field: np.ndarray) -> np.ndarray:
        """
        Apply frequency sweep excitation to the system.

        Physical Meaning:
            Applies frequency sweep external field and
            computes the response across frequency range.
        """
        # Time array
        t = np.arange(0, self.duration, self.dt)

        # Frequency sweep
        omega_start, omega_end = self.frequency_range
        omega_t = omega_start + (omega_end - omega_start) * t / self.duration

        # Sweep excitation
        excitation = self.amplitude * np.sin(2 * np.pi * omega_t * t)

        # Apply to system
        response = self._apply_excitation(external_field, excitation)

        return response

    def _apply_excitation(
        self, external_field: np.ndarray, excitation: np.ndarray
    ) -> np.ndarray:
        """
        Apply excitation to the system and compute response.

        Physical Meaning:
            Applies the external excitation to the system
            and computes the resulting response.
        """
        # Get system dynamics
        dynamics_matrix = self.system._compute_dynamics_matrix()

        # Initialize response
        n_particles = len(self.system.particles)
        response = np.zeros((n_particles, len(excitation)))

        # Time integration
        for i, t in enumerate(np.arange(0, self.duration, self.dt)):
            if i == 0:
                continue

            # External force
            F = self._compute_external_force(external_field, excitation[i])

            # Solve dynamics equation
            # M ẍ + K x = F
            # This is simplified - in practice would use proper time integration
            response[:, i] = np.linalg.solve(dynamics_matrix, F)

        return response

    def _compute_external_force(
        self, external_field: np.ndarray, excitation_amplitude: float
    ) -> np.ndarray:
        """
        Compute external force on particles.

        Physical Meaning:
            Calculates the external force acting
            on each particle due to the external field.
        """
        n_particles = len(self.system.particles)
        forces = np.zeros(n_particles)

        for i, particle in enumerate(self.system.particles):
            # Force from external field at particle position
            # This is simplified - in practice would interpolate field
            # Handle different field dimensions
            if external_field.ndim == 3:
                force = external_field[0, 0, 0] * excitation_amplitude * particle.charge
            elif external_field.ndim == 7:
                force = (
                    external_field[0, 0, 0, 0, 0, 0, 0]
                    * excitation_amplitude
                    * particle.charge
                )
            else:
                # Use mean value for other dimensions
                force = np.mean(external_field) * excitation_amplitude * particle.charge
            forces[i] = force

        return forces

    def _find_spectral_peaks(
        self, spectrum: np.ndarray, frequencies: np.ndarray
    ) -> Dict[str, Any]:
        """
        Find spectral peaks in the response.

        Physical Meaning:
            Identifies resonant frequencies in the
            system response spectrum.
        """
        # Find peaks above threshold
        peak_indices = []
        peak_frequencies = []
        peak_amplitudes = []

        for i in range(1, len(spectrum) - 1):
            if (
                spectrum[i] > spectrum[i - 1]
                and spectrum[i] > spectrum[i + 1]
                and spectrum[i] > self.peak_threshold
            ):
                peak_indices.append(i)
                peak_frequencies.append(frequencies[i])
                peak_amplitudes.append(spectrum[i])

        return {
            "indices": peak_indices,
            "frequencies": peak_frequencies,
            "amplitudes": peak_amplitudes,
        }

    def _analyze_damping(self, response: np.ndarray) -> Dict[str, Any]:
        """
        Analyze damping in the system response.

        Physical Meaning:
            Computes damping rates for collective modes
            from the temporal decay of the response.
        """
        # Analyze each particle's response
        damping_rates = []
        decay_times = []

        for i in range(response.shape[0]):
            # Fit exponential decay
            t = np.arange(response.shape[1]) * self.dt
            y = np.abs(response[i, :])

            # Find decay region
            max_idx = np.argmax(y)
            decay_region = y[max_idx:]
            t_decay = t[max_idx:]

            if len(decay_region) > 1:
                # Fit exponential
                try:
                    log_y = np.log(decay_region + 1e-10)
                    p = np.polyfit(t_decay, log_y, 1)
                    gamma = -p[0]  # Damping rate
                    damping_rates.append(gamma)
                    decay_times.append(1.0 / gamma if gamma > 0 else np.inf)
                except:
                    damping_rates.append(0.0)
                    decay_times.append(np.inf)
            else:
                damping_rates.append(0.0)
                decay_times.append(np.inf)

        return {
            "damping_rates": damping_rates,
            "decay_times": decay_times,
            "mean_damping": np.mean(damping_rates),
            "mean_decay_time": np.mean(decay_times),
        }

    def _compute_participation_ratios(self, response: np.ndarray) -> np.ndarray:
        """
        Compute participation ratios for collective modes.

        Physical Meaning:
            Calculates how much each particle participates
            in the collective response.
        """
        # Compute participation from response amplitudes
        participation = np.zeros(response.shape[0])

        for i in range(response.shape[0]):
            # Participation based on response amplitude
            response_amplitude = np.max(np.abs(response[i, :]))
            participation[i] = response_amplitude

        # Normalize
        if np.sum(participation) > 0:
            participation = participation / np.sum(participation)

        return participation

    def _compute_quality_factors(
        self, peaks: Dict[str, Any], damping_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute quality factors for collective modes.

        Physical Meaning:
            Calculates quality factors Q = ω/γ for
            collective modes.
        """
        peak_frequencies = peaks["frequencies"]
        damping_rates = damping_analysis["damping_rates"]

        quality_factors = []
        for freq in peak_frequencies:
            # Find corresponding damping rate
            if damping_rates:
                gamma = np.mean(damping_rates)
                Q = freq / gamma if gamma > 0 else np.inf
            else:
                Q = np.inf
            quality_factors.append(Q)

        return np.array(quality_factors)

    def _solve_dispersion_equation(self, k: float) -> float:
        """
        Solve dispersion equation for given wave vector.

        Physical Meaning:
            Solves the dispersion equation ω²(k) = ω₀² + c²k²
            for the given wave vector k.
        """
        # Get system parameters
        modes = self.system.find_collective_modes()
        base_frequency = np.mean(modes["frequencies"])

        # Dispersion relation: ω² = ω₀² + c²k²
        c = 1.0  # Sound speed
        omega_squared = (2 * np.pi * base_frequency) ** 2 + c**2 * k**2

        return np.sqrt(omega_squared) / (2 * np.pi)

    def _compute_group_velocity(self, k: float, omega: float) -> float:
        """
        Compute group velocity v_g = dω/dk.

        Physical Meaning:
            Calculates the group velocity for the
            given wave vector and frequency.
        """
        # Numerical derivative
        dk = 0.01
        omega_plus = self._solve_dispersion_equation(k + dk)
        omega_minus = self._solve_dispersion_equation(k - dk)

        v_g = (omega_plus - omega_minus) / (2 * dk)

        return v_g

    def _fit_dispersion_relation(
        self, k_values: np.ndarray, frequencies: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fit dispersion relation to computed data.

        Physical Meaning:
            Fits the dispersion relation ω²(k) = ω₀² + c²k²
            to the computed frequency data.
        """
        # Fit quadratic relation
        frequencies = np.array(frequencies)
        omega_squared = (2 * np.pi * frequencies) ** 2
        p = np.polyfit(k_values, omega_squared, 1)

        # Extract parameters
        omega_0_squared = p[1]  # Intercept
        c_squared = p[0]  # Slope

        omega_0 = np.sqrt(omega_0_squared) / (2 * np.pi)
        c = np.sqrt(c_squared)

        # Compute R²
        omega_fit = np.sqrt(omega_0_squared + c_squared * k_values) / (2 * np.pi)
        r_squared = 1 - np.sum((frequencies - omega_fit) ** 2) / np.sum(
            (frequencies - np.mean(frequencies)) ** 2
        )

        return {"omega_0": omega_0, "c": c, "r_squared": r_squared, "coefficients": p}

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data for this model.

        Physical Meaning:
            Performs comprehensive analysis of collective excitations,
            including response analysis and dispersion relations.

        Args:
            data (Any): Input data to analyze (external field)

        Returns:
            Dict: Analysis results including response and dispersion
        """
        # Create external field if not provided
        if data is None:
            external_field = np.random.randn(*self.domain.shape) * 0.1
        else:
            external_field = data

        # Excite system
        response = self.excite_system(external_field)

        # Analyze response
        response_analysis = self.analyze_response(response)

        # Compute dispersion relations
        dispersion = self.compute_dispersion_relations()

        return {
            "response": response,
            "response_analysis": response_analysis,
            "dispersion": dispersion,
        }
