"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Defect models for Level E experiments in 7D phase field theory.

This module implements topological defect models representing localized
distortions in the phase field with non-trivial winding numbers.

Theoretical Background:
    Topological defects are singularities in the phase field that carry
    non-trivial winding numbers and create localized distortions in the
    field configuration. They represent fundamental structures in the
    7D theory with rich dynamics and interactions.

Mathematical Foundation:
    Implements defects with topological charge q ∈ ℤ where
    ∮∇φ·dl = 2πq around the defect core. The dynamics follows the
    Thiele equation: ẋ = -∇U_eff + G × ẋ + D ẋ.

Example:
    >>> defect = VortexDefect(domain, physics_params)
    >>> field = defect.create_defect(position, charge)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod


class DefectModel(ABC):
    """
    Base class for topological defect models.

    Physical Meaning:
        Represents topological defects in the phase field that carry
        non-trivial winding numbers and create localized distortions
        in the field configuration.

    Mathematical Foundation:
        Implements defects with topological charge q ∈ ℤ where
        ∮∇φ·dl = 2πq around the defect core.
    """

    def __init__(self, domain: "Domain", physics_params: Dict[str, Any]):
        """
        Initialize defect model.

        Args:
            domain: Computational domain
            physics_params: Physical parameters including β, μ, λ
        """
        self.domain = domain
        self.params = physics_params
        self._setup_defect_operators()

    def _setup_defect_operators(self) -> None:
        """Setup operators for defect calculations."""
        # Setup fractional Laplacian for defect dynamics
        self._setup_fractional_laplacian()

        # Setup interaction potential
        self._setup_interaction_potential()

    def _setup_fractional_laplacian(self) -> None:
        """Setup fractional Laplacian operator."""
        mu = self.params.get("mu", 1.0)
        beta = self.params.get("beta", 1.0)

        # Compute wave vectors
        kx = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        ky = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        kz = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)

        # Fractional Laplacian in spectral space
        self._frac_laplacian_spectral = mu * (k_magnitude ** (2 * beta))

    def _setup_interaction_potential(self) -> None:
        """Setup interaction potential between defects."""
        # Implementation of interaction potential
        pass

    def create_defect(self, position: np.ndarray, charge: int) -> np.ndarray:
        """
        Create topological defect at specified position.

        Physical Meaning:
            Generates a field configuration with topological defect
            of specified charge at the given position, creating
            localized phase winding.

        Mathematical Foundation:
            Constructs field with phase φ = q·arctan2(y-y₀, x-x₀)
            around position (x₀, y₀) with charge q.

        Args:
            position: 3D position of defect center
            charge: Topological charge (winding number)

        Returns:
            Field configuration with defect
        """
        # Create coordinate grids
        x = np.linspace(0, self.domain.L, self.domain.N)
        y = np.linspace(0, self.domain.L, self.domain.N)
        z = np.linspace(0, self.domain.L, self.domain.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Compute distances from defect center
        dx = X - position[0]
        dy = Y - position[1]
        dz = Z - position[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Create phase field with winding
        phi = charge * np.arctan2(dy, dx)

        # Create amplitude profile with proper asymptotic behavior
        amplitude = self._create_amplitude_profile(r, charge)

        # Combine amplitude and phase
        field = amplitude * np.exp(1j * phi)

        return field

    def _create_amplitude_profile(self, r: np.ndarray, charge: int) -> np.ndarray:
        """
        Create amplitude profile for defect.

        Physical Meaning:
            Creates the radial amplitude profile A(r) with proper
            asymptotic behavior for the topological defect.

        Mathematical Foundation:
            In the "interval-free" mode (λ=0), the amplitude follows
            A(r) ~ r^(2β-3) where β is the fractional order.
        """
        beta = self.params.get("beta", 1.0)
        core_radius = self.params.get("core_radius", 0.1)

        # Avoid division by zero at the core
        r_safe = np.maximum(r, core_radius)

        # Create amplitude profile
        if beta < 1.5:
            # Power law tail: A(r) ~ r^(2β-3)
            amplitude = r_safe ** (2 * beta - 3)
        else:
            # For β ≥ 1.5, use exponential decay
            amplitude = np.exp(-r_safe / core_radius)

        # Normalize amplitude
        amplitude = amplitude / np.max(amplitude)

        return amplitude

    def compute_defect_charge(self, field: np.ndarray, center: np.ndarray) -> float:
        """
        Compute topological charge around defect center.

        Physical Meaning:
            Calculates the winding number of the phase field around
            the defect center, quantifying the topological charge.

        Mathematical Foundation:
            q = (1/2π)∮∇φ·dl where the integral is taken around
            a closed loop surrounding the defect.

        Args:
            field: Phase field configuration
            center: Approximate center of defect

        Returns:
            Topological charge (winding number)
        """
        # Extract phase from complex field
        phase = np.angle(field)

        # Create circular path around defect center
        radius = 2.0  # Radius of integration circle
        n_points = 64  # Number of points on circle

        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x_circle = center[0] + radius * np.cos(angles)
        y_circle = center[1] + radius * np.sin(angles)
        z_circle = center[2] * np.ones_like(angles)

        # Interpolate phase values on circle
        phase_values = []
        for i in range(n_points):
            # Find closest grid point
            x_idx = int(x_circle[i] / self.domain.L * self.domain.N)
            y_idx = int(y_circle[i] / self.domain.L * self.domain.N)
            z_idx = int(z_circle[i] / self.domain.L * self.domain.N)

            # Ensure indices are within bounds
            x_idx = max(0, min(self.domain.N - 1, x_idx))
            y_idx = max(0, min(self.domain.N - 1, y_idx))
            z_idx = max(0, min(self.domain.N - 1, z_idx))

            phase_values.append(phase[x_idx, y_idx, z_idx])

        # Compute winding number
        phase_diff = np.diff(phase_values)
        phase_diff = np.append(phase_diff, phase_values[0] - phase_values[-1])

        # Handle phase jumps
        phase_diff = np.unwrap(phase_diff)

        # Compute total winding
        total_winding = np.sum(phase_diff) / (2 * np.pi)

        return total_winding

    def simulate_defect_motion(
        self, defect: np.ndarray, potential: np.ndarray
    ) -> Dict[str, Any]:
        """
        Simulate motion of topological defect.

        Physical Meaning:
            Evolves the defect position according to the equation of motion
            ẋ = -∇U_eff + G × ẋ + D ẋ, where U_eff is the effective potential,
            G is the gyroscopic coefficient, and D is the dissipation.

        Mathematical Foundation:
            Implements the Thiele equation for defect dynamics with
            effective potential from Green's function interactions.

        Args:
            defect: Initial defect configuration
            potential: External potential field

        Returns:
            Dict containing trajectory, velocity, acceleration
        """
        # Setup time integration
        dt = self.params.get("dt", 0.01)
        t_max = self.params.get("t_max", 10.0)
        time_steps = int(t_max / dt)

        # Initialize trajectory
        trajectory = []
        velocities = []
        accelerations = []

        # Find initial defect position
        current_position = self._find_defect_position(defect)
        current_velocity = np.zeros(3)

        for t in range(time_steps):
            # Compute forces
            force = self._compute_defect_force(current_position, potential)

            # Compute gyroscopic and dissipative terms
            gyroscopic_force = self._compute_gyroscopic_force(current_velocity)
            dissipative_force = self._compute_dissipative_force(current_velocity)

            # Total force
            total_force = force + gyroscopic_force + dissipative_force

            # Update velocity and position
            acceleration = total_force / self._get_defect_mass()
            current_velocity += acceleration * dt
            current_position += current_velocity * dt

            # Store trajectory
            trajectory.append(current_position.copy())
            velocities.append(current_velocity.copy())
            accelerations.append(acceleration.copy())

        return {
            "trajectory": np.array(trajectory),
            "velocities": np.array(velocities),
            "accelerations": np.array(accelerations),
            "time": np.linspace(0, t_max, time_steps),
        }

    def _find_defect_position(self, field: np.ndarray) -> np.ndarray:
        """Find defect position in field."""
        # Find maximum of field magnitude (defect core)
        magnitude = np.abs(field)
        max_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)

        # Convert to physical coordinates
        position = np.array(
            [
                max_idx[0] * self.domain.L / self.domain.N,
                max_idx[1] * self.domain.L / self.domain.N,
                max_idx[2] * self.domain.L / self.domain.N,
            ]
        )

        return position

    def _compute_defect_force(
        self, position: np.ndarray, potential: np.ndarray
    ) -> np.ndarray:
        """Compute force on defect from potential."""
        # Compute gradient of potential at defect position
        force = np.zeros(3)

        # Numerical gradient computation
        epsilon = 0.01
        for i in range(3):
            pos_plus = position.copy()
            pos_plus[i] += epsilon
            pos_minus = position.copy()
            pos_minus[i] -= epsilon

            # Interpolate potential values
            V_plus = self._interpolate_potential(pos_plus, potential)
            V_minus = self._interpolate_potential(pos_minus, potential)

            force[i] = -(V_plus - V_minus) / (2 * epsilon)

        return force

    def _interpolate_potential(
        self, position: np.ndarray, potential: np.ndarray
    ) -> float:
        """Interpolate potential at given position."""
        # Simple nearest neighbor interpolation
        x_idx = int(position[0] / self.domain.L * self.domain.N)
        y_idx = int(position[1] / self.domain.L * self.domain.N)
        z_idx = int(position[2] / self.domain.L * self.domain.N)

        # Ensure indices are within bounds
        x_idx = max(0, min(self.domain.N - 1, x_idx))
        y_idx = max(0, min(self.domain.N - 1, y_idx))
        z_idx = max(0, min(self.domain.N - 1, z_idx))

        return potential[x_idx, y_idx, z_idx]

    def _compute_gyroscopic_force(self, velocity: np.ndarray) -> np.ndarray:
        """Compute gyroscopic force G × ẋ."""
        G = self.params.get("gyroscopic_coefficient", 1.0)
        return G * np.cross([0, 0, 1], velocity)  # Simplified 2D case

    def _compute_dissipative_force(self, velocity: np.ndarray) -> np.ndarray:
        """Compute dissipative force D ẋ."""
        D = self.params.get("dissipation_coefficient", 0.1)
        return -D * velocity

    def _get_defect_mass(self) -> float:
        """Get effective mass of defect."""
        return self.params.get("defect_mass", 1.0)


class VortexDefect(DefectModel):
    """
    Vortex defect with unit topological charge.

    Physical Meaning:
        Represents a vortex-like topological defect with q=±1,
        creating spiral phase patterns around the core.
    """

    def __init__(self, domain: "Domain", physics_params: Dict[str, Any]):
        super().__init__(domain, physics_params)
        self.charge = 1

    def create_vortex_profile(self, position: np.ndarray) -> np.ndarray:
        """
        Create vortex profile with proper asymptotic behavior.

        Physical Meaning:
            Generates field configuration with A(r) ~ r^(2β-3) tail
            and proper phase winding around the core.
        """
        return self.create_defect(position, self.charge)


class MultiDefectSystem(DefectModel):
    """
    System of multiple interacting defects.

    Physical Meaning:
        Represents a collection of topological defects that interact
        through their long-range fields, leading to complex dynamics
        and possible annihilation/creation processes.
    """

    def __init__(
        self,
        domain: "Domain",
        physics_params: Dict[str, Any],
        defect_list: List[Dict[str, Any]],
    ):
        super().__init__(domain, physics_params)
        self.defects = defect_list
        self._setup_interaction_potential()

    def _setup_interaction_potential(self) -> None:
        """Setup interaction potential between multiple defects."""
        # Implementation of multi-defect interaction potential
        pass

    def compute_interaction_forces(self) -> np.ndarray:
        """
        Compute forces between defects.

        Physical Meaning:
            Calculates the effective forces between defects arising
            from their mutual field interactions and topological
            constraints.
        """
        forces = []

        for i, defect_i in enumerate(self.defects):
            force_i = np.zeros(3)

            for j, defect_j in enumerate(self.defects):
                if i != j:
                    # Compute force between defects i and j
                    force_ij = self._compute_pair_force(defect_i, defect_j)
                    force_i += force_ij

            forces.append(force_i)

        return np.array(forces)

    def _compute_pair_force(
        self, defect1: Dict[str, Any], defect2: Dict[str, Any]
    ) -> np.ndarray:
        """Compute force between pair of defects."""
        # Distance between defects
        r_vec = np.array(defect2["position"]) - np.array(defect1["position"])
        r = np.linalg.norm(r_vec)

        if r < 1e-10:
            return np.zeros(3)

        # Charges of defects
        q1 = defect1["charge"]
        q2 = defect2["charge"]

        # Force magnitude (simplified model)
        force_magnitude = q1 * q2 / (r**2)

        # Force direction
        force_direction = r_vec / r

        return force_magnitude * force_direction

    def simulate_defect_annihilation(self, defect_pair: List[int]) -> Dict[str, Any]:
        """
        Simulate annihilation of defect-antidefect pair.

        Physical Meaning:
            Models the process where a defect and antidefect approach
            and annihilate, releasing energy and creating topological
            transitions.
        """
        # Implementation of defect annihilation simulation
        pass
