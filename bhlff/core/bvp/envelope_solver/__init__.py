"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Envelope solver package for BVP envelope equation.

This package provides modular components for solving the 7D BVP
envelope equation, including residual computation, Jacobian calculation,
and Newton-Raphson system solving.

Physical Meaning:
    Provides modular components for solving the nonlinear 7D envelope
    equation ∇·(κ(|a|)∇a) + k₀²χ(|a|)a = s(x,φ,t) using advanced
    numerical methods in M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.

Mathematical Foundation:
    Implements Newton-Raphson method with line search and regularization
    for robust solution of nonlinear 7D envelope equations with gradients
    in all 7 dimensions.

Example:
    >>> from .envelope_solver_core import EnvelopeSolverCore
    >>> from .residual_computer import ResidualComputer
    >>> core = EnvelopeSolverCore(domain, config)
    >>> residual = core.compute_residual(envelope, source)
"""

from .envelope_solver_core import EnvelopeSolverCore
# ResidualComputer and JacobianComputer removed - functionality moved to AbstractSolverCore
from .newton_solver import NewtonSolver
from .gradient_computer import GradientComputer

__all__ = [
    "EnvelopeSolverCore",
    "NewtonSolver",
    "GradientComputer",
]
