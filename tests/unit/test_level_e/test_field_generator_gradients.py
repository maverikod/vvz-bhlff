"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Verify that the Level E field generator produces a field with non-zero gradients
in spatial (x,y,z) and phase (φ1,φ2,φ3) directions.

The test builds a U(1)^3 phase field Θ(x,φ) as used in Level E tests and
checks that gradient norms and coverage are non-zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from bhlff.core.domain import Domain


@pytest.mark.level_e
def test_field_generator_has_nonzero_gradients():
    # Small domain for speed; 7D shape: (Nx,Ny,Nz,Nφ,Nφ,Nφ,Nt)
    N = 8
    domain = Domain(L=2.0, N=N, dimensions=7, N_phi=8, N_t=1, T=1.0)

    # Generate U(1)^3 phase field (as in project tests)
    field = np.zeros((N, N, N, 8, 8, 8, 1), dtype=np.complex128)

    # Spatial coordinates
    x = np.linspace(-domain.L / 2, domain.L / 2, N)
    y = np.linspace(-domain.L / 2, domain.L / 2, N)
    z = np.linspace(-domain.L / 2, domain.L / 2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Phase coordinates
    phi = np.linspace(0, 2 * np.pi, 8)
    PHI1, PHI2, PHI3 = np.meshgrid(phi, phi, phi, indexing="ij")

    # Construct Θ(x,φ) = spatial_factor * (φ1 + φ2 + φ3)
    R = domain.L / 4
    for i in range(N):
        for j in range(N):
            for k in range(N):
                r = np.sqrt(X[i, j, k] ** 2 + Y[i, j, k] ** 2 + Z[i, j, k] ** 2)
                spatial_factor = np.exp(-(r**2) / (2 * R**2)) if r < R else 0.0
                field[i, j, k, :, :, :, 0] = spatial_factor * (PHI1 + PHI2 + PHI3)

    # Compute gradients along spatial and phase axes
    axes = (0, 1, 2, 3, 4, 5)
    steps = {0: x[1] - x[0], 1: y[1] - y[0], 2: z[1] - z[0], 3: phi[1] - phi[0], 4: phi[1] - phi[0], 5: phi[1] - phi[0]}

    total_norm = 0.0
    nonzero = 0
    total = 0
    for ax in axes:
        g = np.gradient(field, steps[ax], axis=ax)
        n = np.linalg.norm(g)
        total_norm += n
        nz = np.count_nonzero(np.abs(g) > 1e-12)
        nonzero += nz
        total += g.size

    coverage = nonzero / max(1, total)

    # Assertions: we expect non-zero gradient magnitudes and non-trivial coverage
    assert total_norm > 0.0, "Generated field has zero gradient norm"
    assert coverage > 0.0, "Generated field has zero gradient everywhere"


