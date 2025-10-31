"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA vs CPU parity tests for Level F multi-particle potential.

These tests validate that the CUDA-accelerated block-based analyzer
produces results consistent with the CPU analyzer on a small domain,
and that the system falls back to CPU when CUDA is unavailable.
"""

import numpy as np
import pytest

from bhlff.models.level_f.multi_particle_potential import (
    MultiParticlePotentialAnalyzer as CPUAnalyzer,
)
from bhlff.models.level_f.multi_particle_system import MultiParticleSystem
from bhlff.models.level_f.multi_particle.data_structures import Particle
from bhlff.core.domain import Domain


def _cuda_available() -> bool:
    try:
        import cupy as cp  # noqa: F401

        return True
    except Exception:
        return False


@pytest.fixture
def domain() -> Domain:
    return Domain(L=20.0, N=16, N_phi=4, N_t=8, T=1.0, dimensions=7)


@pytest.fixture
def particles():
    return [
        Particle(position=np.array([5.0, 10.0, 10.0]), charge=1, phase=0.0),
        Particle(position=np.array([15.0, 10.0, 10.0]), charge=-1, phase=np.pi),
    ]


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_cpu_parity_small_domain(domain: Domain, particles):
    cpu_an = CPUAnalyzer(domain, particles, interaction_range=5.0)
    cpu_potential = cpu_an.compute_effective_potential()

    # Build CUDA analyzer indirectly via system facade
    system = MultiParticleSystem(
        domain, particles, interaction_range=5.0, use_cuda=True
    )
    gpu_potential = system.compute_effective_potential()

    assert isinstance(cpu_potential, np.ndarray)
    assert isinstance(gpu_potential, np.ndarray)
    assert cpu_potential.shape == gpu_potential.shape

    # Step potentials should match exactly on grid for small sizes
    # Allow a tiny tolerance for dtype conversions
    assert np.allclose(cpu_potential, gpu_potential, atol=1e-12)


def test_cpu_fallback_when_no_cuda(domain: Domain, particles, monkeypatch):
    # Force CPU path by disabling CUDA through import error
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore
        if name == "cupy":
            raise ImportError("no CUDA in test env")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    system = MultiParticleSystem(
        domain, particles, interaction_range=5.0, use_cuda=True
    )

    potential = system.compute_effective_potential()
    assert isinstance(potential, np.ndarray)
    assert potential.shape == (domain.N, domain.N, domain.N)
