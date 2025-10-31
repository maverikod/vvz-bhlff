"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

CUDA parity and fallback tests for CollectiveExcitations (Level F).
"""

import numpy as np
import pytest

from bhlff.models.level_f.collective import CollectiveExcitations
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
def test_collective_cuda_cpu_parity(domain: Domain, particles):
    system = MultiParticleSystem(
        domain, particles, interaction_range=5.0, use_cuda=False
    )
    params = {
        "type": "harmonic",
        "frequency_range": [0.2, 0.2],
        "amplitude": 0.1,
        "duration": 1.0,
        "dt": 0.01,
    }
    model_cpu = CollectiveExcitations(system, params)

    field = np.ones((domain.N, domain.N, domain.N), dtype=np.float64)
    resp_cpu = model_cpu.excite_system(field)
    analysis_cpu = model_cpu.analyze_response(resp_cpu)
    disp_cpu = model_cpu.compute_dispersion_relations()

    system_gpu = MultiParticleSystem(
        domain, particles, interaction_range=5.0, use_cuda=True
    )
    model_gpu = CollectiveExcitations(system_gpu, params)
    resp_gpu = model_gpu.excite_system(field)
    analysis_gpu = model_gpu.analyze_response(resp_gpu)
    disp_gpu = model_gpu.compute_dispersion_relations()

    assert resp_cpu.shape == resp_gpu.shape
    assert analysis_cpu["spectrum"].shape == analysis_gpu["spectrum"].shape
    assert disp_cpu["k_values"].shape == disp_gpu["k_values"].shape


def test_collective_cpu_fallback(domain: Domain, particles, monkeypatch):
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
    params = {
        "type": "harmonic",
        "frequency_range": [0.2, 0.2],
        "amplitude": 0.1,
        "duration": 0.5,
        "dt": 0.01,
    }
    model = CollectiveExcitations(system, params)

    field = np.ones((domain.N, domain.N, domain.N), dtype=np.float64)
    resp = model.excite_system(field)
    analysis = model.analyze_response(resp)
    disp = model.compute_dispersion_relations()

    assert resp.ndim == 2 and resp.shape[0] == len(system.particles)
    assert "spectrum" in analysis and analysis["spectrum"].ndim >= 1
    assert "k_values" in disp and disp["k_values"].ndim == 1
