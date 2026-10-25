import pytest
import numpy as np
from solver import AdvectionSolver, profile_gaussian, profile_step


def test_courant_one_exactness():
    solver = AdvectionSolver(L=20.0, N=100, u=1.0)
    s_num, _ = solver.solve("upwind", profile_gaussian, Co=1.0, t_final=5.0)
    s_exact = solver.exact_solution(profile_gaussian, t=5.0)
    assert np.allclose(s_num, s_exact, atol=1e-12)


def test_mass_conservation_upwind():
    solver = AdvectionSolver(L=20.0, N=150, u=1.0)
    s0 = profile_gaussian(solver.x)
    initial_mass = np.sum(s0) * solver.dx

    s_num, _ = solver.solve("upwind", profile_gaussian, Co=0.6, t_final=6.0)
    final_mass = np.sum(s_num) * solver.dx
    assert pytest.approx(initial_mass, rel=1e-5) == final_mass


def test_mass_conservation_lax_wendroff():
    solver = AdvectionSolver(L=20.0, N=150, u=1.0)
    s0 = profile_step(solver.x)
    initial_mass = np.sum(s0) * solver.dx

    s_num, _ = solver.solve("lax_wendroff", profile_step, Co=0.7, t_final=4.0)
    final_mass = np.sum(s_num) * solver.dx
    assert pytest.approx(initial_mass, rel=1e-5) == final_mass


def test_ftcs_instability():
    solver = AdvectionSolver(L=20.0, N=100, u=1.0)
    s_num, _ = solver.solve("ftcs", profile_gaussian, Co=0.5, t_final=12.0)
    assert np.max(np.abs(s_num)) > 2.0 or np.any(np.isnan(s_num))


def test_upwind_instability_cfl_violation():
    solver = AdvectionSolver(L=20.0, N=100, u=1.0)
    s_num, _ = solver.solve("upwind", profile_step, Co=1.3, t_final=8.0)
    assert np.max(np.abs(s_num)) > 2.0 or np.any(np.isnan(s_num))