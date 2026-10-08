import numpy as np

from ddqaoa import ConvergenceMonitor, build_problem, generate_instance, interp_transfer, run_ddqaoa


def test_transfer_p1_uses_adiabatic_coefficients():
    g, b = interp_transfer([0.5], [0.4])
    assert np.allclose(g, [0.5, 0.6]) and np.allclose(b, [0.4, 0.32])


def test_transfer_keeps_endpoints_and_grows_by_one():
    for p in (2, 3, 4, 6):
        g, b = interp_transfer(np.linspace(0.1, 0.9, p), np.linspace(0.9, 0.1, p))
        assert len(g) == len(b) == p + 1
        assert np.isclose(g[0], 0.1) and np.isclose(g[-1], 0.9)


def test_monitor_detects_plateau():
    m = ConvergenceMonitor(eps=1e-4, sigma=0.0, k=5)
    fired = [m.update(e) for e in [5, 4, 3, 3, 3, 3, 3, 3, 3]]
    assert fired.index(True) == 7  # 5 non-improving steps after the last improvement


def test_ddqaoa_smoke():
    inst = generate_instance(n_nodes=4, seed=1)
    _, problem = build_problem(inst)
    res = run_ddqaoa(problem, p_max=3, max_steps=300, stepsize=0.05, k=20)
    assert 1 <= res.final_depth <= 3
    assert 0.0 <= res.approx_ratio <= 1.0
    assert res.cumulative_cnots > 0
