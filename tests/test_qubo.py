import numpy as np
import pytest

from ddqaoa import build_problem, decode, generate_instance, solve_classical
from ddqaoa.circuits import index_to_bits


@pytest.mark.parametrize("seed", range(5))
def test_ground_state_is_classical_optimum(seed):
    inst = generate_instance(n_nodes=4, seed=seed)
    q, problem = build_problem(inst)
    path, cost = solve_classical(inst)

    bits = index_to_bits(int(problem.ground_states[0]), problem.n_qubits)
    chosen = set(decode(bits, q))
    assert chosen == set(zip(path, path[1:]))
    # Ising energy (unscaled) equals the QUBO energy, which equals the path cost.
    assert np.isclose(problem.e_min * problem.scale, cost)
    assert np.isclose(q.energy(bits), cost)


def test_qubo_and_ising_agree_on_random_bitstrings():
    inst = generate_instance(n_nodes=4, seed=42)
    q, problem = build_problem(inst, normalize=False)
    rng = np.random.default_rng(0)
    for idx in rng.integers(0, 2**problem.n_qubits, size=50):
        bits = index_to_bits(int(idx), problem.n_qubits)
        assert np.isclose(q.energy(bits), problem.diag[idx])
