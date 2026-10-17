"""Checks on the circuit, the cost and the gradient against exact references.

The reference here is *analytic* or *independent*: closed-form amplitudes for
one and two qubits, a central finite difference for the gradient, and a manual
statevector built with NumPy matrix products instead of Qiskit.  A wrong gate or
a wrong parameter-shift convention would fail these tests.
"""

import numpy as np

from qnn import (
    LAYERS,
    all_gradients,
    cost_aer,
    cost_statevector,
    create_qnn,
    parameter_shift_gradient,
)


def test_cost_is_exactly_zero_for_the_identity_circuit():
    """<0...0|0...0> = 1, so the global cost must be exactly 0 at zero angles."""
    for n in (2, 4, 6):
        assert cost_statevector(n, np.zeros((LAYERS, n))) == 0.0


def test_cost_of_a_single_ry_layer_matches_the_closed_form():
    """One qubit, one RY(theta): C = 1 - cos^2(theta/2) = sin^2(theta/2)."""
    for theta in (0.0, 0.3, 1.1, 2.7, np.pi):
        expected = np.sin(theta / 2) ** 2
        got = cost_statevector(1, np.array([[theta]]), layers=1)
        assert abs(got - expected) < 1e-12


def test_gradient_matches_the_analytic_derivative_for_one_qubit():
    """d/dtheta sin^2(theta/2) = sin(theta)/2, exactly, by parameter shift."""
    for theta in (0.0, 0.4, 1.3, 2.9):
        expected = np.sin(theta) / 2
        got = parameter_shift_gradient(1, np.array([[theta]]), (0, 0), layers=1)
        assert abs(got - expected) < 1e-12


def test_gradient_matches_a_finite_difference():
    """Central finite differences are an independent numerical reference."""
    rng = np.random.default_rng(7)
    for n in (2, 3, 5):
        params = rng.uniform(0, 2 * np.pi, size=(LAYERS, n))
        for index in ((0, 0), (1, n - 1), (LAYERS - 1, 0)):
            analytic = parameter_shift_gradient(n, params, index, layers=LAYERS)
            step = 1e-6
            plus, minus = params.copy(), params.copy()
            plus[index] += step
            minus[index] -= step
            numeric = (cost_statevector(n, plus) - cost_statevector(n, minus)) / (2 * step)
            assert abs(analytic - numeric) < 1e-6


def test_cost_matches_an_independent_numpy_simulation():
    """Rebuild the 2-qubit state with plain NumPy and compare the amplitude."""
    rng = np.random.default_rng(11)
    params = rng.uniform(0, 2 * np.pi, size=(LAYERS, 2))
    n = 2

    def ry(theta):
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])

    def cnot():
        matrix = np.zeros((4, 4))
        matrix[0, 0] = matrix[1, 1] = 1.0  # control |0>: identity
        matrix[2, 3] = matrix[3, 2] = 1.0  # control |1>: flip target
        return matrix

    state = np.array([1.0, 0.0, 0.0, 0.0])
    for layer in range(LAYERS):
        state = np.kron(ry(params[layer, 0]), ry(params[layer, 1])) @ state
        if layer < LAYERS - 1:
            state = cnot() @ state

    expected = 1.0 - abs(state[0]) ** 2
    assert abs(cost_statevector(n, params) - expected) < 1e-12


def test_appended_cnot_layer_does_not_change_the_cost():
    """Justifies dropping the trailing CNOT layer: CX cannot change P(|0...0>)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(3)
    for n in (2, 4, 6):
        params = rng.uniform(0, 2 * np.pi, size=(LAYERS, n))
        circuit = create_qnn(n, params)
        for qubit in range(n - 1):
            circuit.cx(qubit, qubit + 1)
        reference = 1.0 - abs(Statevector.from_instruction(circuit).data[0]) ** 2
        assert abs(reference - cost_statevector(n, params)) < 1e-12


def test_cost_stays_in_range_and_all_gradients_has_the_right_shape():
    rng = np.random.default_rng(5)
    for n in (2, 4, 8):
        params = rng.uniform(0, 2 * np.pi, size=(LAYERS, n))
        cost = cost_statevector(n, params)
        assert 0.0 <= cost <= 1.0
        gradient = all_gradients(n, params)
        assert gradient.shape == (LAYERS, n)
        assert np.isfinite(gradient).all()


def test_sampled_cost_approaches_the_exact_cost_with_many_shots():
    """The Aer path is a physical simulation: with many shots it converges."""
    rng = np.random.default_rng(13)
    params = rng.uniform(0, 2 * np.pi, size=(LAYERS, 4))
    exact = cost_statevector(4, params)
    sampled = cost_aer(4, params, shots=200_000, seed=1)
    sigma = np.sqrt(exact * (1 - exact) / 200_000)
    assert abs(sampled - exact) < 5 * sigma + 1e-3


def test_aer_shots_are_conserved():
    """The counts must add up to the number of shots requested."""
    from qiskit_aer import AerSimulator

    rng = np.random.default_rng(17)
    params = rng.uniform(0, 2 * np.pi, size=(LAYERS, 3))
    simulator = AerSimulator()
    circuit = create_qnn(3, params, measure=True)
    counts = simulator.run(circuit, shots=500, seed_simulator=2).result().get_counts()
    assert sum(counts.values()) == 500
