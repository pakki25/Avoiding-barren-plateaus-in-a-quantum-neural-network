"""Quantum neural network used throughout this project.

Track 2 (Quantum machine learning), Problem 4: *A quantum neural network that
avoids barren plateaus*.

The ansatz is a hardware-efficient circuit: ``layers`` repetitions of an RY
rotation on every qubit followed by a chain of CNOTs between neighbours.  The
cost function is the **global** cost

    C(theta) = 1 - |<0...0|psi(theta)>|^2 ,

and gradients are obtained with the parameter-shift rule

    dC/dtheta_k = [C(theta_k + pi/2) - C(theta_k - pi/2)] / 2 .

Why a global cost: a global observable that touches every qubit is one of the
standard causes of a barren plateau (Cerezo et al., *Variational quantum
algorithms: a review of methods and the path to quantum advantage*, Nature
Reviews Physics 3, 625 (2021)).  Problem 4 asks us to demonstrate the plateau
and then mitigate it, so measuring the plateau of a global cost is the
interesting baseline.

Everything in this module is an exact statevector simulation except
``cost_aer``, which is a physical circuit simulation with a finite number of
shots (Aer).  No statistical shortcut is used anywhere: the cost and its
gradient are evaluated on the simulated quantum state, not from a fitted
formula.
"""

from __future__ import annotations

import os

# qiskit-aer builds its own OpenMP runtime, which hard-crashes (segmentation
# fault) at import time on some Windows machines when it is allowed to spawn the
# default number of threads.  A single thread loads and runs reliably, and it
# does not change any simulated result.  Set this before importing qiskit_aer.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

#: Circuit depth used everywhere unless stated otherwise.
LAYERS = 3

#: Default number of measurement shots for shot-noise experiments.
SHOTS = 1024


def create_qnn(n: int, params: np.ndarray, layers: int = LAYERS, measure: bool = False) -> QuantumCircuit:
    """Build the hardware-efficient ansatz for ``n`` qubits.

    ``params`` has shape ``(layers, n)``.

    Note: the last rotation layer is deliberately not followed by a CNOT
    layer.  A CNOT layer maps the computational basis onto itself and satisfies
    ``<0...0| U_CNOT = <0...0|``, so it cannot change the cost used here - which
    is why this circuit and the one with a trailing CNOT layer (used for the Aer
    and IBM hardware runs) give exactly the same cost.  ``tests/`` checks this.
    """
    params = np.asarray(params, dtype=float)
    if params.shape != (layers, n):
        raise ValueError(f"params must have shape ({layers}, {n}), got {params.shape}")

    qc = QuantumCircuit(n)
    for layer in range(layers):
        for qubit in range(n):
            qc.ry(params[layer, qubit], qubit)
        if layer < layers - 1:
            for qubit in range(n - 1):
                qc.cx(qubit, qubit + 1)
    if measure:
        qc.measure_all()
    return qc


def cost_statevector(n: int, params: np.ndarray, layers: int = LAYERS) -> float:
    """Exact (noiseless, infinite-shot) global cost ``C = 1 - P(|0...0>)``."""
    state = Statevector.from_instruction(create_qnn(n, params, layers))
    return float(1.0 - abs(state.data[0]) ** 2)


def probability_zero_state(n: int, params: np.ndarray, layers: int = LAYERS) -> float:
    """Exact probability of measuring the all-zero bit string."""
    return 1.0 - cost_statevector(n, params, layers)


def cost_aer(
    n: int,
    params: np.ndarray,
    shots: int = SHOTS,
    seed: int | None = None,
    simulator: AerSimulator | None = None,
    layers: int = LAYERS,
) -> float:
    """Cost estimated from ``shots`` measurements on the Aer simulator.

    This is a physical circuit simulation with sampling noise: the circuit is
    built, measured and counted.  It is *not* a statistical shortcut.
    """
    qc = create_qnn(n, params, layers, measure=True)
    simulator = simulator if simulator is not None else AerSimulator()
    result = simulator.run(qc, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()

    zero_state = "0" * n
    return float(1.0 - counts.get(zero_state, 0) / shots)


def parameter_shift_gradient(
    n: int,
    params: np.ndarray,
    index: tuple[int, int] = (0, 0),
    cost_function=cost_statevector,
    layers: int = LAYERS,
) -> float:
    """Parameter-shift gradient of one parameter (``index = (layer, qubit)``).

    For an RY rotation the shift is exactly ``pi/2``, so this is not an
    approximation: it is the exact derivative of the global cost.
    """
    layer, qubit = index
    plus = np.array(params, dtype=float)
    minus = np.array(params, dtype=float)
    plus[layer, qubit] += np.pi / 2
    minus[layer, qubit] -= np.pi / 2
    return float((cost_function(n, plus, layers) - cost_function(n, minus, layers)) / 2)


def all_gradients(n: int, params: np.ndarray, cost_function=cost_statevector, layers: int = LAYERS) -> np.ndarray:
    """Parameter-shift gradient for every parameter, shape ``(layers, n)``."""
    params = np.asarray(params, dtype=float)
    gradient = np.zeros_like(params)
    for layer in range(params.shape[0]):
        for qubit in range(params.shape[1]):
            gradient[layer, qubit] = parameter_shift_gradient(n, params, (layer, qubit), cost_function, layers)
    return gradient


def initialize_parameters(n: int, initialization: str, layers: int = LAYERS, rng=np.random) -> np.ndarray:
    """Draw initial parameters for one of the two strategies compared here.

    ``random``        - angles uniform on ``[0, 2*pi)``: a generic, random point.
    ``near_identity`` - angles uniform on ``[-0.1, 0.1]``: a near-identity block,
                        the mitigation strategy of Problem 4.
    """
    if initialization == "random":
        return rng.uniform(0.0, 2.0 * np.pi, size=(layers, n))
    if initialization == "near_identity":
        return rng.uniform(-0.1, 0.1, size=(layers, n))
    raise ValueError("initialization must be 'random' or 'near_identity'")


def train_qnn(
    n: int,
    initialization: str,
    iterations: int = 100,
    learning_rate: float = 0.1,
    layers: int = LAYERS,
    cost_function=cost_statevector,
    rng=np.random,
) -> tuple[np.ndarray, np.ndarray]:
    """Plain gradient descent on the global cost.

    Returns ``(cost_history, params)`` where ``cost_history[k]`` is the cost
    *after* ``k`` updates, so ``cost_history[0]`` is the initial cost and
    ``cost_history[iterations]`` is the final one.
    """
    params = initialize_parameters(n, initialization, layers, rng)

    cost_history = [float(cost_function(n, params, layers))]
    for _ in range(iterations):
        gradients = all_gradients(n, params, cost_function, layers)
        params = params - learning_rate * gradients
        cost_history.append(float(cost_function(n, params, layers)))

    return np.array(cost_history), params
