"""Barren plateaus: how the gradient variance scales with the number of qubits.

Core tasks 1 and 2 of Problem 4.

For each qubit number ``n`` we draw ``TRIALS`` independent random parameter
sets, evaluate one gradient component with the parameter-shift rule and take
the variance of those gradients.  A barren plateau shows up as an
exponentially decreasing variance, so the numbers are fitted with

    Var[dC/dtheta] ~ A * exp(-b * n)

and the decay rate per qubit ``b`` is reported.  ``b = 0`` means "no plateau".

Everything here is an exact statevector simulation: the cost is an exact
amplitude, so the only randomness is in the random parameters.  Both
strategies draw from one legacy NumPy stream seeded with ``SEED`` in a fixed
order (all random initialisations first, then all near-identity ones), which is
what makes the numbers reproducible and lets ``tests/`` compare them with the
values recorded in the Colab notebook (``tests/test_reference_values.py``).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # write figures without a display

import matplotlib.pyplot as plt
import numpy as np

from qnn import LAYERS, cost_statevector, parameter_shift_gradient

SEED = 42
TRIALS = 100
QUBITS = list(range(2, 11))
RESULTS = "results/barren_plateau.json"
FIGURE = "figures/main_result.png"


def draw_parameters(n: int, initialization: str, rng) -> np.ndarray:
    """One initial parameter set of shape ``(layers, n)``."""
    if initialization == "random":
        return rng.uniform(0.0, 2.0 * np.pi, size=(LAYERS, n))
    if initialization == "near_identity":
        return rng.uniform(-0.1, 0.1, size=(LAYERS, n))
    raise ValueError("initialization must be 'random' or 'near_identity'")


def gradient_variance(
    n: int, initialization: str, rng, trials: int = TRIALS, cost_function=cost_statevector
) -> dict:
    """Variance of one gradient component over ``trials`` random parameter sets.

    ``cost_function`` is the cost whose gradient is measured, so the same
    measurement can be repeated for a different cost (see
    ``run_local_cost.py``, which compares the global cost with a local one).
    """
    gradients = np.empty(trials)
    for trial in range(trials):
        params = draw_parameters(n, initialization, rng)
        gradients[trial] = parameter_shift_gradient(n, params, (0, 0), cost_function)

    variance = float(np.var(gradients, ddof=1))
    # Standard error of a sample variance under an approximately Gaussian
    # gradient distribution: SE = Var * sqrt(2 / (T - 1)).
    stderr = variance * np.sqrt(2.0 / (trials - 1))
    return {
        "qubits": n,
        "initialization": initialization,
        "trials": trials,
        "cost": cost_function.__name__,
        "mean_gradient": float(np.mean(gradients)),
        "variance": variance,
        "variance_stderr": float(stderr),
    }


def exponential_fit(qubits, variances, variances_stderr) -> dict:
    """Weighted least-squares fit of ``log Var`` against ``n``.

    Returns the decay rate per qubit ``b`` and the prefactor ``A`` of
    ``Var ~ A * exp(-b * n)``.  The scatter of ``log Var`` is used as weight,
    which is the relative error of the variance.
    """
    qubits = np.asarray(qubits, dtype=float)
    variances = np.asarray(variances, dtype=float)
    log_var = np.log(variances)
    sigma = np.asarray(variances_stderr, dtype=float) / variances  # relative error of Var
    sigma = np.where(sigma > 0, sigma, 1.0)

    slope, intercept = np.polyfit(qubits, log_var, 1, w=1.0 / sigma)

    residual = log_var - (slope * qubits + intercept)
    dof = max(len(qubits) - 2, 1)
    chi2 = float(np.sum((residual / sigma) ** 2))
    scale = max(chi2 / dof, 1.0)
    sxx = float(np.sum((qubits - np.mean(qubits)) ** 2))
    slope_stderr = float(np.sqrt(scale / sxx))

    return {
        "decay_rate_per_qubit": float(-slope),
        "decay_rate_stderr": slope_stderr,
        "prefactor": float(np.exp(intercept)),
        "log_slope": float(slope),
        "log_intercept": float(intercept),
        "chi2_per_dof": chi2 / dof,
    }


def run(trials: int = TRIALS, seed: int = SEED) -> dict:
    """Run both initialisation strategies and fit the decay rates."""
    rng = np.random
    rng.seed(seed)

    random_records = [gradient_variance(n, "random", rng, trials) for n in QUBITS]
    identity_records = [gradient_variance(n, "near_identity", rng, trials) for n in QUBITS]

    random_fit = exponential_fit(
        QUBITS,
        [r["variance"] for r in random_records],
        [r["variance_stderr"] for r in random_records],
    )
    identity_fit = exponential_fit(
        QUBITS,
        [r["variance"] for r in identity_records],
        [r["variance_stderr"] for r in identity_records],
    )

    return {
        "experiment": "gradient variance vs number of qubits",
        "layers": LAYERS,
        "trials": trials,
        "seed": seed,
        "cost": "C(theta) = 1 - P(|0...0>) (global cost)",
        "gradient_component": "dC/dtheta for layer 0, qubit 0, parameter-shift rule",
        "simulator": "Qiskit statevector (exact, noiseless)",
        "random": {"records": random_records, "fit": random_fit},
        "near_identity": {"records": identity_records, "fit": identity_fit},
    }


def plot(data: dict, path: str = FIGURE) -> str:
    """Main-result figure: gradient variance against qubit number, log scale."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for key, label, colour, marker in (
        ("random", "Random initialisation  U[0, 2pi)", "tab:red", "o"),
        ("near_identity", "Near-identity initialisation  U[-0.1, 0.1]", "tab:blue", "s"),
    ):
        records = data[key]["records"]
        variances = np.array([r["variance"] for r in records])
        errors = np.array([r["variance_stderr"] for r in records])
        fit = data[key]["fit"]
        ax.errorbar(
            QUBITS,
            variances,
            yerr=errors,
            marker=marker,
            color=colour,
            linewidth=2,
            capsize=3,
            label=(f"{label}\nfit: Var ~ {fit['prefactor']:.1e} * exp(-{fit['decay_rate_per_qubit']:.3f} n)"),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits $n$", fontsize=12)
    ax.set_ylabel("Variance of one gradient component (log scale)", fontsize=12)
    ax.set_title("Barren plateau and its mitigation: gradient variance vs qubit number", fontsize=13)
    ax.set_xticks(QUBITS)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(results_path: str = RESULTS, figure_path: str = FIGURE) -> dict:
    data = run()
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plot(data, figure_path)

    for key in ("random", "near_identity"):
        fit = data[key]["fit"]
        print(f"{key:14s} decay rate b = {fit['decay_rate_per_qubit']:.3f} "
              f"+/- {fit['decay_rate_stderr']:.3f} per qubit, prefactor {fit['prefactor']:.2e}")
    print(f"wrote {results_path} and {figure_path}")
    return data


if __name__ == "__main__":
    main()
