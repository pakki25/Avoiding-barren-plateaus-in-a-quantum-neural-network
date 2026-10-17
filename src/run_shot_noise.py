"""Finite-shot noise on the Aer simulator, compared with the exact cost.

The same four-qubit circuit (parameters drawn with seed 42, the parameters that
were also sent to the IBM hardware) is evaluated

* exactly, on the statevector (infinite shots), and
* 20 times on Aer with 1024 shots each, which is a physical circuit simulation
  with sampling noise.

The spread of the 20 estimates shows how much of the difference between a
simulator and real hardware could be explained by shot noise alone.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from qnn import LAYERS, SHOTS, cost_aer, cost_statevector

SEED = 42
QUBITS = 4
TRIALS = 20
RESULTS = "results/shot_noise.json"
FIGURE = "figures/shot_noise.png"
HARDWARE_RECORD = "results/hardware_ibm_fez_db97h70c77vs738116qg.json"


def hardware_parameters(n: int = QUBITS, seed: int = SEED) -> np.ndarray:
    """Parameters used for the shot-noise and hardware runs (seed 42)."""
    rng = np.random
    rng.seed(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=(LAYERS, n))


def run(n: int = QUBITS, trials: int = TRIALS, shots: int = SHOTS, seed: int = SEED) -> dict:
    params = hardware_parameters(n, seed)
    exact = cost_statevector(n, params)

    costs = np.array([cost_aer(n, params, shots=shots, seed=seed + trial) for trial in range(trials)])

    data = {
        "experiment": "shot noise of the cost estimate",
        "qubits": n,
        "layers": LAYERS,
        "shots": shots,
        "trials": trials,
        "seed": seed,
        "simulator": "Qiskit Aer, ideal gates with measurement sampling (physical circuit simulation)",
        "parameters": params.tolist(),
        "exact_cost": exact,
        "aer_costs": costs.tolist(),
        "aer_mean": float(np.mean(costs)),
        "aer_std": float(np.std(costs, ddof=1)),
        "mean_absolute_error_vs_exact": float(np.mean(np.abs(costs - exact))),
        "shot_noise_standard_deviation_expected": float(np.sqrt(exact * (1.0 - exact) / shots)),
    }

    if os.path.exists(HARDWARE_RECORD):
        with open(HARDWARE_RECORD, encoding="utf-8") as handle:
            hardware = json.load(handle)
        data["hardware_cost"] = hardware["cost"]
        data["hardware_job_id"] = hardware["job_id"]

    return data


def plot(data: dict, path: str = FIGURE) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.hist(data["aer_costs"], bins=10, color="tab:blue", alpha=0.75, label=f"Aer, {data['shots']} shots ({data['trials']} runs)")
    ax.axvline(data["exact_cost"], color="k", linewidth=2, label=f"Exact statevector cost = {data['exact_cost']:.4f}")
    if "hardware_cost" in data:
        ax.axvline(data["hardware_cost"], color="tab:red", linewidth=2, linestyle="--",
                   label=f"IBM ibm_fez hardware = {data['hardware_cost']:.4f}")

    ax.set_xlabel("Estimated cost  $C = 1 - P(|0...0>)$")
    ax.set_ylabel("Number of runs")
    ax.set_title("Shot noise of the cost estimate (same circuit and parameters everywhere)")
    ax.grid(True, alpha=0.3)
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

    print(f"exact cost            {data['exact_cost']:.6f}")
    print(f"Aer mean (1024 shots) {data['aer_mean']:.6f} +/- {data['aer_std']:.6f}")
    print(f"expected shot-noise s.d. {data['shot_noise_standard_deviation_expected']:.6f}")
    if "hardware_cost" in data:
        print(f"recorded hardware cost   {data['hardware_cost']:.6f} (job {data['hardware_job_id']})")
    print(f"wrote {results_path} and {figure_path}")
    return data


if __name__ == "__main__":
    main()
