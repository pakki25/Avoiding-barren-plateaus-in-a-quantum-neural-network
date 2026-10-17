"""Real IBM hardware: comparison figure and (optional) fresh submission.

Two modes:

``python src/run_hardware.py``            rebuild ``figures/cost_comparison.png`` from the
                                          hardware records already in ``results/``.  This works
                                          with no account and no token, which is the path the
                                          README tells a reader to use.

``python src/run_hardware.py --submit``   submit the same circuit again to a real backend.  This
                                          needs an IBM Quantum account and the token in the
                                          environment variable ``QISKIT_IBM_TOKEN``; it is the
                                          only step in this repository that needs one.

The token is only ever read from the environment.  It is never written to disk,
never printed and never part of this repository.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from qnn import LAYERS, SHOTS, cost_statevector, create_qnn

RESULTS_GLOB = "results/hardware_*.json"
SHOT_NOISE_RESULTS = "results/shot_noise.json"
FIGURE = "figures/cost_comparison.png"
TOKEN_ENV_VAR = "QISKIT_IBM_TOKEN"


def load_records(pattern: str = RESULTS_GLOB) -> list[dict]:
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        record["path"] = path
        records.append(record)
    return records


def plot_comparison(records: list[dict], path: str = FIGURE) -> str:
    """Bar chart: exact statevector, Aer with 1024 shots, and real hardware."""
    labels = ["Statevector\n(exact)"]
    values = [records[0]["exact_cost"] if "exact_cost" in records[0] else np.nan]
    errors = [0.0]
    colours = ["tab:green"]

    if os.path.exists(SHOT_NOISE_RESULTS):
        with open(SHOT_NOISE_RESULTS, encoding="utf-8") as handle:
            aer = json.load(handle)
        labels.append(f"Aer, {aer['shots']} shots\n(mean of {aer['trials']} runs)")
        values.append(aer["aer_mean"])
        errors.append(aer["aer_std"])
        colours.append("tab:blue")

    for record in records:
        labels.append(f"{record['backend']}\n(job {record['job_id'][:8]}..., {record['shots']} shots)")
        values.append(record["cost"])
        # One run: the shot-noise standard deviation is the honest error bar.
        errors.append(float(np.sqrt(record["cost"] * (1.0 - record["cost"]) / record["shots"])))
        colours.append("tab:red")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    positions = range(len(labels))
    ax.bar(positions, values, yerr=errors, color=colours, capsize=5)
    for position, value in zip(positions, values):
        ax.text(position, value + 0.02, f"{value:.4f}", ha="center", fontsize=10)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("QNN cost  $C = 1 - P(|0...0>)$")
    ax.set_title("Same circuit, same parameters: ideal vs shot noise vs real hardware")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def submit(n: int, shots: int, backend_name: str, token: str, out_dir: str = "results") -> dict:
    """Submit the circuit once to a real backend and record the result.

    The recorded file contains no token and no account information: backend
    name, job id, timestamp, shots, parameters, raw counts and the cost.
    """
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(backend_name)

    rng = np.random
    rng.seed(42)
    params = rng.uniform(0.0, 2.0 * np.pi, size=(LAYERS, n))

    circuit = create_qnn(n, params, LAYERS, measure=True)
    compiled = transpile(circuit, backend=backend, optimization_level=1)

    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)
    job = sampler.run([compiled], shots=shots)
    job_id = job.job_id()
    print(f"submitted job {job_id} to {backend.name}; waiting for the result")

    result = job.result()
    counts = result[0].data.meas.get_counts()
    zero_count = counts.get("0" * n, 0)

    record = {
        "experiment": "QNN cost on real IBM Quantum hardware",
        "backend": backend.name,
        "backend_qubits": backend.num_qubits,
        "job_id": job_id,
        "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shots": shots,
        "qubits": n,
        "layers": LAYERS,
        "parameters": params.tolist(),
        "transpiled_circuit": {
            "optimization_level": 1,
            "depth": compiled.depth(),
            "count_ops": {key: int(value) for key, value in compiled.count_ops().items()},
        },
        "counts": {key: int(value) for key, value in counts.items()},
        "zero_state_count": int(zero_count),
        "zero_state_probability": zero_count / shots,
        "cost": 1.0 - zero_count / shots,
        "exact_cost": cost_statevector(n, params),
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"hardware_{backend.name}_{job_id}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"recorded {path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="IBM hardware comparison for the QNN cost")
    parser.add_argument("--submit", action="store_true", help="submit a new job (needs QISKIT_IBM_TOKEN)")
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--figure", default=FIGURE)
    args = parser.parse_args()

    if args.submit:
        token = os.environ.get(TOKEN_ENV_VAR)
        if not token:
            raise SystemExit(
                f"{TOKEN_ENV_VAR} is not set. Create an IBM Quantum API key and export it as an "
                f"environment variable to submit a new job; the repository works without it."
            )
        record = submit(args.qubits, args.shots, args.backend, token)
        records = [record] + [r for r in load_records() if r["job_id"] != record["job_id"]]
    else:
        records = load_records()

    if not records:
        raise SystemExit("no hardware records found in results/; nothing to plot")

    os.makedirs(os.path.dirname(args.figure), exist_ok=True)
    plot_comparison(records, args.figure)
    print(f"wrote {args.figure} from {len(records)} hardware record(s)")


if __name__ == "__main__":
    main()
