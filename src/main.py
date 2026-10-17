"""Run every simulator experiment and write every figure.

    python src/main.py

Three experiments, in order:

1. Barren plateau: gradient variance against the number of qubits, for a random
   initialisation and a near-identity one (``figures/main_result.png``).
2. Shot noise on the Aer simulator against the exact statevector cost
   (``figures/shot_noise.png``).
3. Ideal vs shot noise vs real IBM hardware, rebuilt from the recorded job in
   ``results/`` (``figures/cost_comparison.png``).  This needs no IBM Quantum
   account; submitting a *new* job is ``python src/run_hardware.py --submit``
   and needs ``QISKIT_IBM_TOKEN`` in the environment.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_barren_plateau
import run_shot_noise
import run_hardware


def step(number: int, total: int, title: str) -> float:
    print("=" * 72)
    print(f"{number}/{total}  {title}")
    print("=" * 72)
    return time.time()


def report(start: float) -> None:
    print(f"    done in {time.time() - start:.1f} s")
    print()


def main() -> None:
    overall = time.time()
    total = 3

    start = step(1, total, "Barren plateau: gradient variance vs number of qubits")
    run_barren_plateau.main()
    report(start)

    start = step(2, total, "Shot noise: Aer vs the exact statevector cost")
    run_shot_noise.main()
    report(start)

    start = step(3, total, "Ideal vs shot noise vs real IBM hardware")
    records = run_hardware.load_records()
    if records:
        os.makedirs("figures", exist_ok=True)
        run_hardware.plot_comparison(records, run_hardware.FIGURE)
        print(f"wrote {run_hardware.FIGURE} from {len(records)} recorded hardware job(s)")
    else:
        print("no hardware records found; skipping the comparison figure")
    report(start)

    print(f"all {total} steps finished in {time.time() - overall:.1f} s")
    print("figures:", ", ".join(sorted(os.listdir("figures"))))
    print()


if __name__ == "__main__":
    main()
