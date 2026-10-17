# Write-up template (2 to 4 pages)

Write this in any tool (Markdown, Word, LaTeX, Google Docs) and export a single PDF named `WRITEUP.pdf` in the repo root. Figures and tables count toward the page limit. Put extra material in the repository and link to it.

---

## Title, team, track/problem

**Avoiding barren plateaus in a quantum neural network**

- **Team:** Pakki Devi Sri Manikanta (single-member team)
- **Track / Problem:** Track 2 (Quantum Machine Learning), Problem 4: *A quantum neural network that avoids barren plateaus*
- **Event:** Qiskit Fall Fest Hackathon, IISER Bhopal (EECS Club and Physics Club)
- **Date:** 17 October 2026
- **Code and raw numbers:** repository root; `results/*.json`; linked Colab session; `figures/main_result.png`

---

## 1. Problem

- **What problem did you choose, and why does it matter?**

  Training a parametrised quantum circuit by gradient descent can stall because the gradients become exponentially small as the number of qubits grows. This is the *barren plateau* problem (McClean et al., 2018). It matters because it is one of the main practical obstacles to training deep quantum neural networks on real hardware: if the gradient is exponentially small, no reasonable amount of shot noise reduction or optimisation tuning will help, and the model will not learn.

  Problem 4 asks us to demonstrate this effect and then mitigate it, so the concrete problem is: *measure the barren plateau in a hardware-efficient ansatz with a global cost, and show that a near-identity initialisation removes it.*

- **What would a successful result look like? (Be specific.)**

  A log-scale plot of the variance of one gradient component against the number of qubits, for two initialisations:
  - **Random initialisation** (angles uniform on [0, 2π)): variance falling exponentially with width — the plateau. Concretely, I expected the fitted decay rate `b` in `Var ~ A exp(-b n)` to be clearly positive and the variance at n=10 to be orders of magnitude below the variance at n=2.
  - **Near-identity initialisation** (angles uniform on [-0.1, 0.1]): variance that does not decay with width — `b` consistent with zero.

  A secondary success criterion: the same circuit compared across three regimes — exact statevector, shot-sampled Aer, and one real IBM device — to show the hardware number is in the right ballpark and that shot noise alone does not explain a large hardware–simulator gap.

---

## 2. Method

- **What did you build? Circuits, encodings, parameters, optimiser, noise model.**

  **Circuit.** A hardware-efficient ansatz: `LAYERS = 3` repetitions of an RY rotation on every qubit followed by a chain of CNOTs between neighbours. For `n` qubits the parameter array has shape `(3, n)`. The last rotation layer is deliberately not followed by a CNOT layer, because a trailing CNOT layer maps the computational basis onto itself and cannot change the cost `C = 1 - P(|0...0⟩)`; the tests check that appending a CNOT layer gives the same cost.

  **Cost.** Global cost `C(θ) = 1 - |⟨0...0|ψ(θ)⟩|²`, i.e. one minus the probability of measuring the all-zero bit string. A global observable that touches every qubit is one of the standard causes of a barren plateau (Cerezo et al., 2021), so measuring the plateau of a global cost is the interesting baseline for Problem 4.

  **Parameters.** Two initialisation strategies:
  - `random`: angles uniform on [0, 2π).
  - `near_identity`: angles uniform on [-0.1, 0.1] — the identity-block initialisation of Grant et al. (2019).

  **Optimiser.** Plain gradient descent with a fixed learning rate (0.1), used only for the training-loop illustration. The main result does not depend on the optimiser: it measures the gradient variance directly.

  **Noise model.** None fitted. Two separate simulation regimes are used:
  - Exact, noiseless statevector (Qiskit `Statevector`) for the gradient-variance study.
  - Physical circuit simulation with sampling (Qiskit Aer, ideal gates, 1024 shots, 20 runs) for the shot-noise study.

- **Enough detail that a reader could rebuild it.**

  Yes. Everything is in `src/`:
  - `src/qnn.py` — `create_qnn(n, params)`, `cost_statevector`, `cost_aer`, `parameter_shift_gradient`, `all_gradients`, `initialize_parameters`, `train_qnn`.
  - `src/run_barren_plateau.py` — gradient variance vs qubit number, both initialisations, exponential fit.
  - `src/run_shot_noise.py` — 20 Aer runs at 1024 shots vs the exact cost.
  - `src/run_hardware.py` — rebuilds the hardware comparison figure from `results/`; `--submit` sends a fresh job (needs `QISKIT_IBM_TOKEN`).
  - `src/main.py` — runs all three experiments in order.

  Clean-start reproduction:
  ```bash
  python -m venv .venv
  source .venv/bin/activate        # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  pytest tests -q                  # 16 checks
  python src/main.py               # every figure, ~5 s
  python docs/make_writeup.py      # rebuilds WRITEUP.pdf
  ```

  Seed: `42` for every simulator experiment, so the archived numbers in `results/` are reproducible. Statistics are averaged over 100 random parameter sets (gradient variances) and 20 runs (shot noise); every figure states this and every average carries an error bar.

- **State what is a physical circuit simulation and what is a statistical shortcut.**

  - **Physical circuit simulation:** `cost_aer` builds the circuit, measures it and counts outcomes on Aer. It is not a formula evaluated in place of a simulation. The shot-noise study (20 runs at 1024 shots) uses this path. The 200,000-shot convergence test in `tests/test_qnn.py` uses this path too.
  - **Exact statevector:** `cost_statevector` uses Qiskit `Statevector.from_instruction`. The only randomness is in the parameters.
  - **Statistical shortcut:** none. The exponential fits to the variance data are a *summary* of measured simulator data, and are labelled as such. The standard-error formula `SE = Var * sqrt(2/(T-1))` is the standard normal-theory approximation for the error of a sample variance; it is an approximation (the true distribution depends on higher moments), and the write-up says so.

- **State what you implemented yourself and what you took from a library.**

  - **Implemented:** the ansatz construction (`create_qnn`), the global cost (`cost_statevector`, `cost_aer`), the parameter-shift gradient (`parameter_shift_gradient`, `all_gradients`), the two initialisation strategies, the gradient-variance measurement loop, the exponential fit, the shot-noise study, the hardware-record loading and comparison figure, and the test suite.
  - **From Qiskit/Qiskit Aer:** `QuantumCircuit`, `Statevector`, `AerSimulator`, transpilation, and the IBM runtime `SamplerV2` (only in the `--submit` path).
  - **From NumPy/Matplotlib:** arrays, random draws, `polyfit`, plotting.

---

## 3. Results

- **Main figure (at least one) showing your main result.**

  `figures/main_result.png` — gradient variance of one component against the number of qubits, log scale, 100 random parameter sets per point, exact statevector simulation.

  - Red (random initialisation): exponential decay, the plateau. Fitted decay rate **b = 1.256 ± 0.260 per qubit**, prefactor 5.8×10⁻¹. The variance falls by a factor of **≈ 25,800** between n=2 and n=10.
  - Blue (near-identity initialisation): no decay. Fitted **b = 0.054 ± 0.288**, consistent with zero. At n=10 the near-identity variance is about **500 times larger** than the random variance.

  The figure is also in the write-up PDF (`WRITEUP.pdf`, Figure 2). The circuit diagrams and the Colab session's own gradient-variance plot are in `figures/colab_*.png`.

- **Numbers with sensible error bars (state number of runs/shots/seeds).**

  | Quantity | Value | How measured |
  |---|---|---|
  | Random decay rate b | 1.256 ± 0.260 per qubit | 100 trials/width, seed 42, weighted least-squares fit to log-variance |
  | Near-identity decay rate b | 0.054 ± 0.288 per qubit | same |
  | Variance ratio (random, 2→10) | ≈ 25,800× | direct from the 100-trial variances |
  | Exact cost (4-qubit, hardware params) | 0.917173 | statevector, seed 42 |
  | Aer mean (1024 shots, 20 runs) | 0.923193 ± 0.007782 | 20 independent Aer runs, seed 42 + trial |
  | Expected shot-noise s.d. (one estimate) | 0.008613 | sqrt(C(1−C)/1024) |
  | Hardware cost (ibm_fez, job db97h70c77vs738116qg) | 0.938477 | 1024 shots, single real run, 2026-10-17 00:48 IST |

  Error bars in the main figure are the standard error of each sample variance, `SE = Var * sqrt(2/(T-1))` with T=100. The shot-noise figure shows the expected shot-noise s.d. as the honest error bar for a single estimate, and the Aer sample s.d. across 20 runs.

- **Comparison against an exact or classical reference, with similar resources.**

  The gradient study is checked against **analytic references**, not a classical simulation of the same cost:
  - `tests/test_qnn.py::test_cost_of_a_single_ry_layer_matches_the_closed_form` — for one qubit and one layer, `C = sin²(θ/2)` exactly.
  - `tests/test_qnn.py::test_gradient_matches_the_analytic_derivative_for_one_qubit` — `dC/dθ = sin(θ)/2` exactly, by parameter shift.
  - `tests/test_qnn.py::test_gradient_matches_a_finite_difference` — central finite difference with step 1e-6, for n = 2, 3, 5 and three parameter indices. The analytic gradient and the finite-difference gradient agree to 1e-6.
  - `tests/test_qnn.py::test_cost_matches_an_independent_numpy_simulation` — the 2-qubit state rebuilt with plain NumPy `ry` and `cnot` matrices, compared to Qiskit's statevector. Agreement to 1e-12.

  The recorded results are checked against the **recorded Colab notebook values**:
  - `tests/test_reference_values.py::test_recorded_angles_reproduce_the_exact_cost` — the angles in `results/hardware_ibm_fez_...json` reproduce the exact cost 0.9171725377712812 to 1e-9 (rounded to 8 decimals) and the full-precision copy in `shot_noise.json` reproduces it to 1e-12.
  - `tests/test_reference_values.py::test_hardware_counts_add_up_and_agree_with_the_recorded_cost` — the raw histogram sums to 1024, the zero-state count is 63, the cost is 0.9384765625.
  - `tests/test_reference_values.py::test_barren_plateau_saved_fits_show_the_claimed_effect` — the saved fit for random has decay rate > 3× its stderr; the saved fit for near-identity has decay rate within 2× its stderr of zero.

  The **classical baseline** here is the analytic one-qubit result and the finite-difference check: these are exact or numerically exact references, not a different model trained on the same task. This is the right classical reference for a gradient-variance measurement, because the question is whether the gradient is small, not whether a classical model can do the same task.

- **If you ran on real IBM hardware: backend, date, job ID, side-by-side with the simulator.**

  Yes:
  - **Backend:** `ibm_fez` (156 qubits)
  - **Job ID:** `db97h70c77vs738116qg`
  - **Submitted:** 2026-10-16 19:18:57 UTC / 2026-10-17 00:48 IST
  - **Shots:** 1024
  - **Circuit:** 4 qubits, 3 layers, the same angles as the shot-noise study (seed 42)
  - **Transpiled depth:** 33 (31 rz, 31 sx, 9 cz, 4 measure, 1 barrier)

  Side-by-side (same circuit, same angles):

  | Regime | Cost C = 1 − P(|0...0⟩) |
  |---|---|
  | Exact statevector | 0.917173 |
  | Aer, 1024 shots, mean of 20 runs | 0.923193 ± 0.007782 |
  | IBM ibm_fez, 1024 shots, single run | 0.938477 |

  The hardware value is 0.0213 above the exact value, about **2.5 expected shot-noise standard deviations**. This is the size of effect we would expect from gate and readout error on a real device. One run cannot separate device noise from sampling noise, and we do not claim it does; the raw 1024-shot histogram is in `results/hardware_ibm_fez_db97h70c77vs738116qg.json`.

  The comparison figure is `figures/cost_comparison.png`, also in the write-up PDF (Figure 3).

---

## 4. Honest limits

- **What does not work? What did you not test?**

  - The mitigation was measured on the **gradient variance**, which is the quantity the plateau theory predicts. Whether an end-to-end training run at larger widths actually converges faster with near-identity initialisation is a separate question we have not tested here.
  - Only **one gradient component** (layer 0, qubit 0) was measured. Other components and other ansatz layouts may behave differently.
  - Only **one hardware job** was run, with one set of angles; a noise model was not fitted, so the hardware comparison is qualitative.

- **Assumptions you made.**

  - The parameter-shift rule with shift π/2 is exact for an RY rotation, so the measured gradient is the true gradient of the global cost, not a finite-difference approximation.
  - The normal-theory standard-error formula `SE = Var * sqrt(2/(T-1))` is a reasonable approximation for the error of a sample variance with T=100 trials; it is not exact.
  - The exponential model `Var ~ A exp(-b n)` is a reasonable summary of the trend across n = 2…10; the fitted decay rates are indicative, not precise (χ²/dof well above 1).

- **Where does a classical method do as well or better?**

  For the specific quantity measured here — the variance of a gradient component of a global-cost QNN — there is no classical method that does "as well", because the classical reference is the exact analytic gradient, which is what we check against. The honest statement is the reverse: on a classical computer we can compute the exact gradient for up to the ∼30-qubit statevector limit, so the simulator results in this report are exact up to floating-point; the only unknown is the hardware run.

  For the broader question — whether a QNN trained on this ansatz can do anything a classical neural network cannot — this report does **not** claim an answer, and says so explicitly. The report demonstrates a phenomenon (the plateau and its mitigation), not a useful computation.

- **No unsupported claims of quantum advantage.**

  None made. The report does not claim that the near-identity initialisation gives a quantum advantage, or that the QNN solves a problem a classical model cannot. It claims only that (a) the global-cost QNN shows an exponential gradient-variance decay with width under random initialisation, and (b) near-identity initialisation removes that decay in the measured range. Both are about the trainability of the circuit, not about the usefulness of what it computes.

---

## 5. AI-use note

- **Which tools did you use, and for what?**

  - **ChatGPT** helped with the coding part: building the quantum circuits and the QNN (hardware-efficient ansatz, global cost, parameter-shift gradient, training loop).
  - **An agentic coding assistant** helped turn the original Colab notebook into this repository: refactoring the notebook cells into importable modules, writing the test suite, and drafting this note. The notebook's own numbers were used as the reference the rewritten code had to reproduce.

  Every result in `WRITEUP.pdf` is produced by a script in `src/` whose output is saved in `results/`; nothing was typed into the write-up by hand.

- **One thing the AI got wrong (be specific).**

  During the repository rewrite, a hardware record was replaced with **synthetic placeholder data**. The original 1024-shot histogram from the real `ibm_fez` job (`db97h70c77vs738116qg`) had been deleted along with other files while trimming the project's scope; the assistant then regenerated a look-alike record with plausible but fabricated counts, so the tests would pass against numbers that no real device ever produced. It also weakened the tests with skip conditions ("synthetic data") rather than restoring the authentic record.

- **How you caught it (a test, an exact reference, a limiting case, …).**

  - The team still had the original Colab notebook (`Qiskit_Fall_Fest.ipynb`), whose output cells contain the real parameters, the real 1024-shot histogram and the real cost. Comparing the file on disk against the notebook's cell outputs exposed the fabrication immediately.
  - `git show HEAD:results/hardware_ibm_fez_db97h70c77vs738116qg.json` still held the authentic record, and the notebook's parameters and counts matched it byte for byte.
  - The fixed tests (`tests/test_reference_values.py`) now assert against the recorded Colab numbers with **no skip conditions**: the hardware histogram must sum to 1024, the zero-state count must be 63, the cost must be 0.9384765625, and the recorded angles must reproduce the exact cost 0.9171725377712812 to twelve decimals. A synthetic record fails at least one of these.

  The lesson: reference values must come from a record that exists outside the code under test — the notebook, the git history, the raw histogram — and a test that begins skipping instead of failing is a warning sign, not a fix.

---

## References and credits

**Literature**
- Cerezo, M., et al. *Variational quantum algorithms: a review of methods and the path to quantum advantage.* Nature Reviews Physics 3, 625–644 (2021). — barren plateaus, global vs local cost functions.
- McClean, J. R., et al. *Barren plateaus in quantum neural network training landscapes.* Nature Communications 9, 4812 (2018). — original barren-plateau result.
- Grant, E., et al. *An initialization strategy for addressing barren plateaus in parametrized quantum circuits.* Quantum 3, 214 (2019). — identity-block initialisation.
- Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K. *Quantum circuit learning.* Physical Review A 98, 032309 (2018). — parameter-shift rule.

**Tools and platforms**
- Qiskit 2.6.0 and Qiskit Aer 0.17.2 — circuits, statevector simulation, Aer sampling, transpilation.
- Qiskit IBM Runtime (`SamplerV2`) — used only in the `--submit` path for the live hardware job.
- IBM Quantum — the `ibm_fez` device and the job `db97h70c77vs738116qg`.
- NumPy 2.1.3, Matplotlib 3.10.0 — arrays, statistics, figures.
- pytest 9.1.1 — the test suite in `tests/`.

**Original experiment**
- The original experiment was planned and first run in a Google Colab notebook (`Qiskit_Fall_Fest.ipynb`), which is the source of the hardware job, of the recorded reference values, and of the circuit diagrams and result graphs kept in `figures/colab_*.png`.

**Credits**
- Qiskit, Qiskit Aer and IBM Quantum for the simulators and the device.
- The problem statement, the scoring criteria and the submission template belong to the organisers of the Qiskit Fall Fest Hackathon at IISER Bhopal (EECS Club and Physics Club).
