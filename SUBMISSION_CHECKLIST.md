# Submission checklist (filled in)

Team: Pakki Devi Sri Manikanta  |  Track 2, Problem 4: *A quantum neural network that avoids barren plateaus*

## Required parts

- [x] Repository runs from a clean start: fresh virtual environment, `pip install -r requirements.txt`,
      then the commands in `README.md` (re-run from scratch before submitting, with the noted runtimes)
- [x] `requirements.txt` pins every direct package with `==`; the Python version is in `README.md` (3.12.10)
- [x] `README.md` has the team, the problem, a summary, the exact commands, the expected runtime and the seed
- [x] `WRITEUP.pdf` at the repository root: Problem, Method, Results, Honest limits
- [x] At least one main-result figure, in the write-up and in `figures/main_result.png`
- [x] Results checked against exact references: closed-form amplitudes, central finite differences for every
      gradient, the recorded Colab values and the raw hardware histogram (`tests/`, 16 checks)
- [x] The comparison covers simulator vs simulator vs hardware: exact statevector, Aer at 1024 shots
      (20 runs with error bars), and one real `ibm_fez` job at the same shot count
- [x] `AI_USE_NOTE.md` names the tools, one specific error (a hardware record briefly replaced with synthetic
      data) and how it was caught against the Colab notebook and git history
- [x] `README.md` states what is simulated (exact statevector / Aer sampling / one real IBM job) and that
      nothing is a statistical shortcut
- [x] No API token, secret or personal data in the repository. The IBM token used in the original Colab
      session is **revoked** and must never be committed; the hardware path only reads
      `QISKIT_IBM_TOKEN` from the environment
- [x] Bonus: a real IBM hardware run, `ibm_fez`, job `db97h70c77vs738116qg`, 2026-10-17 00:48 IST,
      compared with the simulator and with Aer at the same shot count
- [x] Negative and weak results are reported, not hidden: the near-identity fit is consistent with zero but
      carries a large error bar, and one hardware run cannot separate device noise from sampling noise
- [x] Colab session artefacts kept: the original notebook `Qiskit_Fall_Fest.ipynb` (52 cells, 18 with output) is in the repo root; its IBM Quantum API token has been replaced with a placeholder before committing. The circuit diagrams and result graphs extracted from the notebook's own output cells are in `figures/colab_*.png`

## Before submitting

- [ ] Push to a public GitHub repository (or private with the organisers added as collaborators)
- [ ] Confirm the repository has no `.venv/`, no `results/`-external scratch files and no token in the
      git history (`git log -p | grep -i token` should be empty)
- [ ] Submit the repository link through the organisers' Google Form before Oct 17, 2026, 12:00 noon
- [ ] (Advised) Prepare a short pitch deck for the Oct 18 presentation, in case the team is invited
