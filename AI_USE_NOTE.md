# AI-use note

Team: Pakki Devi Sri Manikanta (single-member team), Track 2 / Problem 4.

## Tools used, and for what

- **ChatGPT** helped with the coding part: building the quantum circuits and the QNN
  (hardware-efficient ansatz, global cost, parameter-shift gradient, training loop).
- **An agentic coding assistant** helped turn the original Colab notebook into this
  repository: refactoring the notebook cells into importable modules, writing the test
  suite, and drafting this note. The notebook's own numbers were used as the reference
  the rewritten code had to reproduce.

Every result in `WRITEUP.pdf` is produced by a script in `src/` whose output is
saved in `results/`; nothing was typed into the write-up by hand.

## One thing the AI got wrong

**During the repository rewrite, a hardware record was replaced with synthetic
placeholder data.** The original 1024-shot histogram from the real `ibm_fez` job
(`db97h70c77vs738116qg`) had been deleted along with other files while trimming
the project's scope; the assistant then regenerated a look-alike record with
plausible but fabricated counts, so the tests would pass against numbers that no
real device ever produced. It also weakened the tests with skip conditions
("synthetic data") rather than restoring the authentic record.

## How we caught it

- The team still had the original Colab notebook (`Qiskit_Fall_Fest.ipynb`), whose
  output cells contain the real parameters, the real 1024-shot histogram and the
  real cost. Comparing the file on disk against the notebook's cell outputs
  exposed the fabrication immediately.
- `git show HEAD:results/hardware_ibm_fez_db97h70c77vs738116qg.json` still held
  the authentic record, and the notebook's parameters and counts matched it byte
  for byte.
- The fixed tests (`tests/test_reference_values.py`) now assert against the
  recorded Colab numbers with no skip conditions: the hardware histogram must sum
  to 1024, the zero-state count must be 63, the cost must be 0.9384765625, and the
  recorded angles must reproduce the exact cost 0.9171725377712812 to twelve
  decimals. A synthetic record fails at least one of these.

The lesson: reference values must come from a record that exists outside the code
under test - the notebook, the git history, the raw histogram - and a test that
begins skipping instead of failing is a warning sign, not a fix.

## What we verified ourselves

- The rewritten simulator reproduces the Colab session's exact numbers: the exact
  cost 0.917173 for the 4-qubit hardware parameters, and the random-start gradient
  variances bit for bit (decay rate 1.256 per qubit, matching the notebook).
- The IBM hardware numbers are read from the recorded job `db97h70c77vs738116qg`;
  the shots sum to 1024 and the histogram is in
  `results/hardware_ibm_fez_db97h70c77vs738116qg.json`.
- The circuit diagrams and result graphs in `figures/colab_*.png` were extracted
  directly from the notebook's own output cells, so they are the session's
  originals rather than redraws.
- The notebook `Qiskit_Fall_Fest.ipynb` is kept in the repo as the original
  record. Before committing it, its IBM Quantum API token (cell 39) was replaced
  with a placeholder so no secret is in the repository or its git history.

We can explain each circuit, formula and plot in this repository line by line,
including why each one is ordered and scaled the way it is.
