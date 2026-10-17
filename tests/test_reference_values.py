"""Checks that the saved results still agree with the recorded reference values.

The reference points here come from outside the code under test: the numbers the
team recorded in the original Colab session and from the real IBM job
``db97h70c77vs738116qg``.  If a change to the simulator or to the experiments
invalidated them, these tests would fail.
"""

import json
import os

import numpy as np
import pytest

from qnn import LAYERS, cost_statevector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The exact cost of the 4-qubit, 3-layer circuit used for the hardware run.
NOTEBOOK_EXACT_COST = 0.9171725377712812

#: The cost measured on ibm_fez, job db97h70c77vs738116qg, 1024 shots.
NOTEBOOK_HARDWARE_COST = 0.9384765625


def load(name: str) -> dict:
    with open(os.path.join(ROOT, "results", name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_recorded_angles_reproduce_the_exact_cost():
    """The angles sent to hardware give exactly the cost recorded in the notebook.

    This ties the real-device run to this repository's simulator: same circuit,
    same parameters, same exact cost.  The hardware record stores the angles
    rounded to eight decimals, which moves the cost by 1.2e-10, so there are two
    assertions: the hardware record to that tolerance, and the full-precision
    copy of the same angles in ``shot_noise.json`` to twelve decimals.
    """
    hardware = load("hardware_ibm_fez_db97h70c77vs738116qg.json")
    params = np.array(hardware["parameters"])
    assert params.shape == (LAYERS, hardware["qubits"])
    assert hardware["exact_cost"] == pytest.approx(NOTEBOOK_EXACT_COST, abs=1e-12)

    rounded_cost = cost_statevector(hardware["qubits"], params, layers=LAYERS)
    assert rounded_cost == pytest.approx(NOTEBOOK_EXACT_COST, abs=1e-9)

    full_precision = np.array(load("shot_noise.json")["parameters"])
    assert np.abs(full_precision - params).max() < 1e-7  # same angles, not rounded
    assert cost_statevector(hardware["qubits"], full_precision, layers=LAYERS) == pytest.approx(
        NOTEBOOK_EXACT_COST, abs=1e-12
    )


def test_hardware_counts_add_up_and_agree_with_the_recorded_cost():
    """The raw hardware histogram is complete and consistent with the cost."""
    hardware = load("hardware_ibm_fez_db97h70c77vs738116qg.json")
    counts = hardware["counts"]

    assert sum(counts.values()) == hardware["shots"] == 1024
    assert hardware["zero_state_count"] == counts["0" * hardware["qubits"]]
    assert hardware["zero_state_probability"] == pytest.approx(hardware["zero_state_count"] / hardware["shots"])
    assert hardware["cost"] == pytest.approx(1.0 - hardware["zero_state_probability"])
    assert hardware["cost"] == pytest.approx(NOTEBOOK_HARDWARE_COST)


def test_shot_noise_behaves_like_sampling_noise():
    """Twenty Aer runs scatter by about sqrt(C(1-C)/shots), no more, no less."""
    shot = load("shot_noise.json")
    expected = shot["shot_noise_standard_deviation_expected"]
    assert expected == pytest.approx(
        np.sqrt(shot["exact_cost"] * (1.0 - shot["exact_cost"]) / shot["shots"]), rel=1e-9
    )

    # The mean of 20 runs should sit within a few standard errors of the exact
    # value, and the observed scatter should be close to the binomial prediction.
    standard_error = expected / np.sqrt(shot["trials"])
    assert abs(shot["aer_mean"] - shot["exact_cost"]) < 4.0 * standard_error
    assert shot["aer_std"] == pytest.approx(expected, rel=0.4)
    assert len(shot["aer_costs"]) == shot["trials"]
    assert shot["mean_absolute_error_vs_exact"] == pytest.approx(
        float(np.mean(np.abs(np.array(shot["aer_costs"]) - shot["exact_cost"]))), abs=1e-12
    )


def test_the_hardware_run_sits_above_the_simulated_spread():
    """The recorded hardware cost is above the ideal value, as gate error suggests.

    This is a sanity check on the claim in the write-up that the hardware excess is
    a little larger than shot noise alone, not a claim that it must be.
    """
    shot = load("shot_noise.json")
    excess = NOTEBOOK_HARDWARE_COST - shot["exact_cost"]
    assert 0.0 < excess
    assert excess < 6.0 * shot["shot_noise_standard_deviation_expected"]


@pytest.mark.parametrize(
    "results_file,expected",
    [
        ("barren_plateau.json", {"trials": 100, "seed": 42, "layers": 3}),
        ("shot_noise.json", {"trials": 20, "shots": 1024, "seed": 42}),
    ],
)
def test_saved_results_use_the_documented_settings(results_file, expected):
    """The archived results were produced with the settings the write-up describes."""
    saved = load(results_file)
    for key, value in expected.items():
        assert saved[key] == value, f"{results_file}: {key} is {saved[key]}, expected {value}"


def test_barren_plateau_saved_fits_show_the_claimed_effect():
    """The archive supports the two claims made about width: decay, and no decay."""
    barren = load("barren_plateau.json")
    random_fit = barren["random"]["fit"]
    identity_fit = barren["near_identity"]["fit"]

    assert random_fit["decay_rate_per_qubit"] > 3.0 * random_fit["decay_rate_stderr"]
    assert abs(identity_fit["decay_rate_per_qubit"]) < 2.0 * identity_fit["decay_rate_stderr"]

    random_variances = {r["qubits"]: r["variance"] for r in barren["random"]["records"]}
    assert random_variances[10] < random_variances[2] / 1000.0
