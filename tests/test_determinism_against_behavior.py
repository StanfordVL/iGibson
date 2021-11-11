import os
import tempfile

import bddl
import pytest

import igibson
from igibson.examples.behavior import behavior_demo_replay

bddl.set_backend("iGibson")


@pytest.mark.parametrize(
    "demo_filename",
    [
        "cleaning_windows_0_Rs_int_2021-05-23_23-11-46.hdf5",  # A simpler test.
        "bottling_fruit_0_Wainscott_0_int_0_2021-09-15_18-30-15.hdf5",  # A longer test involving more state changes.
    ],
)
def test_determinism_with_existing_vr_demo(demo_filename):
    DEMO_FILE = os.path.join(igibson.ig_dataset_path, "tests", demo_filename)

    with tempfile.TemporaryDirectory() as directory:
        replay_file = os.path.join(directory, "replay.hdf5")

        # Replay the canonical demo.
        behavior_demo_replay.safe_replay_demo(DEMO_FILE, out_log_path=replay_file, mode="headless")
