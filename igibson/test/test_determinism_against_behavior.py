import os
import tempfile

import bddl

import igibson
from igibson.examples.behavior import behavior_demo_replay

bddl.set_backend("iGibson")


def test_determinism_with_existing_vr_demo():
    DEMO_FILE = os.path.join(igibson.ig_dataset_path, "tests", "cleaning_windows_0_Rs_int_2021-05-23_23-11-46.hdf5")

    with tempfile.TemporaryDirectory() as directory:
        replay_file = os.path.join(directory, "replay.hdf5")

        # Replay the canonical demo.
        behavior_demo_replay.safe_replay_demo(DEMO_FILE, out_log_path=replay_file, mode="headless")
