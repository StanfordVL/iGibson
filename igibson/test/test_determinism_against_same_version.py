import os
import tempfile

import bddl

from igibson.examples.behavior import behavior_demo_collection, behavior_demo_replay

bddl.set_backend("iGibson")


def test_determinism_with_new_demo():
    # First record a random demo.
    with tempfile.TemporaryDirectory() as directory:
        demo_file = os.path.join(directory, "demo.hdf5")
        print("Saving demo.")
        behavior_demo_collection.collect_demo(
            "cleaning_out_drawers", 0, "Benevolence_1_int", vr_log_path=demo_file, no_vr=True, max_steps=400
        )

        # Then replay the random demo.
        print("Replaying demo.")
        replay_file = os.path.join(directory, "replay.hdf5")
        behavior_demo_replay.safe_replay_demo(demo_file, out_log_path=replay_file, mode="headless")
