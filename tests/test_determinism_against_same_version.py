import os
import tempfile

from igibson.examples.learning.demo_collection_example import collect_demo
from igibson.examples.learning.demo_replaying_example import safe_replay_demo


def test_determinism_with_new_demo():
    # First record a random demo.
    with tempfile.TemporaryDirectory() as directory:
        demo_file = os.path.join(directory, "demo.hdf5")
        print("Saving demo.")
        collect_demo("Benevolence_1_int", "cleaning_out_drawers", 0, 0, log_path=demo_file)

        # Then replay the random demo.
        print("Replaying demo.")
        replay_file = os.path.join(directory, "replay.hdf5")
        safe_replay_demo(demo_file, out_log_path=replay_file, mode="headless")
