import os
import tempfile

import pytest

from igibson.examples.learning.demo_collection_example import collect_demo
from igibson.examples.learning.demo_replaying_example import replay_demo_with_determinism_check


def test_determinism_with_new_demo():
    # First record a random demo.
    with tempfile.TemporaryDirectory() as directory:
        demo_file = os.path.join(directory, "demo.hdf5")
        print("Saving demo.")
        collect_demo("Benevolence_1_int", "cleaning_out_drawers", 0, 0, demo_file=demo_file, short_exec=True)

        # Then replay the random demo.
        print("Replaying demo.")
        replay_file = os.path.join(directory, "replay.hdf5")
        replay_demo_with_determinism_check(demo_file, replay_demo_file=replay_file, mode="headless")
