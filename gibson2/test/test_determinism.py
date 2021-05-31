import tempfile
import tasknet
from gibson2.examples.demo.vr_demos.atus import behavior_demo_collection, behavior_demo_replay
import gibson2
import os

tasknet.set_backend("iGibson")


def test_determinism_with_new_demo():
    # First record a random demo.
    with tempfile.TemporaryDirectory() as directory:
        demo_file = os.path.join(directory, "demo.hdf5")
        print("Saving demo.")
        behavior_demo_collection.collect_demo(
            "cleaning_out_drawers", 0, "Benevolence_1_int", vr_log_path=demo_file, no_vr=True, max_steps=400)

        # Then replay the random demo.
        print("Replaying demo.")
        replay_file = os.path.join(directory, "replay.hdf5")
        behavior_demo_replay.replay_demo(demo_file, out_log_path=replay_file, no_vr=True)


def test_determinism_with_existing_vr_demo():
    DEMO_FILE = os.path.join(gibson2.ig_dataset_path, 'tests',
                             'cleaning_windows_0_Rs_int_2021-05-23_23-11-46.hdf5')

    with tempfile.TemporaryDirectory() as directory:
        replay_file = os.path.join(directory, "replay.hdf5")

        # Replay the canonical demo.
        behavior_demo_replay.replay_demo(DEMO_FILE, out_log_path=replay_file, no_vr=True)
