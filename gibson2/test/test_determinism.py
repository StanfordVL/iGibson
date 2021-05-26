import tempfile
import tasknet
from gibson2.examples.demo.vr_demos.atus import behavior_demo_collection, behavior_demo_replay
import gibson2
import os

tasknet.set_backend("iGibson")


def test_determinism_with_new_demo():
    # First record a random demo.
    with tempfile.NamedTemporaryFile() as demo_file:
        print("Saving demo.")
        behavior_demo_collection.collect_demo(
            "cleaning_out_drawers", 0, "Benevolence_1_int", vr_log_path=demo_file.name, no_vr=True, max_steps=400)

        # Then replay the random demo.
        print("Replaying demo.")
        with tempfile.NamedTemporaryFile() as replay_file:
            replay_determinism = behavior_demo_replay.replay_demo(
                demo_file.name, vr_replay_log_path=replay_file.name, no_vr=True)

    # Assert for the completion state.
    assert replay_determinism, "Replay was not deterministic."

def test_determinism_with_existing_vr_demo():
    DEMO_FILE = os.path.join(gibson2.ig_dataset_path, 'tests',
                             'clearing_the_table_after_dinner_0_Beechwood_0_int_2021-05-19_23-26-26.hdf5')
    # Replay the canonical demo.
    with tempfile.NamedTemporaryFile() as replay_file:
        replay_determinism = behavior_demo_replay.replay_demo(
            DEMO_FILE, vr_replay_log_path=replay_file.name, no_vr=True)

    # Assert for the completion state.
    assert replay_determinism, "Replay was not deterministic."