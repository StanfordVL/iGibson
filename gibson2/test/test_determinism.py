import tempfile

import tasknet

from gibson2.examples.demo.vr_demos.atus import behavior_demo_collection, behavior_demo_replay

tasknet.set_backend("iGibson")


def test_determinism_with_new_demo():
    # First record a random demo.
    with tempfile.NamedTemporaryFile() as demo_file:
        behavior_demo_collection.collect_demo(
            "cleaning_out_drawers", 0, "Benevolence_1_int", vr_log_path=demo_file.name, no_vr=True)

        # Then replay the random demo.
        with tempfile.NamedTemporaryFile() as replay_file:
            replay_determinism = behavior_demo_replay.replay_demo(
                demo_file.name, vr_replay_log_path=replay_file.name, no_vr=True)

    # Assert for the completion state.
    assert replay_determinism, "Replay was not deterministic."


# To be enabled once we have a canonical test demo.
# def test_determinism_with_existing_demo():
#     DEMO_FILE = ""  # TODO: Put a demo file somewhere, put its path here.
#
#     # Replay the random demo.
#     with tempfile.NamedTemporaryFile() as replay_file:
#         replay_determinism = behavior_demo_replay.replay_demo(
#             DEMO_FILE, vr_replay_log_path=replay_file.name, no_vr=True)
#
#     # Assert for the completion state.
#     assert replay_determinism, "Replay was not deterministic."
