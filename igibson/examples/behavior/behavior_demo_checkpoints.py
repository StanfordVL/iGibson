"""Save checkpoints from a BEHAVIOR demo."""
import os

import bddl

import igibson
from igibson.examples.behavior.behavior_demo_replay import safe_replay_demo
from igibson.utils.checkpoint_utils import save_checkpoint


def create_checkpoints(demo_file, checkpoint_directory, checkpoint_every_n_steps):
    # Create a step callback function to feed replay steps into checkpoints.
    def step_callback(env, _):
        if not env.task.current_success and env.simulator.frame_count % checkpoint_every_n_steps == 0:
            save_checkpoint(env.simulator, checkpoint_directory)

    safe_replay_demo(demo_file, mode="headless", step_callbacks=[step_callback])


def main():
    demo_file = os.path.join(igibson.ig_dataset_path, "tests", "cleaning_windows_0_Rs_int_2021-05-23_23-11-46.hdf5")
    checkpoint_directory = "checkpoints"
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    create_checkpoints(demo_file, checkpoint_directory, 30)


if __name__ == "__main__":
    main()
