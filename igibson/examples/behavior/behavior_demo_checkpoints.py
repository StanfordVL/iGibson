"""Save checkpoints from a BEHAVIOR demo."""
import os

import bddl

import igibson
from igibson.examples.demo.vr_demos.atus.behavior_demo_replay import safe_replay_demo
from igibson.utils.checkpoint_utils import save_checkpoint

bddl.set_backend("iGibson")


def create_checkpoints(demo_file, checkpoint_directory, checkpoint_every_n_steps):
    # Create a step callback function to feed replay steps into checkpoints.
    def step_callback(igbhvr_act_inst):
        if (
            not igbhvr_act_inst.current_success
            and igbhvr_act_inst.simulator.frame_count % checkpoint_every_n_steps == 0
        ):
            save_checkpoint(igbhvr_act_inst.simulator, checkpoint_directory)

    safe_replay_demo(demo_file, mode="headless", step_callback=step_callback)


def main():
    demo_file = os.path.join(igibson.ig_dataset_path, "tests", "storing_food_0_Rs_int_2021-05-31_11-49-30.hdf5")
    checkpoint_directory = "checkpoints"
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    create_checkpoints(demo_file, checkpoint_directory, 50)


if __name__ == "__main__":
    main()
