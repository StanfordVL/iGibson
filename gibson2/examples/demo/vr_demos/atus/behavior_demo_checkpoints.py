"""Save checkpoints from a BEHAVIOR demo."""
import os

import tasknet

import gibson2
from gibson2.examples.demo.vr_demos.atus.behavior_demo_replay import replay_demo
from gibson2.utils.checkpoint_utils import save_checkpoint

tasknet.set_backend("iGibson")


def create_checkpoints(demo_file, checkpoint_directory, checkpoint_every_n_steps):
    # Create a step callback function to feed replay steps into checkpoints.
    def step_callback(igtn_task):
        if igtn_task.simulator.frame_count % checkpoint_every_n_steps == 0:
            save_checkpoint(igtn_task.simulator, checkpoint_directory)

    replay_demo(demo_file, no_vr=True, step_callback=step_callback)


def main():
    demo_file = os.path.join(gibson2.ig_dataset_path, 'tests',
                             'storing_food_0_Rs_int_2021-05-31_11-49-30.hdf5')
    checkpoint_directory = "checkpoints"
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    create_checkpoints(demo_file, checkpoint_directory, 50)


if __name__ == "__main__":
    main()
