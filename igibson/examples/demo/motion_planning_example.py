import argparse
import os

import numpy as np

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper


def run_example(args):
    nav_env = iGibsonEnv(
        config_file=args.config, mode=args.mode, action_timestep=1.0 / 120.0, physics_timestep=1.0 / 120.0
    )

    motion_planner = MotionPlanningWrapper(nav_env)
    state = nav_env.reset()

    while True:
        for i in range(90):
            action = np.random.uniform(-1, 1, nav_env.action_space.shape)
            state, reward, done, _ = nav_env.step(action)
        base_path = motion_planner.plan_base_motion([1, 0, 0])
        print(base_path)
        motion_planner.dry_run_base_plan(base_path)
        for i in range(30):
            action = np.random.uniform(-1, 1, nav_env.action_space.shape)
            state, reward, done, _ = nav_env.step(action)
        joint_states = motion_planner.get_arm_joint_positions([0.3, 0.3, 0.8])
        print(joint_states)
        # action = np.random.uniform(-1, 1, nav_env.action_space.shape)
        # state, reward, done, _ = nav_env.step(action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default=os.path.join(igibson.example_config_path, "fetch_motion_planning.yaml"),
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "gui", "iggui"],
        default="iggui",
        help="which mode for simulation (default: iggui)",
    )

    args = parser.parse_args()
    run_example(args)
