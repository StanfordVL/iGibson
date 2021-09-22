import argparse
import os

import numpy as np

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper


def run_example(args):
    environment = iGibsonEnv(
        config_file=args.config, mode=args.mode, action_timestep=1.0 / 120.0, physics_timestep=1.0 / 120.0
    )

    motion_planner = MotionPlanningWrapper(environment)
    state = environment.reset()

    while True:
        for i in range(1):
            action = np.zeros(environment.action_space.shape)
            # action = np.random.uniform(-1, 1, environment.action_space.shape)
            state, reward, done, _ = environment.step(action)
        # base_path = motion_planner.plan_base_motion([1, 0, 0])
        # print(base_path)
        # motion_planner.dry_run_base_plan(base_path)
        # for i in range(30):
        #     action = np.random.uniform(-1, 1, environment.action_space.shape)
        #     state, reward, done, _ = environment.step(action)
        joint_states = motion_planner.get_arm_joint_positions([0.6, 0.1, 0.8])
        print("IK solution: ", joint_states)
        if joint_states:
            print("Finding path to the IK solution")
            joint_states_path = motion_planner.plan_arm_motion(joint_states)
            print(joint_states_path)
        # action = np.random.uniform(-1, 1, environment.action_space.shape)
        # state, reward, done, _ = environment.step(action)


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
