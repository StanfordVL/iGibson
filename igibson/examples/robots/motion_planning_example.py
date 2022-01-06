import argparse
import logging
import os

import numpy as np

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.constants import ViewerMode
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper


def print_mp_info():
    """
    Prints out useful information for user for knowing how to use this MP demo
    """

    def print_command(char, info):
        char += " " * (10 - len(char))
        print("{}\t{}".format(char, info))

    print()
    print("*" * 30)
    print("MOTION PLANNING DEMO")
    print("*" * 30)
    print()
    print("Use your mouse to generate and execute motion plans:")
    print()
    print_command(
        "Left click",
        "Create (click), visualize (drag) and plan / execute (release) a base motion subgoal "
        "for the robot base to reach the physical point that corresponds to the clicked pixel",
    )
    print_command(
        "Middle click",
        "Create, and plan / execute an arm motion subgoal for the robot end-effector "
        "to reach the physical point that corresponds to the clicked pixel",
    )
    print()
    print("Press 'm' to toggle between viewer modes:")
    print()
    print_command("Nav mode", "Enables direct robot navigation and movement of viewer camera")
    print_command("Manip mode", "Enables direct manipulation of objects within the scene")
    print_command("Planning mode", "Enables generation and execution of motion plans (default)")
    print()
    print("Press 'h' for on-screen help.")
    print()
    print("*" * 30)


def run_example(args):
    nav_env = iGibsonEnv(
        config_file=args.config, mode=args.mode, action_timestep=1.0 / 120.0, physics_timestep=1.0 / 120.0
    )

    motion_planner = MotionPlanningWrapper(nav_env)
    state = nav_env.reset()

    # Set viewer camera and directly set mode to planning mode
    nav_env.simulator.viewer.initial_pos = [1.5, 0.1, 2.3]
    nav_env.simulator.viewer.initial_view_direction = [-0.7, -0.3, -0.6]
    nav_env.simulator.viewer.reset_viewer()
    nav_env.simulator.viewer.mode = ViewerMode.PLANNING

    # Print out helpful information for this demo
    print_mp_info()

    while True:
        action = np.zeros(nav_env.action_space.shape)
        state, reward, done, _ = nav_env.step(action)


def main():
    """
    Example of usage of the motion planner wrapper
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
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
        choices=["headless", "headless_tensor", "gui_non_interactive", "gui_interactive"],
        default="gui_interactive",
        help="which mode for simulation (default: gui_interactive)",
    )

    args = parser.parse_args()
    run_example(args)


if __name__ == "__main__":
    main()
