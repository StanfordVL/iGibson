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

    motion_planner = MotionPlanningWrapper(environment, amp_based_on_sensing=True)
    state = environment.reset()

    arm_controller_type = motion_planner.robot.arm_controller_type
    print("Arm controller type: ", arm_controller_type)

    while True:
        # for i in range(1000):
        #     action = np.zeros(environment.action_space.shape)
        #     jointss = motion_planner.robot.untucked_default_joints
        #     action[2:12] = jointss[2:12]
        #     print(environment.action_space.shape)
        #     # action = np.random.uniform(-1, 1, environment.action_space.shape)
        #     state, reward, done, _ = environment.step(action)
        # base_path = motion_planner.plan_base_motion([1, 0, 0])
        # print(base_path)
        # motion_planner.dry_run_base_plan(base_path)
        # for i in range(30):
        #     action = np.random.uniform(-1, 1, environment.action_space.shape)
        #     state, reward, done, _ = environment.step(action)
        goal_in_rf = np.array([0.6, 0.1, 0.8])
        joint_states = motion_planner.get_arm_joint_positions(arm_ik_goal_rf=goal_in_rf)
        print("IK solution: ", joint_states)
        if joint_states:
            print("Finding path to the IK solution")
            joint_states_path = motion_planner.plan_arm_motion(joint_states)
            if not joint_states_path:
                print("No collision free path found")
            else:
                print("Collision free path found!!!!!!!!!!!!!!!!!!")

                print("Dry running it")
                motion_planner.dry_run_arm_plan(joint_states_path)

                print("Executing it")

                if arm_controller_type == "cartesian_ik":
                    cartesian_path = motion_planner.joint_to_cartesian_space_rf(joint_states_path)
                    action = np.zeros(environment.action_space.shape)
                    for controller_steps in range(15 * len(cartesian_path)):
                        delta = motion_planner.cartesian_traj_controller_vel(cartesian_path)
                        action[4:7] = delta
                        print("Executing robot action: ", action)
                        state, reward, done, _ = environment.step(action)
                    if motion_planner.has_converged(goal_in_rf):
                        print("Goal achieved. Accuracy: ", motion_planner.convergence_accuracy(goal_in_rf))
                    else:
                        print(
                            "Failed path execution. Error in convergence: ",
                            motion_planner.convergence_accuracy(goal_in_rf),
                        )
                elif arm_controller_type == "joint_space_velocity":
                    action = np.zeros(environment.action_space.shape)
                    for controller_steps in range(10 * len(joint_states_path)):
                        delta_q = motion_planner.joint_traj_controller_vel(joint_states_path)
                        action[2] = delta_q[0]
                        action[5:12] = delta_q[1:8]
                        print("Executing robot action: ")
                        print(action)
                        state, reward, done, _ = environment.step(action)
                    if motion_planner.has_converged(goal_in_rf):
                        print("Goal achieved. Accuracy: ", motion_planner.convergence_accuracy(goal_in_rf))
                    else:
                        print(
                            "Failed path execution. Error in convergence: ",
                            motion_planner.convergence_accuracy(goal_in_rf),
                        )

                elif arm_controller_type == "joint_space_position":
                    action = np.zeros(environment.action_space.shape)
                    for controller_steps in range(3 * len(joint_states_path)):
                        delta_q = motion_planner.joint_traj_controller_pos(joint_states_path)
                        action[2] = delta_q[0]
                        action[5:12] = delta_q[1:8]
                        print("Executing robot action: ", action)
                        state, reward, done, _ = environment.step(action)
                    if motion_planner.has_converged(goal_in_rf):
                        print("Goal achieved. Accuracy: ", motion_planner.convergence_accuracy(goal_in_rf))
                    else:
                        print(
                            "Failed path execution. Error in convergence: ",
                            motion_planner.convergence_accuracy(goal_in_rf),
                        )
                else:
                    print("unexpected arm controller type")
                    exit(-1)

                # for joint_conf in joint_states_path:
                #     print(joint_conf)
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
