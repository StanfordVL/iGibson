from gibson2.envs.locomotor_env import NavigationRandomEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
from gibson2.external.pybullet_tools.utils import set_joint_positions
from gibson2.external.pybullet_tools.utils import joints_from_names
from gibson2.external.pybullet_tools.utils import control_joints, velocity_control_joints, get_joint_positions
from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import ArticulatedObject
import json


import argparse
import numpy as np
from IPython import embed
import pybullet as p


def run_example(args):
    nav_env = NavigationRandomEnv(config_file=args.config,
                                  mode=args.mode,
                                  action_timestep=1.0 / 20.0,
                                  physics_timestep=1.0 / 120.0)

    motion_planner = MotionPlanningWrapper(nav_env)
    state = nav_env.reset()
    env = nav_env
    env.robots[0].set_position_orientation(
        [-0.8, 2.3, 0.03], [0, 0, 0.7071068, 0.7071068])
    # robot_constraint = \
    #     p.createConstraint(0, -1, motion_planner.robot_id,
    #                        -1, p.JOINT_FIXED,
    #                        [0, 0, 1],
    #                        env.robots[0].get_position(),
    #                        [0, 0, 0],
    #                        env.robots[0].get_orientation(),
    #                        [0, 0, 0, 1])
    # env.robots[0].set_position_orientation(
    #     [-0.8, 2.3, 0.0], [0, 0, 0.7071068, 0.7071068])

    # ycb_object = YCBObject('025_mug')
    ycb_object = ArticulatedObject(
        '/cvgl2/u/chengshu/gibsonv2/gibson2/assets/models/mugs/1eaf8db2dd2b710c7d5b1b70ae595e60/1eaf8db2dd2b710c7d5b1b70ae595e60.urdf', scale=0.15)
    nav_env.simulator.import_object(ycb_object)
    # motion_planner.mp_obstacles.append(ycb_object.body_id)
    p.changeDynamics(ycb_object.body_id, -1, mass=0.1, lateralFriction=5.0)
    p.changeDynamics(motion_planner.robot_id, 20, lateralFriction=5.0)
    p.changeDynamics(motion_planner.robot_id, 21, lateralFriction=5.0)

    new_arm_default_joint_positions = (
        0.4, 0, -np.pi / 4, 0, np.pi / 2, 0, np.pi / 4, 0)
    motion_planner.arm_default_joint_positions = new_arm_default_joint_positions

    env.simulator.set_timestep(1 / 480.0, 1 / 20.0)
    motion_planner.set_gripper_max_force(2000.0)
    motion_planner.set_arm_max_force(2000.0)

    embed()
    # ycb_object_pos = np.array([-0.58, 2.9, 0.7])
    ycb_object_pos = np.array([-0.45, 2.95, 0.7])
    ycb_object.set_position_orientation(ycb_object_pos, [0, 0, 0, 1])
    planner_pos = np.array(
        [ycb_object_pos[0] - 0.07, ycb_object_pos[1], ycb_object_pos[2] + 0.02])
    plan = motion_planner.plan_arm_push(planner_pos, [0, 0, -1])
    motion_planner.dry_run_arm_plan(plan)
    motion_planner.marker_direction.set_position([100, 100, 0])
    motion_planner.marker.set_position([100, 100, 0])
    env.robots[0].keep_still()
    motion_planner.close_gripper()

    with open('bc_data/demo_0.json') as f:
        demo = json.load(f)

    # demo trajectory
    obj_traj = np.array([item['obj_pos'] for item in demo])
    goal = obj_traj[-1]

    step_size = 0.025
    while True:
        obj_pos = np.array(ycb_object.get_position())
        if np.linalg.norm(goal - obj_pos) < 0.03:
            motion_planner.open_gripper()
            print('SUCCESS')
            break

        # find the closest point to the current obj position in the demo trajectory
        closest_idx = np.argmin(np.linalg.norm(obj_traj - obj_pos, axis=1))
        dist = np.linalg.norm(obj_traj[closest_idx] - obj_pos)
        next_idx = closest_idx
        # find the next target waypoint in the demo trajectory to move towards
        while dist < step_size and next_idx < (len(obj_traj) - 1):
            next_idx += 1
            dist = np.linalg.norm(obj_traj[next_idx] - obj_pos)
        delta = obj_traj[next_idx] - obj_pos
        motion_planner.move_gripper(delta)
        print('next_idx', next_idx, delta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')

    args = parser.parse_args()
    run_example(args)
