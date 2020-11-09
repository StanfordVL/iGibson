from gibson2.envs.locomotor_env import NavigationRandomEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
from gibson2.external.pybullet_tools.utils import set_joint_positions
from gibson2.external.pybullet_tools.utils import joints_from_names
from gibson2.external.pybullet_tools.utils import control_joints, velocity_control_joints, get_joint_positions
from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import ArticulatedObject
import json
import torch

import argparse
import numpy as np
from IPython import embed
import pybullet as p
from behavior_cloning_training import Model
import os


def run_example(args):
    # Load BC model
    model = Model(input_size=3,
                  hidden_size=64,
                  num_layers=2,
                  arm_action_size=3,
                  gripper_action_size=2)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load environment
    nav_env = NavigationRandomEnv(config_file=args.config,
                                  mode=args.mode,
                                  action_timestep=1.0 / 20.0,
                                  physics_timestep=1.0 / 120.0)
    motion_planner = MotionPlanningWrapper(nav_env)
    nav_env.reset()

    # Place robot and change initial arm configuration
    env = nav_env
    env.robots[0].set_position_orientation(
        [-0.8, 2.3, 0.03], [0, 0, 0.7071068, 0.7071068])
    new_arm_default_joint_positions = (
        0.4, 0, -np.pi / 4, 0, np.pi / 2, 0, np.pi / 4, 0)
    motion_planner.arm_default_joint_positions = new_arm_default_joint_positions

    # Place third-person camera for video recording
    nav_env.simulator.viewer.px = -1.3
    nav_env.simulator.viewer.py = 3.3
    nav_env.simulator.viewer.pz = 1.2
    nav_env.simulator.viewer.view_direction = np.array(
        [np.sqrt(0.48), -0.6, -0.4])

    # Load mug
    mugs = []
    root_dir = '/cvgl2/u/chengshu/gibsonv2/gibson2/assets/models/mugs'
    for obj_dir in os.listdir(root_dir):
        if obj_dir != '1eaf8db2dd2b710c7d5b1b70ae595e60':
            continue
        urdf = os.path.join(root_dir, obj_dir, obj_dir + '.urdf')
        random_scale = np.random.uniform(0.123, 0.205)
        mug = ArticulatedObject(urdf, scale=random_scale)
        nav_env.simulator.import_object(
            mug, use_pbr=False, use_pbr_mapping=False)
        p.changeDynamics(mug.body_id, -1, mass=0.1, lateralFriction=2.0)
        mugs.append(mug)

    # Add mug into mp_obstacles
    # motion_planner.mp_obstacles.append(ycb_object.body_id)

    # Increase friction
    p.changeDynamics(motion_planner.robot_id, 20, lateralFriction=5.0)
    p.changeDynamics(motion_planner.robot_id, 21, lateralFriction=5.0)

    # Set physics timestep and gripper/arm position control max force
    env.simulator.set_timestep(1 / 480.0, 1 / 20.0)
    motion_planner.set_gripper_max_force(20000.0)
    motion_planner.set_arm_max_force(20000.0)

    # Retrieve mean and stddev of input from training
    with open('bc_data/train.json') as f:
        train_data = json.load(f)
    input_mean = np.mean(train_data['obj_pos'], axis=0)
    input_std = np.std(train_data['obj_pos'], axis=0)

    episode_i = 0
    num_episodes = 100
    successes = 0.0
    episode_length = 50
    while True:
        if episode_i > 0:
            print('success rate: {} [{}/{}]'.format(
                successes / episode_i, successes, episode_i))
        if episode_i == num_episodes:
            break

        # Reset mug
        ycb_object_pos = np.random.uniform(
            [-0.475, 2.9, 0.7], [-0.425, 3.0, 0.7])
        # ycb_object_pos = np.array([-0.45, 2.95, 0.7])

        for i, mug in enumerate(mugs):
            mug.set_position_orientation([100 + i, 100, 0], [0, 0, 0, 1])
        ycb_object = np.random.choice(mugs)
        ycb_object.set_position_orientation(ycb_object_pos, [0, 0, 0, 1])

        # Reset initial grasping position and plan
        planner_pos = np.array(
            [ycb_object_pos[0] - 0.07, ycb_object_pos[1], ycb_object_pos[2] + 0.02])
        plan = None
        # Loop until a plan is found
        while plan is None:
            plan = motion_planner.plan_arm_push(planner_pos, [0, 0, -1])
        motion_planner.dry_run_arm_plan(plan)

        # Remove marker for better visualization
        # motion_planner.marker_direction.set_position([100, 100, 0])
        # motion_planner.marker.set_position([100, 100, 0])

        # Clear out robot internal velocities
        env.robots[0].keep_still()

        # Close gripper
        motion_planner.close_gripper()
        loose_gripper = False
        success = False
        for step in range(episode_length):
            # During the episode, the gripper should maintain contact with the object
            has_contact = len(p.getContactPoints(
                motion_planner.robot_id, ycb_object.body_id)) > 0
            if not has_contact:
                loose_gripper = True
                break

            obj_pos = np.array(ycb_object.get_position())
            # normalize obj_pos for network input
            obj_pos = (obj_pos - input_mean) / input_std
            obj_pos_torch = torch.from_numpy(
                obj_pos.astype(np.float32)).unsqueeze(0).cuda()
            with torch.no_grad():
                arm_action, gripper_action = model(obj_pos_torch)
                arm_action = arm_action[0]
                gripper_action = gripper_action[0]
                gripper_action_prob = torch.nn.functional.softmax(
                    gripper_action, dim=0)
                arm_action_np = arm_action.detach().cpu().numpy()
                gripper_action_prob_np = gripper_action_prob.detach().cpu().numpy()

            open_prob = gripper_action_prob_np[1]
            if open_prob > 0.5:
                success = True
                motion_planner.open_gripper()
                # Let mug fall down for 0.5 seconds
                for _ in range(int(0.5 / nav_env.simulator.render_timestep)):
                    nav_env.simulator_step()
                break
            else:
                # During training, the target delta is a bit too small for the
                # control frequency 20hz. Scale by 2 works well.
                motion_planner.move_gripper(arm_action_np * 2.0)

        if not loose_gripper:
            successes += float(success)
            episode_i += 1


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
    parser.add_argument('--ckpt',
                        help='policy checkpoint path',
                        required=True)
    args = parser.parse_args()
    run_example(args)
