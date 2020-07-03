import os
import pybullet as p
import time
import numpy as np
import gibson2.external.pybullet_tools.transformations as T
import gibson2.external.pybullet_tools.utils as PBU

import gibson2.envs.kitchen.plan_utils as PU
import gibson2.envs.kitchen.skills as skills
from gibson2.envs.kitchen.envs import BaseKitchenEnv
from gibson2.envs.kitchen.env_utils import pose_to_array, pose_to_action


"""
Task plans -> skill parameters
Parameterized skill library
Skills + parameters -> joint-space motion plan
Motion plan -> task-space path
task-space path -> gripper actuation
"""


def execute_planned_path(env, path):
    """Execute a planned path an relabel actions."""

    all_obs = []
    actions = []
    rewards = []
    states = []

    for i in range(len(path)):
        tpose = path.arm_path[i]
        grip = path.gripper_path[i]

        cpose = pose_to_array(env.robot.get_eef_position_orientation())
        tpose = pose_to_array(tpose)

        action = np.zeros(env.action_dimension)
        action[-1] = grip
        action[:-1] = pose_to_action(cpose, tpose, max_dpos=env.MAX_DPOS, max_drot=env.MAX_DROT)
        actions.append(action)

        rewards.append(float(env.is_success()))
        states.append(env.sim_state)
        all_obs.append(env.get_observation())

        env.step(action)

    all_obs.append(env.get_observation())
    actions.append(np.zeros(env.action_dimension))
    rewards.append(float(env.is_success()))
    states.append(env.sim_state)

    all_obs = dict((k, np.array([all_obs[i][k] for i in range(len(all_obs))])) for k in all_obs[0])
    return states, actions, rewards, all_obs


def get_demo(env):
    env.reset()
    all_states = []
    all_actions = []
    all_rewards = []
    all_obs = []

    drawer_grasp_pose = (
        [0.3879213,  0.0072391,  0.71218301],
        T.quaternion_multiply(T.quaternion_about_axis(np.pi, axis=[0, 0, 1]), T.quaternion_about_axis(np.pi / 2, axis=[1, 0, 0]))
    )
    path = skills.plan_skill_open_prismatic(
        env.planner,
        obstacles=env.objects.body_ids,
        grasp_pose=drawer_grasp_pose,
        reach_distance=0.05,
        retract_distance=0.25,
        joint_resolutions=(0.25, 0.25, 0.25, 0.2, 0.2, 0.2)
    )
    states, actions, rewards, obs = execute_planned_path(env, path)
    all_states.append(states)
    all_actions.append(actions)
    all_rewards.append(rewards)
    all_obs.append(obs)

    can_grasp_pose = ((0.03, -0.005, 1.06), (0, 0, 1, 0))
    path = skills.plan_skill_grasp(
        env.planner,
        obstacles=env.objects.body_ids,
        grasp_pose=can_grasp_pose,
        reach_distance=0.05,
        lift_height=0.1,
        joint_resolutions=(0.1, 0.1, 0.1, 0.2, 0.2, 0.2)
    )
    states, actions, rewards, obs = execute_planned_path(env, path)
    all_states.append(states)
    all_actions.append(actions)
    all_rewards.append(rewards)
    all_obs.append(obs)

    can_drop_pose = ((0.469, 0, 0.952), (0, 0, 1, 0))
    path = skills.plan_skill_place(
        env.planner,
        obstacles=env.objects.body_ids,
        holding=env.objects["can"].body_id,
        place_pose=can_drop_pose,
        joint_resolutions=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    )

    path.append_pause(30)
    states, actions, rewards, obs = execute_planned_path(env, path)
    all_states.append(states)
    all_actions.append(actions)
    all_rewards.append(rewards)
    all_obs.append(obs)

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_obs = dict((k, np.concatenate([all_obs[i][k] for i in range(len(all_obs))], axis=0)) for k in all_obs[0])
    return all_states, all_actions, all_rewards, all_obs


def create_dataset(args):
    import h5py
    import json

    env_kwargs = dict(
        robot_base_pose=([0, 0.3, 1.2], [0, 0, 1, 0]),
        num_sim_per_step=5,
        sim_time_step=1./240.
    )
    env = BaseKitchenEnv(**env_kwargs, use_planner=True, hide_planner=False, use_gui=args.gui)

    if os.path.exists(args.file):
        os.remove(args.file)
    f = h5py.File(args.file)
    f_sars_grp = f.create_group("data")

    env_args = dict(
        type=4,
        env_name=env.name,
        env_kwargs=env_kwargs,
    )
    f_sars_grp.attrs["env_args"] = json.dumps(env_args)

    success_i = 0
    total_i = 0
    while success_i < args.n:
        try:
            states, actions, rewards, all_obs = get_demo(env)
        except PU.NoPlanException as e:
            print(e)
            continue
        total_i += 1
        if not env.is_success():
            continue

        f_demo_grp = f_sars_grp.create_group("demo_{}".format(success_i))
        f_demo_grp.attrs["num_samples"] = (states.shape[0] - 1)
        f_demo_grp.create_dataset("states", data=states[:-1])
        f_demo_grp.create_dataset("actions", data=actions[:-1])
        for k in all_obs:
            f_demo_grp.create_dataset("obs/{}".format(k), data=all_obs[k][:-1])
            f_demo_grp.create_dataset("next_obs/{}".format(k), data=all_obs[k][1:])
        success_i += 1
        print("{}/{}".format(success_i, total_i))
    f.close()


def playback(args):
    import h5py
    import json

    f = h5py.File(args.file, 'r')
    env_args = json.loads(f["data"].attrs["env_args"])

    env = BaseKitchenEnv(**env_args["env_kwargs"], use_gui=True)
    demos = list(f["data"].keys())
    for demo_id in demos:
        env.reset()
        actions = f["data/{}/actions".format(demo_id)][:]

        for i in range(400):
            if i >= len(actions):
                for _ in range(2):
                    p.stepSimulation()
            else:
                env.step(actions[i])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    # playback(args)
    create_dataset(args)
    p.disconnect()


def interactive_session(env):
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)
    PBU.set_camera(45, -40, 2, (0, 0, 0))

    env.reset()
    robot = env.robot
    gripper = env.robot.gripper
    pos = np.array(robot.get_eef_position())
    rot = np.array(robot.get_eef_orientation())
    grasped = False

    rot_yaw_pos = T.quaternion_about_axis(0.01, [0, 0, 1])
    rot_yaw_neg = T.quaternion_about_axis(-0.01, [0, 0, 1])
    rot_pitch_pos = T.quaternion_about_axis(0.01, [1, 0, 0])
    rot_pitch_neg = T.quaternion_about_axis(-0.01, [1, 0, 0])

    prev_key = None
    for i in range(24000):  # at least 100 seconds
        print(env.is_success())

        prev_rot = rot.copy()
        prev_pos = pos.copy()
        keys = p.getKeyboardEvents()

        p.stepSimulation()
        if ord('c') in keys and prev_key != keys:
            if grasped:
                gripper.ungrasp()
            else:
                gripper.grasp()
            grasped = not grasped

        if p.B3G_ALT in keys and p.B3G_LEFT_ARROW in keys:
            rot = T.quaternion_multiply(rot_yaw_pos, rot)
        if p.B3G_ALT in keys and p.B3G_RIGHT_ARROW in keys:
            rot = T.quaternion_multiply(rot_yaw_neg, rot)

        if p.B3G_ALT in keys and p.B3G_UP_ARROW in keys:
            rot = T.quaternion_multiply(rot_pitch_pos, rot)
        if p.B3G_ALT in keys and p.B3G_DOWN_ARROW in keys:
            rot = T.quaternion_multiply(rot_pitch_neg, rot)

        if p.B3G_ALT not in keys and p.B3G_LEFT_ARROW in keys:
            pos[1] -= 0.005
        if p.B3G_ALT not in keys and p.B3G_RIGHT_ARROW in keys:
            pos[1] += 0.005

        if p.B3G_ALT not in keys and p.B3G_UP_ARROW in keys:
            pos[0] -= 0.005
        if p.B3G_ALT not in keys and p.B3G_DOWN_ARROW in keys:
            pos[0] += 0.005

        if ord(',') in keys:
            pos[2] += 0.005
        if ord('.') in keys:
            pos[2] -= 0.005

        if not np.all(prev_pos == pos) or not np.all(prev_rot == rot):
            robot.set_eef_position_orientation(pos, rot)

        time.sleep(1./240.)
        prev_key = keys

    p.disconnect()


if __name__ == '__main__':
    main()
