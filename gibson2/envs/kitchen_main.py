import h5py
import json

import os
import pybullet as p
import time
import numpy as np
import gibson2.external.pybullet_tools.transformations as T
import gibson2.external.pybullet_tools.utils as PBU

import gibson2.envs.kitchen.plan_utils as PU
import gibson2.envs.kitchen.skills as skills
from gibson2.envs.kitchen.envs import env_factory
from gibson2.envs.kitchen.env_utils import pose_to_array, pose_to_action_euler, pose_to_action_axis_vector
import cv2

"""
Task plans -> skill parameters
Parameterized skill library
Skills + parameters -> joint-space motion plan
Motion plan -> task-space path
task-space path -> gripper actuation
"""

ACTION_NOISE = (0.01, 0.01, 0.01, np.pi / 16, np.pi / 16, np.pi / 16)


def execute_planned_path(env, path, noise=None):
    """Execute a planned path an relabel actions."""

    # all_obs = []
    actions = []
    rewards = []
    states = []
    task_specs = []

    for i in range(len(path)):
        task_specs.append(env.task_spec)
        tpose = path.arm_path[i]
        grip = path.gripper_path[i]

        cpose = pose_to_array(env.robot.get_eef_position_orientation())
        tpose = pose_to_array(tpose)

        action = np.zeros(env.action_dimension)
        action[-1] = grip
        action[:-1] = pose_to_action_euler(cpose, tpose, max_dpos=env.MAX_DPOS, max_drot=env.MAX_DROT)
        # action[:-1] = pose_to_action_axis_vector(cpose, tpose, max_dpos=env.MAX_DPOS, max_drot=env.MAX_DROT)
        if noise is not None:
            assert len(noise) == (env.action_dimension - 1)
            noise_arr = np.array(noise)
            action[:6] += np.clip(np.random.randn(len(noise)) * noise_arr, -noise_arr * 2, noise_arr * 2)
        actions.append(action)
        states.append(env.serialized_world_state)
        # all_obs.append(env.get_observation())

        env.step(action)
        rewards.append(float(env.is_success()))

    # all_obs.append(env.get_observation())
    actions.append(actions[-1])
    rewards.append(float(env.is_success()))
    states.append(env.serialized_world_state)
    task_specs.append(env.task_spec)

    # all_obs = dict((k, np.array([all_obs[i][k] for i in range(len(all_obs))])) for k in all_obs[0])
    states = np.stack(states)
    actions = np.stack(actions)
    rewards = np.stack(rewards)
    task_specs = np.stack(task_specs)
    return {"states": states, "actions": actions, "rewards": rewards, "task_specs": task_specs}


class Buffer(object):
    def __init__(self):
        self.data = dict()

    def append(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def aggregate(self):
        for k, v in self.data.items():
            if isinstance(v[0], dict):
                self.data[k] = dict((k, np.concatenate([v[i][k] for i in range(len(v))], axis=0)) for k in v[0])
            else:
                self.data[k] = np.concatenate(v, axis=0)
        return self.data


def get_demo_can_to_drawer(env, perturb=False):
    buffer = Buffer()
    env.reset()

    drawer_grasp_pose = (
        [0.3879213,  0.0072391,  0.71218301],
        T.quaternion_multiply(T.quaternion_about_axis(np.pi, axis=[0, 0, 1]), T.quaternion_about_axis(np.pi / 2, axis=[1, 0, 0]))
    )
    path = skills.plan_skill_open_prismatic(
        env.planner,
        obstacles=env.obstacles,
        grasp_pose=drawer_grasp_pose,
        reach_distance=0.05,
        retract_distance=0.25,
        joint_resolutions=(0.25, 0.25, 0.25, 0.2, 0.2, 0.2)
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))

    can_grasp_pose = ((0.03, -0.005, 1.06), (0, 0, 1, 0))
    path = skills.plan_skill_grasp(
        env.planner,
        obstacles=env.obstacles,
        grasp_pose=can_grasp_pose,
        reach_distance=0.05,
        lift_height=0.1,
        joint_resolutions=(0.1, 0.1, 0.1, 0.2, 0.2, 0.2)
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))

    can_drop_pose = ((0.469, 0, 0.952), (0, 0, 1, 0))
    path = skills.plan_skill_place(
        env.planner,
        obstacles=env.obstacles,
        holding=env.objects["can"].body_id,
        object_target_pose=can_drop_pose,
        joint_resolutions=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    )

    path.append_pause(30)
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))
    return buffer.aggregate()


def get_demo_lift_can(env, perturb=False):
    buffer = Buffer()
    env.reset()

    can_pos = np.array(env.objects["can"].get_position())
    can_pos[0] += 0.02
    can_grasp_pose = (tuple(can_pos.tolist()), (0, 0, 1, 0))
    path = skills.plan_skill_grasp(
        env.planner,
        obstacles=env.obstacles,
        grasp_pose=can_grasp_pose,
        reach_distance=0.05,
        lift_height=0.2,
        joint_resolutions=(0.1, 0.1, 0.1, 0.2, 0.2, 0.2)
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))
    return buffer.aggregate()


def get_demo_pour(env, perturb=False):
    buffer = Buffer()
    env.reset()

    mug_pos = np.array(env.objects["mug_red"].get_position())
    mug_pos[0] += 0.02
    mug_grasp_pose = (tuple(mug_pos.tolist()), (0, 0, 1, 0))
    path = skills.plan_skill_grasp(
        env.planner,
        obstacles=env.obstacles,
        grasp_pose=mug_grasp_pose,
        reach_distance=0.05,
        lift_height=0.1,
        joint_resolutions=(0.1, 0.1, 0.1, 0.2, 0.2, 0.2),
        lift_speed=0.015
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))

    bowl_pos = np.array(env.objects["bowl_red"].get_position())
    pour_pos = bowl_pos + np.array([0, 0.05, 0.2])
    pour_pose = (tuple(pour_pos.tolist()), PBU.multiply_quats(T.quaternion_about_axis(np.pi * 2 / 3, (1, 0, 0)), env.objects["mug_red"].get_orientation()))
    path = skills.plan_skill_pour(
        env.planner,
        obstacles=env.obstacles,
        object_target_pose=pour_pose,
        pour_angle=np.pi / 4,
        holding=env.objects["mug_red"].body_id,
        joint_resolutions=(0.1, 0.1, 0.1, 0.2, 0.2, 0.2),
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))

    return buffer.aggregate()


def get_demo_arrange(env, perturb=False):
    buffer = Buffer()

    env.reset()
    # orn = T.quaternion_from_euler(0, np.pi / 2, np.pi * float(np.random.rand(1)) * 2)
    orn = T.quaternion_from_euler(0, np.pi / 2, 0)  # top-down grasp
    can_grasp_pose = PU.compute_grasp_pose(
        object_frame=env.objects["can"].get_position_orientation(), grasp_orientation=orn, grasp_distance=0.02)

    path = skills.plan_skill_grasp(
        env.planner,
        obstacles=env.obstacles,
        grasp_pose=can_grasp_pose,
        reach_distance=0.05,
        lift_height=0.4,
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))

    target_pos = np.array(env.objects["target"].get_position())
    target_pos[2] = PBU.stable_z(env.objects["can"].body_id, env.objects["target"].body_id)
    can_place_pose = (target_pos, env.objects["can"].get_orientation())

    path = skills.plan_skill_place(
        env.planner,
        obstacles=env.obstacles,
        object_target_pose=can_place_pose,
        holding=env.objects["can"].body_id,
        retract_distance=0.1,
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))

    path = skills.plan_move_to(
        env.planner,
        obstacles=env.obstacles,
        target_pose=(env.planner.ref_robot.get_eef_position() + np.array([0, 0, 0.03]), T.quaternion_from_euler(0, np.pi / 2, 0)),
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))

    return buffer.aggregate()


def get_demo_arrange_hard(env, perturb=False):
    buffer = Buffer()

    env.reset()

    # orn = T.quaternion_from_euler(0, np.pi / 2, np.pi * float(np.random.rand(1)) * 2)
    src_object = env.objects.names[env.task_spec[0]]
    tgt_object = env.objects.names[env.task_spec[1]]
    orn = T.quaternion_from_euler(0, np.pi / 2, 0)
    can_grasp_pose = PU.compute_grasp_pose(
        object_frame=env.objects[src_object].get_position_orientation(), grasp_orientation=orn, grasp_distance=0.02)

    path = skills.plan_skill_grasp(
        env.planner,
        obstacles=env.obstacles,
        grasp_pose=can_grasp_pose,
        reach_distance=0.05,
        lift_height=0.2,
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))

    target_pos = np.array(env.objects[tgt_object].get_position())
    target_pos[2] = PBU.stable_z(env.objects[src_object].body_id, env.objects[tgt_object].body_id) + 0.01
    place_pose = (target_pos, env.objects[src_object].get_orientation())

    path = skills.plan_skill_place(
        env.planner,
        obstacles=env.obstacles,
        object_target_pose=place_pose,
        holding=env.objects[src_object].body_id,
        retract_distance=0.1,
    )
    buffer.append(**execute_planned_path(env, path, noise=ACTION_NOISE if perturb else None))

    return buffer.aggregate()


def record_demos(args):
    env_kwargs = dict(
        num_sim_per_step=5,
        sim_time_step=1./240.
    )
    env = env_factory("TableTopArrangeHard", **env_kwargs, use_planner=True, hide_planner=True, use_gui=args.gui)

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
            buffer = get_demo_arrange_hard(env, perturb=args.perturb_demo)
        except PU.NoPlanException as e:
            print(e)
            continue
        for _ in range(30):
            p.stepSimulation()
        total_i += 1
        if not env.is_success():
            print("{}/{}".format(success_i, total_i))
            continue

        f_demo_grp = f_sars_grp.create_group("demo_{}".format(success_i))
        f_demo_grp.create_dataset("states", data=buffer["states"])
        f_demo_grp.create_dataset("actions", data=buffer["actions"])
        f_demo_grp.create_dataset("task_specs", data=buffer["task_specs"])
        f_demo_grp.create_dataset("rewards", data=buffer["rewards"])

        # for k in buffer["obs"]:
        #     f_demo_grp.create_dataset("obs/{}".format(k), data=buffer["obs"][k][:-1])
        #     f_demo_grp.create_dataset("next_obs/{}".format(k), data=buffer["obs"][k][1:])
        success_i += 1
        print("{}/{}".format(success_i, total_i))
    f.close()


def extract_dataset(args, extract_by_action_playback=False):
    f = h5py.File(args.file, 'r')
    demos = list(f["data"].keys())

    extract_name = 'states.hdf5' if args.extract_name is None else args.extract_name
    out_path = os.path.join(os.path.dirname(args.file), extract_name)

    out_f = h5py.File(out_path)
    f_grp = out_f.create_group("data")

    env_args = json.loads(f["data"].attrs["env_args"])
    env_args["env_kwargs"]["obs_image"] = args.extract_image
    env_args["env_kwargs"]["obs_depth"] = args.extract_depth
    env_args["env_kwargs"]["obs_match"] = args.extract_matches
    env_args["env_kwargs"]["obs_segmentation"] = args.extract_segmentation
    env_args["env_kwargs"]["camera_height"] = args.width
    env_args["env_kwargs"]["camera_width"] = args.height

    f_grp.attrs["env_args"] = json.dumps(env_args)

    env = env_factory(env_args["env_name"], **env_args["env_kwargs"])
    env.reset()

    for demo_id in demos:
        states = f["data/{}/states".format(demo_id)][:]
        task_spec = f["data/{}/task_specs".format(demo_id)][0]
        env.reset_to(states[0], return_obs=False)
        env.set_goal(task_specs=task_spec)

        actions = f["data/{}/actions".format(demo_id)][:]

        new_states = []
        obs = []
        for i in range(len(states) - 1):
            obs.append(env.get_observation())
            new_states.append(env.serialized_world_state)  # useful when extracting by playback actions
            if extract_by_action_playback:
                env.step(actions[i])
            else:
                env.reset_to(states[i + 1], return_obs=False)

        new_states.append(env.serialized_world_state)
        obs.append(env.get_observation())

        # aggregate extracted states and observations
        new_states = np.stack(new_states)
        obs = dict((k, np.stack([obs[i][k] for i in range(len(obs))])) for k in obs[0])

        demo_grp = f_grp.create_group(demo_id)
        demo_grp.attrs["num_samples"] = new_states.shape[0] - 1

        # create sars pairs
        demo_grp.create_dataset("actions", data=actions[:-1])
        demo_grp.create_dataset("rewards", data=f["data/{}/rewards".format(demo_id)][:-1])
        demo_grp.create_dataset("task_specs", data=f["data/{}/task_specs".format(demo_id)][:-1])
        demo_grp.create_dataset("states", data=new_states[:-1])
        for k in obs:
            print(k)
            demo_grp.create_dataset("obs/{}".format(k), data=obs[k][:-1])
            demo_grp.create_dataset("next_obs/{}".format(k), data=obs[k][1:])

        print("{} success: {}".format(demo_id, env.is_success()))


def playback(args):
    f = h5py.File(args.file, 'r')
    env_args = json.loads(f["data"].attrs["env_args"])

    env = env_factory(env_args["env_name"], **env_args["env_kwargs"], use_gui=args.gui)
    env.reset()
    demos = list(f["data"].keys())
    for demo_id in demos:
        states = f["data/{}/states".format(demo_id)][:]
        task_spec = f["data/{}/task_specs".format(demo_id)][0]
        env.reset_to(states[0])
        env.set_goal(task_specs=task_spec)
        actions = f["data/{}/actions".format(demo_id)][:]

        for i in range(len(actions)):
            env.step(actions[i])
        print("success: {}".format(env.is_success()))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["demo", "playback", "extract", "extract_by_action_playback"]
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--extract_name",
        type=str,
        default="states.hdf5"
    )

    parser.add_argument(
        "--extract_image",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--extract_depth",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--extract_matches",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--extract_segmentation",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--width",
        default=128,
        type=int
    )
    parser.add_argument(
        "--height",
        default=128,
        type=int
    )

    parser.add_argument(
        "--n",
        type=int,
        default=100
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--perturb_demo",
        action="store_true",
        default=False
    )


    args = parser.parse_args()

    np.random.seed(0)
    if args.mode == 'playback':
        playback(args)
    elif args.mode == 'demo':
        record_demos(args)
    else:
        extract_dataset(args, extract_by_action_playback=(args.mode == "extract_by_action_playback"))
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
