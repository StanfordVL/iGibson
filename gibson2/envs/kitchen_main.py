import h5py
import json

import os
import pybullet as p
import time
import numpy as np
import gibson2.external.pybullet_tools.transformations as T
import gibson2.external.pybullet_tools.utils as PBU

import gibson2.envs.kitchen.plan_utils as PU
from gibson2.envs.kitchen.plan_utils import Buffer
import gibson2.envs.kitchen.skills as skills
from gibson2.envs.kitchen.envs import env_factory, EnvSkillWrapper


"""
Task plans -> skill parameters
Parameterized skill library
Skills + parameters -> joint-space motion plan
Motion plan -> task-space path
task-space path -> gripper actuation
"""

ACTION_NOISE = (0.01, 0.01, 0.01, np.pi / 16, np.pi / 16, np.pi / 16)


def record_demos(args):
    env_kwargs = dict(
        num_sim_per_step=5,
        sim_time_step=1./240.,
    )
    env = env_factory(args.env, **env_kwargs, use_planner=True, hide_planner=True, use_gui=args.gui, use_skills=True)

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

    n_success = 0
    n_total = 0
    n_kept = 0
    while n_kept < args.n:
        t = time.time()
        env.reset()
        try:
            # buffer, plan_exception = env.get_demo_suboptimal(noise=ACTION_NOISE if args.perturb_demo else None)
            skeleton = env.get_random_skeleton(10)
            buffer, plan_exception = env.execute_skeleton(skeleton)
            if plan_exception is not None:
                if not args.keep_interrupted_demos:
                    continue
        except PU.NoPlanException as e:
            print(e)
            continue

        if args.gui:
            for _ in range(100):
                p.stepSimulation()

        elapsed = time.time() - t
        traj_len = len(list(buffer.values())[0])

        n_total += 1
        if not env.is_success():
            n_kept += 1 if args.keep_failed_demos else 0
            print("{}/{}/{}, time={:.5f}, len={}".format(n_success, n_kept, n_total, elapsed, traj_len))
            if not args.keep_failed_demos:
                continue
        else:
            n_success += 1
            n_kept += 1
            print("{}/{}/{}, time={:.5f}, len={}".format(n_success, n_kept, n_total, elapsed, traj_len))

        f_demo_grp = f_sars_grp.create_group("demo_{}".format(n_kept - 1))
        for k in buffer:
            if isinstance(buffer[k], dict):
                for kk in buffer[k]:
                    f_demo_grp.create_dataset(k + "/" + kk, data=buffer[k][kk])
            else:
                f_demo_grp.create_dataset(k, data=buffer[k])
    f.close()


def extract_dataset(args):
    f = h5py.File(args.file, 'r')
    demos = list(f["data"].keys())

    extract_name = 'states.hdf5' if args.extract_name is None else args.extract_name
    out_path = os.path.join(os.path.dirname(args.file), extract_name)

    if os.path.exists(out_path):
        os.remove(out_path)
    out_f = h5py.File(out_path)
    f_grp = out_f.create_group("data")

    env_args = json.loads(f["data"].attrs["env_args"])
    env_args["env_kwargs"]["obs_image"] = args.extract_image
    env_args["env_kwargs"]["obs_depth"] = args.extract_depth
    env_args["env_kwargs"]["obs_segmentation"] = args.extract_segmentation
    env_args["env_kwargs"]["camera_height"] = args.width
    env_args["env_kwargs"]["camera_width"] = args.height
    env_args["env_kwargs"]["obs_crop"] = args.extract_crop

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
            if args.extract_by_action_playback:
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
        org_f_grp = f["data/{}".format(demo_id)]
        for k in org_f_grp.keys():
            if k not in ["states", "obs", "next_obs"]:
                demo_grp.create_dataset(k, data=org_f_grp[k][:-1])

        demo_grp.create_dataset("states", data=new_states[:-1])

        for k in obs:
            demo_grp.create_dataset("obs/{}".format(k), data=obs[k][:-1])
            demo_grp.create_dataset("next_obs/{}".format(k), data=obs[k][1:])

        print("{} success: {}".format(demo_id, env.is_success()))


def extract_dataset_skills(args):
    f = h5py.File(args.file, 'r')
    demos = list(f["data"].keys())

    extract_name = 'states.hdf5' if args.extract_name is None else args.extract_name
    out_path = os.path.join(os.path.dirname(args.file), extract_name)

    if os.path.exists(out_path):
        os.remove(out_path)
    out_f = h5py.File(out_path)
    f_grp = out_f.create_group("data")

    env_args = json.loads(f["data"].attrs["env_args"])
    env_args["env_kwargs"]["obs_image"] = args.extract_image
    env_args["env_kwargs"]["obs_depth"] = args.extract_depth
    env_args["env_kwargs"]["obs_segmentation"] = args.extract_segmentation
    env_args["env_kwargs"]["camera_height"] = args.width
    env_args["env_kwargs"]["camera_width"] = args.height
    env_args["env_kwargs"]["obs_crop"] = args.extract_crop
    if not env_args["env_name"].endswith("Skill"):
        env_args["env_name"] += "Skill"

    env = env_factory(env_args["env_name"], **env_args["env_kwargs"])
    env_args["skill_info"] = dict()
    env_args["skill_info"]["skill_names"] = env.skill_lib.skill_names
    env_args["skill_info"]["skill_action_dimension"] = env.skill_lib.action_dimension

    f_grp.attrs["env_args"] = json.dumps(env_args)

    env.reset()
    for demo_id in demos:
        mask = f["data/{}/skill_begin".format(demo_id)][:].astype(np.bool)
        mask[-1] = True
        mask_inds = np.where(mask)[0]
        skill_len = mask_inds[1:] - mask_inds[:-1]
        # only extract first @skill_frame_ratio fraction of the frames for each skill
        for mi, sl in zip(mask_inds[:-1], skill_len):
            mask[mi: mi + max(1, int(sl * args.skill_frame_ratio))] = True

        # extract the last @skill_frame_ratio fraction of the demo for goals
        # mask[mask_inds[-2] + int(skill_len[-1] * (1 - args.skill_frame_ratio)):] = True

        states = f["data/{}/states".format(demo_id)][mask, ...]
        task_spec = f["data/{}/task_specs".format(demo_id)][0]
        env.reset_to(states[0], return_obs=False)
        env.set_goal(task_specs=task_spec)
        skill_params = f["data/{}/skill_params".format(demo_id)][mask, ...]
        object_index = f["data/{}/skill_object_index".format(demo_id)][mask, ...]
        actions = np.concatenate((skill_params, object_index), axis=1)

        new_states = []
        obs = []
        for i in range(len(states) - 1):
            obs.append(env.get_observation())
            new_states.append(env.serialized_world_state)  # useful when extracting by playback actions
            if args.extract_by_action_playback:
                env.step(actions[i])
            else:
                env.reset_to(states[i + 1], return_obs=False)

        new_states.append(env.serialized_world_state)
        obs.append(env.get_observation())

        # aggregate extracted states and observations
        new_states = np.stack(new_states)
        obs = dict((k, np.stack([obs[i][k] for i in range(len(obs))])) for k in obs[0])

        demo_grp = f_grp.create_group(demo_id)
        # create sars pairs
        org_f_grp = f["data/{}".format(demo_id)]

        ep_indices = np.arange(new_states.shape[0])
        assert len(ep_indices) >= 1
        # optionally extract next obs
        if args.extract_next_obs:
            ep_indices = ep_indices[:-1]

        for k in org_f_grp.keys():
            if k == "skill_param_dict":
                # skill_param_dict is a dictionary
                for kk in org_f_grp[k].keys():
                    demo_grp.create_dataset("{}/{}".format(k, kk), data=org_f_grp[k][kk][mask, ...][ep_indices])
            elif k not in ["states", "actions"]:
                demo_grp.create_dataset(k, data=org_f_grp[k][mask, ...][ep_indices])
        demo_grp.create_dataset("states", data=new_states[ep_indices])
        demo_grp.create_dataset("actions", data=actions[ep_indices])

        for k in obs:
            demo_grp.create_dataset("obs/{}".format(k), data=obs[k][ep_indices])
            if args.extract_next_obs:
                demo_grp.create_dataset("next_obs/{}".format(k), data=obs[k][1:])

        demo_grp.attrs["num_samples"] = demo_grp["states"].shape[0]

        print("{} success: {}".format(demo_id, env.is_success()))


def playback(args):
    f = h5py.File(args.file, 'r')
    env_args = json.loads(f["data"].attrs["env_args"])

    env = env_factory(env_args["env_name"], **env_args["env_kwargs"], use_gui=args.gui)
    # env.reset()
    demos = list(f["data"].keys())
    for demo_id in demos:
        env.reset()
        states = f["data/{}/states".format(demo_id)][:]
        task_spec = f["data/{}/task_specs".format(demo_id)][0]
        env.reset_to(states[0])
        env.set_goal(task_specs=task_spec)
        actions = f["data/{}/actions".format(demo_id)][:]

        for i in range(len(actions)):
            env.step(actions[i])
        print("success: {}".format(env.is_success()))


def playback_compare(args):
    f = h5py.File(args.file, 'r')
    env_args = json.loads(f["data"].attrs["env_args"])

    env = env_factory(env_args["env_name"], **env_args["env_kwargs"], use_gui=args.gui)
    # env.reset()
    demos = list(f["data"].keys())
    for demo_id in demos:
        env.reset()
        states = f["data/{}/states".format(demo_id)][:]
        task_spec = f["data/{}/task_specs".format(demo_id)][0]
        env.reset_to(states[0])
        env.set_goal(task_specs=task_spec)
        actions = f["data/{}/actions".format(demo_id)][:]
        rollout_video = []
        for i in range(len(actions)):
            rollout_video.append(env.render())
            env.step(actions[i])

        org_video = []
        env.reset()
        for i in range(len(actions)):
            env.reset_to(states[i])
            org_video.append(env.render())

        print("success: {}".format(env.is_success()))


def view_scene(args):
    env = env_factory(
        args.env,
        num_sim_per_step=5,
        sim_time_step=1./240.,
        use_planner=True,
        use_skills=True,
        use_gui=args.gui
    )
    env.reset()

    while True:
        p.stepSimulation()
        time.sleep(1/240.)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["demo", "playback", "extract", "extract_skill", "view"]
    )
    parser.add_argument(
        "--extract_by_action_playback",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--skill_frame_ratio",
        default=0.0,
        type=float
    )

    parser.add_argument(
        "--keep_failed_demos",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--keep_interrupted_demos",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--file",
        type=str,
    )

    parser.add_argument(
        "--env",
        type=str,
    )

    parser.add_argument(
        "--extract_name",
        type=str,
        default="states.hdf5"
    )

    parser.add_argument(
        "--extract_image",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--extract_crop",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--extract_depth",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--extract_segmentation",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--extract_next_obs",
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
        default=10
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
    parser.add_argument(
        "--seed",
        default=0,
        type=int
    )

    args = parser.parse_args()

    if args.keep_interrupted_demos:
        assert args.keep_failed_demos

    np.random.seed(args.seed)
    if args.mode == 'playback':
        playback(args)
    elif args.mode == 'demo':
        record_demos(args)
    elif args.mode == "extract":
        extract_dataset(args)
    elif args.mode == "extract_skill":
        extract_dataset_skills(args)
    else:
        view_scene(args)
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
