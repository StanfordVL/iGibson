import h5py
import json
import os
import numpy as np
import argparse
import glob
from gibson2.envs.kitchen.envs import env_factory
import matplotlib.pyplot as plt
import cv2
import io
import imageio
import re


def get_mug_faucet(env, task_spec, state_place, state_grasp):
    env.reset()
    env.set_goal(task_specs=task_spec)
    env.reset_to(state_place)
    mug_pos = np.array(env.objects["mug"].get_position()) - np.array(env.objects["faucet_milk"].get_position())
    env.reset_to(state_grasp)
    success = env.is_success_all_tasks()["fill_mug"]
    return mug_pos, success


def get_pour_bowl(env, task_spec, state_pour):
    env.reset()
    env.set_goal(task_specs=task_spec)
    env.reset_to(state_pour)
    success = env.is_success_all_tasks()["fill_bowl"]
    mug_pos = np.array(env.objects["mug"].get_position()) - np.array(env.objects["bowl"].get_position())
    return mug_pos, success


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_coffee_results(env, states_f):
    demos = list(states_f["data"].keys())
    mug_fill_success = []
    mug_pos = []
    mug_pour_orn = []
    bowl_fill_success = []
    mug_pos_to_bowl = []
    for demo_id in demos:
        # demo_id = "demo_0"
        task_spec = states_f["data/{}/task_specs".format(demo_id)][0]
        states = states_f["data/{}/states".format(demo_id)]
        actions = states_f["data/{}/actions".format(demo_id)]

        mask = states_f["data/{}/skill_begin".format(demo_id)][:].astype(np.bool)
        mask[-1] = True
        mask_inds = np.where(mask)[0]
        pos, success = get_mug_faucet(
            env,
            task_spec=task_spec,
            state_place=states[mask_inds[2]],
            state_grasp=states[mask_inds[3]]
        )
        mug_pos.append(pos)
        mug_fill_success.append(success)

        pos, bowl_success = get_pour_bowl(env, task_spec, states[mask_inds[-1]])
        pour_action = actions[mask_inds[-2], 20]
        mug_pour_orn.append(pour_action)
        bowl_fill_success.append(bowl_success)
        mug_pos_to_bowl.append(pos)

    mug_pos = np.array(mug_pos)
    mug_fill_success = np.array(mug_fill_success)
    return mug_pos, mug_fill_success, np.array(mug_pour_orn), np.array(bowl_fill_success), np.array(mug_pos_to_bowl)


def draw_mug_positions(mug_pos, mug_fill_success, epoch, save_path=None):
    colors = {True: [0, 0, 1], False: [1, 0, 0]}
    success_colors = np.array([colors[c] for c in mug_fill_success])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(mug_pos[:, 1], mug_pos[:, 0], c=success_colors)
    ax.scatter([0, 0.15], [0, 0], s=[200, 200], c='grey')
    ax.set_xlim([-0.075, 0.25])
    ax.set_ylim([-0.07, 0.07])
    ax.text(-0.05, 0.06, "epoch={}, success={:.3f}, no_plan={:.3f}".format(
        epoch, np.mean(mug_fill_success.astype(np.float32)), len(mug_fill_success) / 50))
    if save_path is not None:
        fig.savefig(save_path)
    return get_img_from_fig(fig, dpi=80)


def draw_pour_orns(pour_orn, mug_pos_to_bowl, success, epoch, save_path=None):
    colors = {True: [0, 0, 1], False: [1, 0, 0]}
    success_colors = np.array([colors[c] for c in success])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bowl_dist = np.linalg.norm(mug_pos_to_bowl[:, :2], axis=1)
    ax.scatter(pour_orn, bowl_dist, c=success_colors)
    ax.set_xlim([0, np.pi])
    ax.set_ylim([0, 0.15])
    ax.text(-0.05, 0.12, "epoch={}, success={:.3f}, no_plan={:.3f}".format(
        epoch, np.mean(success.astype(np.float32)), len(success) / 50))
    if save_path is not None:
        fig.savefig(save_path)
    return get_img_from_fig(fig, dpi=80)


def vis_all(states_folder, video_path):
    all_files = glob.glob(os.path.join(states_folder, "*.hdf5"))

    f = h5py.File(all_files[0], 'r')
    env_args = json.loads(f["data"].attrs["env_args"])
    env = env_factory(env_args["env_name"], **env_args["env_kwargs"], use_gui=False)
    f.close()
    video_writer = imageio.get_writer(video_path, fps=20)

    file_epoch = [int(re.split("[._]", os.path.basename(fn))[2]) for fn in all_files]
    inds = np.argsort(file_epoch)
    file_epoch = sorted(file_epoch)
    all_files = [all_files[i] for i in inds]
    for epoch, h5_path in zip(file_epoch, all_files):
        print(h5_path)
        f = h5py.File(h5_path, 'r')
        mug_pos, mug_fill_success, pour_orn, bowl_fill_success, mug_pos_to_bowl = get_coffee_results(env, f)
        im_mug = draw_mug_positions(mug_pos, mug_fill_success, epoch=epoch)
        im_pour = draw_pour_orns(pour_orn, mug_pos_to_bowl, bowl_fill_success, epoch=epoch)
        im = np.concatenate([im_mug, im_pour], axis=1)
        f.close()
        video_writer.append_data(im)

    video_writer.close()


def vis_image(hdf5_file, image_path):
    f = h5py.File(hdf5_file, 'r')
    env_args = json.loads(f["data"].attrs["env_args"])
    env = env_factory(env_args["env_name"], **env_args["env_kwargs"], use_gui=False)

    mug_pos, mug_fill_success, pour_orn, bowl_fill_success, mug_pos_to_bowl = get_coffee_results(env, f)
    f.close()
    draw_mug_positions(mug_pos, mug_fill_success, epoch=0, save_path=image_path + "_place.png")
    draw_pour_orns(pour_orn, mug_pos_to_bowl, bowl_fill_success, epoch=0, save_path=image_path + "_pour.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
    )

    parser.add_argument(
        "--output_path",
        type=str,
    )

    args = parser.parse_args()
    # vis_all(args.input_path, args.output_path)
    vis_image(args.input_path, args.output_path)