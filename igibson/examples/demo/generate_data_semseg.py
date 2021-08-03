import argparse
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from igibson.envs.igibson_env import iGibsonEnv
from igibson.objects.ycb_object import YCBObject
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

mode = "headless"
config = "/home/fei/Development/gibsonv2/igibson/examples/configs/fetch_room_rearrangement.yaml"
scene_id = sys.argv[1]
nav_env = iGibsonEnv(
    config_file=config, mode=mode, scene_id=scene_id, action_timestep=1.0 / 120.0, physics_timestep=1.0 / 120.0
)


def generate_data(nav_env):
    point = nav_env.scene.get_random_point()[1]
    nav_env.robots[0].set_position(point)
    pano_rgb = nav_env.simulator.renderer.get_equi(mode="rgb", use_robot_camera=True)
    pano_seg = nav_env.simulator.renderer.get_equi(mode="seg", use_robot_camera=True)
    pano_3d = nav_env.simulator.renderer.get_equi(mode="3d", use_robot_camera=True)
    depth = np.linalg.norm(pano_3d[:, :, :3], axis=2)
    theta = -np.arange(-np.pi / 2, np.pi / 2, np.pi / 512.0)[:, None]
    phi = np.arange(-np.pi, np.pi, 2 * np.pi / 1024)[None, :]
    # depth = depth / np.cos(theta)
    x = np.cos(theta) * np.sin(phi) * depth
    y = np.sin(theta) * depth
    z = np.cos(theta) * np.cos(phi) * depth

    mask = np.random.uniform(size=(512, 1024)) > 0.98

    pts = (np.stack([x[mask], y[mask], z[mask]]).T).astype(np.float32)
    color = (pano_rgb[mask][:, :3]).astype(np.float32)
    label = (pano_seg[mask][:, 0]).astype(np.int32)

    assert len(pts) == len(label)

    return pts, color, label


pts, color, label = generate_data(nav_env)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=3, c=color[:, :3])
plt.show()
