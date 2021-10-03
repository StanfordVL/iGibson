import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import igibson
from igibson.envs.igibson_env import iGibsonEnv


def get_lidar_sampling_pattern():
    lidar_vertical_low = -15 / 180.0 * np.pi
    lidar_vertical_high = 15 / 180.0 * np.pi
    lidar_vertical_n_beams = 16
    lidar_vertical_beams = np.arange(
        lidar_vertical_low,
        lidar_vertical_high + (lidar_vertical_high - lidar_vertical_low) / (lidar_vertical_n_beams - 1),
        (lidar_vertical_high - lidar_vertical_low) / (lidar_vertical_n_beams - 1),
    )

    lidar_horizontal_low = -45 / 180.0 * np.pi
    lidar_horizontal_high = 45 / 180.0 * np.pi
    lidar_horizontal_n_beams = 468
    lidar_horizontal_beams = np.arange(
        lidar_horizontal_low,
        lidar_horizontal_high,
        (lidar_horizontal_high - lidar_horizontal_low) / (lidar_horizontal_n_beams),
    )

    xx, yy = np.meshgrid(lidar_vertical_beams, lidar_horizontal_beams)
    xx = xx.flatten()
    yy = yy.flatten()

    height = 128

    x_samples = (np.tan(xx) / np.cos(yy) * height // 2 + height // 2).astype(np.int)
    y_samples = (np.tan(yy) * height // 2 + height // 2).astype(np.int)

    x_samples = x_samples.flatten()
    y_samples = y_samples.flatten()
    return x_samples, y_samples


x_samples, y_samples = get_lidar_sampling_pattern()


def generate_data_lidar(nav_env, num_samples=3):

    rgb_all = []
    lidar_all = []
    lidar_all_2 = []
    label_all = []

    point = nav_env.scene.get_random_point()[1]

    for _ in range(num_samples):
        new_point = nav_env.scene.get_random_point()[1]

        while np.linalg.norm(new_point - point) > 1:
            new_point = nav_env.scene.get_random_point()[1]

        delta_pos = new_point - point
        delta_pos = np.array([delta_pos[1], delta_pos[2], delta_pos[0]])
        # print(delta_pos)
        nav_env.robots[0].set_position(new_point)
        pano_rgb = nav_env.simulator.renderer.get_cube(mode="rgb", use_robot_camera=True)
        pano_3d = nav_env.simulator.renderer.get_cube(mode="3d", use_robot_camera=True)
        pano_seg = nav_env.simulator.renderer.get_cube(mode="seg", use_robot_camera=True)

        r3 = np.array(
            [[np.cos(-np.pi / 2), 0, -np.sin(-np.pi / 2)], [0, 1, 0], [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]]
        )
        transformatiom_matrix = np.eye(3)

        for i in range(4):
            lidar_all.append(pano_3d[i][:, :, :3].dot(transformatiom_matrix)[x_samples, y_samples] - delta_pos[None, :])
            rgb_all.append(pano_rgb[i][:, :, :3][x_samples, y_samples])
            label_all.append(pano_seg[i][:, :, 0][x_samples, y_samples] * 255.0)
            lidar_all_2.append(
                pano_3d[i][:, :, :3].dot(transformatiom_matrix)[x_samples, y_samples] * 0.9 - delta_pos[None, :]
            )
            transformatiom_matrix = r3.dot(transformatiom_matrix)

    lidar_all = np.concatenate(lidar_all, 0).astype(np.float32)
    lidar_all_2 = np.concatenate(lidar_all_2, 0).astype(np.float32)
    rgb_all = np.concatenate(rgb_all, 0).astype(np.float32)
    label_all = np.concatenate(label_all, 0).astype(np.int32)

    assert len(label_all) == len(label_all)

    direction = lidar_all - lidar_all_2
    direction = direction / (np.linalg.norm(direction, axis=1)[:, None] + 1e-5)

    print(lidar_all.shape, direction.shape, rgb_all.shape, label_all.shape)
    return lidar_all, direction, rgb_all, label_all


def generate_data_from_scene(scene_id):

    mode = "headless"
    config = os.path.join(igibson.example_path, "configs/fetch_room_rearrangement.yaml")
    nav_env = iGibsonEnv(
        config_file=config, mode=mode, scene_id=scene_id, action_timestep=1.0 / 120.0, physics_timestep=1.0 / 120.0
    )
    # data = []
    # for i in tqdm(range(5)):
    #     data.append(generate_data_lidar(nav_env))

    # lidar_all = [item[0] for item in data]
    # direction = [item[1] for item in data]
    # rgb_all = [item[2] for item in data]
    # label_all = [item[3] for item in data]

    pts, direction, color, label = generate_data_lidar(nav_env)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=3, c=color[:, :3])
    plt.show()

    # np.savez('/data2/point_cloud/data10_v2_{}.npz'.format(scene_id), lidar=lidar_all, direction=direction, rgb=rgb_all, label=label_all)


if __name__ == "__main__":
    generate_data_from_scene("Rs_int")

# scenes = []
# with open('scene_list', 'r') as f:
#     for line in f:
#         scenes.append(line.strip())

# p = Pool(2)
# p.map(generate_data_from_scene, scenes)
