import os
import cv2

import h5py
import numpy as np
from igibson import simulator

from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1024
MAX_NUM_FRAMES = 10000
FRAME_BATCH_SIZE = 5


class GenerateDataset(object):
    def __init__(self, h5py_file):
        self.h5py_file = h5py_file
        self.rgb_dataset = None
        self.depth_dataset = None
        self.curr_frame_idx = 0
        self.frame_count = 0
        self.prev_frame_count = 0

        self.sim = Simulator(
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        )

        scene = InteractiveIndoorScene(scene_id="Ihlen_1_int")
        self.sim.import_scene(scene)
        self.floor = self.sim.scene.get_random_floor()
        self.create_datasets()

    def create_datasets(self):
        self.rgb_dataset = self.h5py_file.create_dataset(
            "/rgb",
            (MAX_NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 4),
            dtype=np.float32,
            compression="lzf",
            chunks=(min(MAX_NUM_FRAMES, FRAME_BATCH_SIZE),
                    IMAGE_HEIGHT, IMAGE_WIDTH, 4),
        )

        self.depth_dataset = self.h5py_file.create_dataset(
            "/depth",
            (MAX_NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 4),
            dtype=np.float32,
            compression="lzf",
            chunks=(min(MAX_NUM_FRAMES, FRAME_BATCH_SIZE),
                    IMAGE_HEIGHT, IMAGE_WIDTH, 4),
        )

        self.camera_extrinsics_dataset = self.h5py_file.create_dataset(
            "/camera_extrinsics",
            (MAX_NUM_FRAMES, 4, 4),
            dtype=np.float32,
            compression="lzf",
            chunks=(min(MAX_NUM_FRAMES, FRAME_BATCH_SIZE), 4, 4),
        )

        self.camera_intrinsics_dataset = self.h5py_file.create_dataset(
            "/camera_intrinsics",
            (3, 3),
            dtype=np.float32,
        )

        self.create_caches()

    def create_caches(self):
        self.rgb_dataset_cache = np.zeros(
            (FRAME_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
        self.depth_dataset_cache = np.zeros(
            (FRAME_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
        self.camera_extrinsics_dataset_cache = np.zeros(
            (FRAME_BATCH_SIZE, 4, 4), dtype=np.float32)

    def write_to_file(self):
        new_lines = self.frame_count - self.prev_frame_count
        self.prev_frame_count = self.frame_count

        if new_lines <= 0:
            return

        start_pos = self.frame_count - new_lines
        self.rgb_dataset[start_pos: self.frame_count] = self.rgb_dataset_cache[:new_lines]
        self.depth_dataset[start_pos: self.frame_count] = self.depth_dataset_cache[:new_lines]
        self.camera_extrinsics_dataset[start_pos:
                                       self.frame_count] = self.camera_extrinsics_dataset_cache[:new_lines]
        self.curr_frame_idx = 0

    def get_shortest_path(self, prev_point, next_point):
        initial_path = self.sim.scene.get_shortest_path(
            self.floor, prev_point[:2], next_point[:2], True)[0]
        interpolated_points = []
        for i in range(1, len(initial_path)):
            temp_points = self.sim.scene.get_shortest_path(
                self.floor, initial_path[i-1], initial_path[i], True)[0]
            for new_point in temp_points:
                interpolated_points.append(new_point)

        path_length = len(interpolated_points)
        z_diff = (next_point[2] - prev_point[2]) / float(path_length)
        z_path = np.array([prev_point[2] + (z_diff * i)
                          for i in range(path_length)]).reshape((1, path_length))
        output_path = np.concatenate(
            (np.array(interpolated_points), z_path.T), axis=1)

        return output_path

    def generate(self):
        # source, target. camera_up
        positions = [[[-5, 1.8, 2], [0.9, 0.9, 0.6], [0, 0, 1]],
                     [[0.9, 0.9, 0.6], [3.5, 0, 0.6], [0, 0, 1]],
                     [[3.5, 0, 0.6], [4, 1.6, 1.3], [0, 0, 1]],
                     [[5, 1.9, 1.4], [1, 0.3, 1.4], [0, 0, 1]],
                     [[1, 0.3, 1.4], [-2.3, 1.9, 1.4], [0, 0, 1]],
                     [[-2.3, 1.9, 1.4], [-2.2, 5.5, 1.4], [0, 0, 1]],
                     [[-2.2, 5.5, 1.4], [-2.2, 4.9, 1.4], [0, 0, 1]],
                     [[-2.2, 4.9, 1.4], [-0.7, 4.6, 1.3], [0, 0, 1]],
                     [[-0.7, 4.6, 1.3], [-0.5, 7.6, 1.3], [0, 0, 1]],
                     [[-0.5, 7.6, 1.3], [0, 7.7, 1.3], [0, 0, 1]],
                     [[0, 7.7, 1.3], [0, 0, 1.3], [0, 0, 1]]
                     ]

        for position in positions:
            prev_point = position[0]
            next_point = position[1]
            steps = self.get_shortest_path(prev_point, next_point)

            for step in steps:
                if self.frame_count == MAX_NUM_FRAMES:
                    break

                x, y, z = step[0], step[1], step[2]
                tar_x, tar_y, tar_z = next_point[0], next_point[1], next_point[2]

                self.sim.renderer.set_camera(
                    [x, y, z], [tar_x, tar_y, tar_z], position[2])
                frames = self.sim.renderer.render(modes=("rgb", "3d"))

                # Render 3d points as depth map
                depth = np.linalg.norm(frames[1][:, :, :3], axis=2)
                depth /= depth.max()
                frames[1][:, :, :3] = depth[..., None]

                self.rgb_dataset_cache[self.curr_frame_idx] = frames[0]
                self.depth_dataset_cache[self.curr_frame_idx] = frames[1]
                self.camera_extrinsics_dataset_cache[self.curr_frame_idx] = self.sim.renderer.V
                self.sim.step()

                self.frame_count += 1
                self.curr_frame_idx += 1
                self.sim.step()

                if self.curr_frame_idx == FRAME_BATCH_SIZE:
                    self.write_to_file()
                prev_point = next_point

        self.camera_intrinsics_dataset[:] = self.sim.renderer.get_intrinsics()

    def disconnect_simulator(self):
        self.write_to_file()
        self.sim.disconnect()


path = os.path.join(os.getcwd(), "data.hdf5")
h5py_file = h5py.File(path, "w")

dataset_generator = GenerateDataset(h5py_file)
dataset_generator.generate()
dataset_generator.disconnect_simulator()
