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

    def generate(self):
        # TODO: Does it suffice to use random points or is it better to use specific points?
        # objects = self.sim.scene.get_objects()
        # object_nav_transitions = [[64, 49],
        #                           [49, 71],
        #                           [71, 64]]

        # for i, object in enumerate(object_nav_transitions):
        #     print(i, object.name, object.category)

        positions = [[-5, 1.8, 1.1],
        [3.4, 1.8, 1.1]]

        for i in range(len([12,12])):
            # prev_point = objects[object_nav_transitions[i][0]].get_position()[:2]
            # curr_point = objects[object_nav_transitions[i][1]].get_position()[:2]
            prev_point = positions[0][:2]
            curr_point = positions[1][:2]
            # TODO: Can we get a motion planner for this stage?
            # TODO: Can we use a game controller to get the initial video?
            steps = self.sim.scene.get_shortest_path(self.floor, prev_point, curr_point, True)[0]
            print(len(steps))
            prev_point = curr_point

            for step in steps:
                if self.frame_count == MAX_NUM_FRAMES:
                    break

                x, y, z, dir_x, dir_y = step[0], step[1], 1.2, 1, 0
                tar_x = x + dir_x
                tar_y = y + dir_y
                tar_z = 1.2
                self.sim.renderer.set_camera([x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
                self.sim.renderer.set_fov(45)
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
                cv2.imshow("image", cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
                
                for i in range(50):
                    self.sim.step()

                if self.curr_frame_idx == FRAME_BATCH_SIZE:
                    self.write_to_file()

        self.camera_intrinsics_dataset[:] = self.sim.renderer.get_intrinsics()

    def disconnect_simulator(self):
        self.write_to_file()
        self.sim.disconnect()


path = os.path.join(os.getcwd(), "data.hdf5")
h5py_file = h5py.File(path, "w")

dataset_generator = GenerateDataset(h5py_file)
dataset_generator.generate()
dataset_generator.disconnect_simulator()
