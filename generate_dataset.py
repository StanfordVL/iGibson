import os
import cv2
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

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
            mode='headless',
        )

        scene = InteractiveIndoorScene(scene_id="Ihlen_1_int")
        self.sim.import_scene(scene)
        self.sim.scene.open_all_doors()
        self.floor = self.sim.scene.get_random_floor()
        self.check_points = []
        # self.create_datasets()
        # self.create_caches()

        for room_instance in self.sim.scene.room_ins_name_to_ins_id:
            lower, upper = self.sim.scene.get_aabb_by_room_instance(room_instance) #Axis Aligned Bounding Box
            x_cord, y_cord, z_cord = (upper - lower)/2 + lower
            self.check_points.append((x_cord, y_cord))

    # def create_datasets(self):
    #     self.rgb_dataset = self.h5py_file.create_dataset(
    #         "/rgb",
    #         (MAX_NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 4),
    #         dtype=np.float32,
    #         compression="lzf",
    #         chunks=(min(MAX_NUM_FRAMES, FRAME_BATCH_SIZE),
    #                 IMAGE_HEIGHT, IMAGE_WIDTH, 4),
    #     )

    #     self.depth_dataset = self.h5py_file.create_dataset(
    #         "/depth",
    #         (MAX_NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 4),
    #         dtype=np.float32,
    #         compression="lzf",
    #         chunks=(min(MAX_NUM_FRAMES, FRAME_BATCH_SIZE),
    #                 IMAGE_HEIGHT, IMAGE_WIDTH, 4),
    #     )

    #     self.camera_extrinsics_dataset = self.h5py_file.create_dataset(
    #         "/camera_extrinsics",
    #         (MAX_NUM_FRAMES, 4, 4),
    #         dtype=np.float32,
    #         compression="lzf",
    #         chunks=(min(MAX_NUM_FRAMES, FRAME_BATCH_SIZE), 4, 4),
    #     )

    #     self.camera_intrinsics_dataset = self.h5py_file.create_dataset(
    #         "/camera_intrinsics",
    #         (3, 3),
    #         dtype=np.float32,
    #     )

    # def create_caches(self):
    #     self.rgb_dataset_cache = np.zeros(
    #         (FRAME_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
    #     self.depth_dataset_cache = np.zeros(
    #         (FRAME_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
    #     self.camera_extrinsics_dataset_cache = np.zeros(
    #         (FRAME_BATCH_SIZE, 4, 4), dtype=np.float32)

    # def write_to_file(self):
    #     new_lines = self.frame_count - self.prev_frame_count
    #     self.prev_frame_count = self.frame_count

    #     if new_lines <= 0:
    #         return

    #     start_pos = self.frame_count - new_lines
    #     self.rgb_dataset[start_pos: self.frame_count] = self.rgb_dataset_cache[:new_lines]
    #     self.depth_dataset[start_pos: self.frame_count] = self.depth_dataset_cache[:new_lines]
    #     self.camera_extrinsics_dataset[start_pos:
    #                                    self.frame_count] = self.camera_extrinsics_dataset_cache[:new_lines]
    #     self.curr_frame_idx = 0

    def prepare_spline_functions(self, shortest_path):
        self.spline_functions = []
        path_length = len(shortest_path)
        self.spline_functions.append(CubicSpline(
            range(path_length), shortest_path[:, 0], bc_type='clamped'))
        self.spline_functions.append(CubicSpline(
            range(path_length), shortest_path[:, 1], bc_type='clamped'))

    def get_interpolated_steps(self, step):
        path_length = 10
        interpolated_points = []
        for i in range(path_length - 1):
            curr_step = step + (1.0/path_length*i)
            interpolated_points.append([self.spline_functions[0](
                curr_step), self.spline_functions[1](curr_step)])
        return np.array(interpolated_points)

    def generate(self):
        # source, target, camera_up
        # TODO: Generate New Waypoints
        check_points = self.check_points
        shortest_path = []
        for i in range(1, len(check_points)):
            steps = self.sim.scene.get_shortest_path(self.floor, check_points[i-1][:2], check_points[i][:2], True)[0]
            for i in range(len(steps)-1):
                step = steps[i]
                shortest_path.append(step)
        shortest_path = np.array(shortest_path)
        self.prepare_spline_functions(shortest_path)

        steps = []
        for i in range(len(shortest_path)):
            for step in self.get_interpolated_steps(i):
                steps.append(step)


        for i in range(len(steps)-1):
            step = steps[i]
            next_step = steps[i+1]
            if self.frame_count == MAX_NUM_FRAMES:
                break

            x, y, z = step[0], step[1], 0.8
            tar_x, tar_y, tar_z = next_step[0], next_step[1], 0.8

            self.sim.renderer.set_camera(
                [x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
            frames = self.sim.renderer.render(modes=("rgb", "3d"))

            # Render 3d points as depth map
            depth = np.linalg.norm(frames[1][:, :, :3], axis=2)
            depth /= depth.max()
            frames[1][:, :, :3] = depth[..., None]

            # self.rgb_dataset_cache[self.curr_frame_idx] = frames[0]
            # self.depth_dataset_cache[self.curr_frame_idx] = frames[1]
            # self.camera_extrinsics_dataset_cache[self.curr_frame_idx] = self.sim.renderer.V
            self.sim.step()
            cv2.imshow("test", cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            self.frame_count += 1
            self.curr_frame_idx += 1

            if self.curr_frame_idx == FRAME_BATCH_SIZE:
                # self.write_to_file()
                pass

        # self.camera_intrinsics_dataset[:] = self.sim.renderer.get_intrinsics()

    def disconnect_simulator(self):
        # self.write_to_file()
        self.sim.disconnect()


path = os.path.join(os.getcwd(), "data.hdf5")
# h5py_file = h5py.File(path, "w")

# dataset_generator = GenerateDataset(h5py_file)
dataset_generator = GenerateDataset([])
dataset_generator.generate()
# dataset_generator.disconnect_simulator()
