import os
import threading

import h5py
import imageio as iio
import numpy as np
from scipy.interpolate import splev, splprep
from tqdm import tqdm

import igibson
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


class GenerateWayPoints(object):
    def __init__(self, scene_name, num_trajectories, height=720, width=1024):
        self.sim = Simulator(
            image_height=height,
            image_width=width,
            mode="headless",
        )
        scene = InteractiveIndoorScene(
            scene_id=scene_name,
            not_load_object_categories=["door"],
            trav_map_type="no_door",
            trav_map_erosion=5,
            trav_map_resolution=0.1,
        )
        self.scene_name = scene_name
        self.height = height
        self.width = width
        self.sim.import_scene(scene)
        # TODO: Are there scenes with multiple floors?
        self.floor = self.sim.scene.get_random_floor()
        self.h5py_file = None
        self.batch_size = 120
        self.curr_frame_idx = 0
        self.frame_count = 0
        self.prev_frame_count = 0

        check_points = []
        for room_instance in self.sim.scene.room_ins_name_to_ins_id:
            # TODO: Sample Traversable Path Around this Point
            lower, upper = self.sim.scene.get_aabb_by_room_instance(room_instance)  # Axis Aligned Bounding Box
            x_cord, y_cord, _ = (upper - lower) / 2 + lower
            check_points.append((x_cord, y_cord))
        check_points = np.array(check_points)

        self.scene_trajectories = []
        for i in range(num_trajectories):
            self.scene_trajectories.append(check_points.copy())
            np.random.shuffle(check_points)

    def create_dataset(self, num_images_in_trajectory):
        # Reset pointers

        self.curr_frame_idx = 0
        self.frame_count = 0
        self.prev_frame_count = 0
        self.rgb_dataset = self.h5py_file.create_dataset(
            "/rgb",
            (num_images_in_trajectory, self.height, self.width, 4),
            dtype=np.float32,
            compression="lzf",
            chunks=(min(num_images_in_trajectory, self.batch_size), self.height, self.width, 4),
        )

        self.depth_dataset = self.h5py_file.create_dataset(
            "/depth",
            (num_images_in_trajectory, self.height, self.width, 4),
            dtype=np.float32,
            compression="lzf",
            chunks=(min(num_images_in_trajectory, self.batch_size), self.height, self.width, 4),
        )

        self.camera_extrinsics_dataset = self.h5py_file.create_dataset(
            "/camera_extrinsics",
            (num_images_in_trajectory, 4, 4),
            dtype=np.float32,
            compression="lzf",
            chunks=(min(num_images_in_trajectory, self.batch_size), 4, 4),
        )

        self.create_caches()

    def create_caches(self):
        self.rgb_dataset_cache = np.zeros((self.batch_size, self.height, self.width, 4), dtype=np.float32)
        self.depth_dataset_cache = np.zeros((self.batch_size, self.height, self.width, 4), dtype=np.float32)
        self.camera_extrinsics_dataset_cache = np.zeros((self.batch_size, 4, 4), dtype=np.float32)

    def write_to_file(self):
        new_lines = self.frame_count - self.prev_frame_count
        self.prev_frame_count = self.frame_count

        if new_lines <= 0:
            return

        start_pos = self.frame_count - new_lines
        self.rgb_dataset[start_pos : self.frame_count] = self.rgb_dataset_cache[:new_lines]
        self.depth_dataset[start_pos : self.frame_count] = self.depth_dataset_cache[:new_lines]
        self.camera_extrinsics_dataset[start_pos : self.frame_count] = self.camera_extrinsics_dataset_cache[:new_lines]
        self.curr_frame_idx = 0

    def get_splined_steps(self, trajectory):
        spline_parameter, _ = splprep([trajectory[:, 0], trajectory[:, 1]], s=0.2)
        time_parameter = np.linspace(0, 1, num=len(trajectory) * 80)
        smoothed_points = np.array(splev(time_parameter, spline_parameter))[:2]
        smoothed_points = np.dstack((smoothed_points[0], smoothed_points[1]))[0]
        return smoothed_points

    def get_rgbd_frames(self, step, next_step):
        camera_height = 0.8
        x, y, z = step[0], step[1], camera_height
        tar_x, tar_y, tar_z = next_step[0], next_step[1], camera_height

        self.sim.renderer.set_camera([x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
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
        if self.curr_frame_idx == self.batch_size:
            self.write_to_file()
        return frames

    def save_trajectory_data_locally(self, uuid, splined_steps):
        number_of_splined_steps = splined_steps.shape[0]
        frame_size = (self.height, self.width)
        frame_rate = 20.0
        data_path = "data/{}/{}".format(self.scene_name, uuid)

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # RGB Video
        rgb_video_filename = os.path.join(data_path, "rgb.mp4")
        rgb_video_writer = iio.get_writer(rgb_video_filename, format="FFMPEG", mode="I", fps=frame_rate)

        # Depth Video
        depth_video_filename = os.path.join(data_path, "depth.mp4")
        depth_video_writer = iio.get_writer(depth_video_filename, format="FFMPEG", mode="I", fps=frame_rate)

        # Image Frames
        image_frames_path = os.path.join(data_path, "data.hdf5")
        self.h5py_file = h5py.File(image_frames_path, "w")
        self.create_dataset(number_of_splined_steps - 1)

        for i in tqdm(range(1, number_of_splined_steps)):
            frames = self.get_rgbd_frames(splined_steps[i - 1], splined_steps[i])
            rgb_frame = np.round(255 * frames[0]).astype(np.uint8)
            depth_frame = np.round(255 * frames[1]).astype(np.uint8)

            rgb_video_writer.append_data(rgb_frame)
            depth_video_writer.append_data(depth_frame)

        rgb_video_writer.close()
        depth_video_writer.close()

        self.h5py_file.attrs["camera_intrinsics"] = self.sim.renderer.get_intrinsics()

    def generate(self):
        scene_frames = []
        for uuid, trajectory in enumerate(self.scene_trajectories):
            first_iteration = True
            trajectory_waypoints = None
            for i in range(1, trajectory.shape[0]):
                current_position = trajectory[i - 1][:2]
                next_position = trajectory[i][:2]
                shortest_path_steps = np.array(
                    self.sim.scene.get_shortest_path(self.floor, current_position, next_position, True)[0]
                )

                if not first_iteration:
                    trajectory_waypoints = np.append(trajectory_waypoints, shortest_path_steps[1:], axis=0)
                else:
                    trajectory_waypoints = shortest_path_steps
                first_iteration = False
            splined_steps = self.get_splined_steps(trajectory_waypoints)

            self.save_trajectory_data_locally(uuid, splined_steps)
            self.write_to_file()

        self.sim.disconnect()
        return scene_frames


class GenerateDataset(object):
    def __init__(self, num_trajectories):
        self.num_trajectories = num_trajectories

    def generate_waypoints(self):
        ig_dataset_path = igibson.ig_dataset_path
        ig_scenes_path = os.path.join(ig_dataset_path, "scenes")

        scene_list = os.listdir(ig_scenes_path)
        scene_list.remove("background")

        for scene in scene_list:
            waypoint_generator = GenerateWayPoints(scene, self.num_trajectories)
            waypoint_generator.generate()


trajectories_per_scene = 100
dataset_generator = GenerateDataset(trajectories_per_scene)
dataset_generator.generate_waypoints()
