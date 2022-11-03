import os
import uuid
from multiprocessing import Pool

import h5py
import imageio as iio
import numpy as np
from scipy.interpolate import splev, splprep

import igibson
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


class WayPointDataset(object):
    def __init__(self, num_images_in_trajectory, image_frames_path, height, width):
        self.curr_frame_idx = 0
        self.frame_count = 0
        self.prev_frame_count = 0
        self.batch_size = 120
        self.h5py_file = h5py.File(image_frames_path, "w")
        self.height = height
        self.width = width
        self.camera_pose_dataset = self.h5py_file.create_dataset(
            "/camera_pose",
            (num_images_in_trajectory, 6),
            dtype=np.float16,
            compression="lzf",
            chunks=(min(num_images_in_trajectory, self.batch_size), 6),
        )

        self.rgb_dataset = self.h5py_file.create_dataset(
            "/rgb",
            (num_images_in_trajectory, self.height, self.width, 4),
            dtype=np.uint8,
            compression="lzf",
            chunks=(min(num_images_in_trajectory, self.batch_size), self.height, self.width, 4),
        )

        self.distance_dataset = self.h5py_file.create_dataset(
            "/distance",
            (num_images_in_trajectory, self.height, self.width),
            dtype=np.float16,
            compression="lzf",
            chunks=(min(num_images_in_trajectory, self.batch_size), self.height, self.width),
        )

        self.seg_dataset = self.h5py_file.create_dataset(
            "/seg",
            (num_images_in_trajectory, self.height, self.width, 1),
            dtype=np.uint16,
            compression="lzf",
            chunks=(min(num_images_in_trajectory, self.batch_size), self.height, self.width, 1),
        )

        self.ins_seg_dataset = self.h5py_file.create_dataset(
            "/ins_seg",
            (num_images_in_trajectory, self.height, self.width, 1),
            dtype=np.uint16,
            compression="lzf",
            chunks=(min(num_images_in_trajectory, self.batch_size), self.height, self.width, 1),
        )

        self.camera_extrinsics_dataset = self.h5py_file.create_dataset(
            "/camera_extrinsics",
            (num_images_in_trajectory, 4, 4),
            dtype=np.float16,
            compression="lzf",
            chunks=(min(num_images_in_trajectory, self.batch_size), 4, 4),
        )

        self.camera_pose_cache = np.zeros((self.batch_size, 6), dtype=np.float16)
        self.rgb_dataset_cache = np.zeros((self.batch_size, self.height, self.width, 4), dtype=np.uint8)
        self.distance_dataset_cache = np.zeros((self.batch_size, self.height, self.width), dtype=np.float16)
        self.seg_dataset_cache = np.zeros((self.batch_size, self.height, self.width, 1), dtype=np.uint16)
        self.in_seg_dataset_cache = np.zeros((self.batch_size, self.height, self.width, 1), dtype=np.uint16)
        self.camera_extrinsics_dataset_cache = np.zeros((self.batch_size, 4, 4), dtype=np.float16)

    def write_to_file(self):
        new_lines = self.frame_count - self.prev_frame_count
        self.prev_frame_count = self.frame_count

        if new_lines <= 0:
            return

        start_pos = self.frame_count - new_lines
        self.camera_pose_dataset[start_pos : self.frame_count] = self.camera_pose_cache[:new_lines]
        self.rgb_dataset[start_pos : self.frame_count] = self.rgb_dataset_cache[:new_lines]
        self.distance_dataset[start_pos : self.frame_count] = self.distance_dataset_cache[:new_lines]
        self.camera_extrinsics_dataset[start_pos : self.frame_count] = self.camera_extrinsics_dataset_cache[:new_lines]
        self.curr_frame_idx = 0

    def set_camera_extrinsics(self, camera_intrinsics):
        self.h5py_file.attrs["camera_intrinsics"] = camera_intrinsics

    def update_cache(self, camera_pose, rgb_image, distance, seg_frame, in_seg_frame, camera_extrinsics):
        self.camera_pose_cache[self.curr_frame_idx] = camera_pose
        self.rgb_dataset_cache[self.curr_frame_idx] = rgb_image
        self.distance_dataset_cache[self.curr_frame_idx] = distance
        self.seg_dataset_cache[self.curr_frame_idx] = seg_frame
        self.in_seg_dataset_cache[self.curr_frame_idx] = in_seg_frame
        self.camera_extrinsics_dataset_cache[self.curr_frame_idx] = camera_extrinsics

        self.frame_count += 1
        self.curr_frame_idx += 1
        if self.curr_frame_idx == self.batch_size:
            self.write_to_file()


class GenerateWayPoints(object):
    def __init__(self, scene_name, num_trajectories=70, height=800, width=1008):
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
        self.floor = self.sim.scene.get_random_floor()
        self.h5py_file = None
        self.curr_frame_idx = 0
        self.frame_count = 0
        self.prev_frame_count = 0
        self.dataset = None

        check_points = []
        for room_instance in self.sim.scene.room_ins_name_to_ins_id:
            lower, upper = self.sim.scene.get_aabb_by_room_instance(room_instance)  # Axis Aligned Bounding Box
            x_cord, y_cord, _ = (upper - lower) / 2 + lower
            check_points.append((x_cord, y_cord))
        check_points = self.obstacle_free_checkpoint(np.array(check_points))

        self.scene_trajectories = []
        for i in range(num_trajectories):
            self.scene_trajectories.append(check_points.copy())
            np.random.shuffle(check_points)


def obstacle_free_checkpoint(sim, floor, checkpoints):
    traversable_checkpoints = np.copy(checkpoints)
    num_checkpoints = checkpoints.shape[0]
    floor_map = sim.scene.floor_map[0]

    for i in range(num_checkpoints):
        position_in_map = sim.scene.world_to_map(checkpoints[i])
        if sim.scene.floor_map[0][position_in_map[0], position_in_map[1]] == 0:
            shorterst_path = np.array(
                sim.scene.get_shortest_path(floor, checkpoints[i], checkpoints[(i + 1) % num_checkpoints], True)[0]
            )
            for point in shorterst_path:
                new_position_in_map = sim.scene.world_to_map([point[0], point[1]])
                if floor_map[new_position_in_map[0], new_position_in_map[1]] == 0:
                    continue
                traversable_checkpoints[i] = point
                break
    return np.array(traversable_checkpoints)


def get_splined_steps(trajectory, knot_points):
    spline_parameter, _ = splprep([trajectory[:, 0], trajectory[:, 1]], s=0.2)
    time_parameter = np.linspace(0, 1, num=len(trajectory))
    closest_period_to_knot = np.zeros((knot_points.shape[0], 2))
    smoothed_points = np.array(splev(time_parameter, spline_parameter))[:2]
    smoothed_points = np.dstack((smoothed_points[0], smoothed_points[1]))[0]

    curr_knot_index = 0
    thresh = 0.6
    prev_delta = 1 / trajectory.shape[0]

    for i in range(time_parameter.shape[0]):
        if curr_knot_index == knot_points.shape[0]:
            break
        curr_knot = knot_points[curr_knot_index][0]
        if np.linalg.norm(curr_knot - smoothed_points[i]) < thresh:
            closest_period_to_knot[curr_knot_index][0] = time_parameter[i]
            curr_knot_index += 1

    curr_knot_index = closest_period_to_knot.shape[0] - 1
    for i in range(time_parameter.shape[0] - 1, 0, -1):
        if curr_knot_index < 0:
            break
        curr_knot = knot_points[curr_knot_index][1]
        if np.linalg.norm(curr_knot - smoothed_points[i]) < thresh:
            closest_period_to_knot[curr_knot_index][1] = time_parameter[i]
            curr_knot_index -= 1

    prev_time = 0
    new_time_parameter = np.array([])
    for period in closest_period_to_knot:
        new_time_parameter = np.append(new_time_parameter, np.arange(prev_time, period[0], prev_delta))
        new_time_parameter = np.append(new_time_parameter, np.arange(period[0], period[1], prev_delta / 10))
        prev_time = period[1]
    new_time_parameter = np.append(new_time_parameter, np.arange(prev_time, 1, prev_delta))
    # print(new_time_parameter)

    # update smoothed points with new values
    smoothed_points = np.array(splev(new_time_parameter, spline_parameter))[:2]
    smoothed_points = np.dstack((smoothed_points[0], smoothed_points[1]))[0]
    return smoothed_points


def get_rgbd_frames(sim, dataset, step, next_step):
    camera_height = 0.8
    x, y, z = step[0], step[1], camera_height
    tar_x, tar_y, tar_z = next_step[0], next_step[1], camera_height

    sim.renderer.set_camera([x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
    frames = sim.renderer.render(modes=("rgb", "3d", "seg", "ins_seg"))

    # Render 3d points as depth map
    distance = np.linalg.norm(frames[1][:, :, :3], axis=2)
    camera_pose = [x, y, z, tar_x, tar_y, tar_z]
    rgb_image = np.round(255 * frames[0]).astype(np.uint8)
    seg_frame = (512 * frames[2][:, :, 0:1]).astype(np.uint16)
    in_seg_frame = (1024 * frames[3][:, :, 0:1]).astype(np.uint16)
    camera_extrinsics = sim.renderer.V

    dataset.update_cache(camera_pose, rgb_image, distance, seg_frame, in_seg_frame, camera_extrinsics)

    sim.step()
    return frames


def save_trajectory_data_locally(sim, scene_name, uuid, splined_steps, height, width):
    number_of_splined_steps = splined_steps.shape[0]
    frame_rate = 20.0
    data_path = "data/{}/{}".format(scene_name, uuid)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # RGB Video
    rgb_video_filename = os.path.join(data_path, "rgb.mp4")
    rgb_video_writer = iio.get_writer(rgb_video_filename, format="FFMPEG", mode="I", fps=frame_rate)

    # Image Pose and Channels
    image_frames_path = os.path.join(data_path, "data.hdf5")

    dataset = WayPointDataset(number_of_splined_steps - 1, image_frames_path, height, width)

    for i in range(1, number_of_splined_steps):
        frames = get_rgbd_frames(sim, dataset, splined_steps[i - 1], splined_steps[i])
        rgb_frame = np.round(255 * frames[0]).astype(np.uint8)
        rgb_video_writer.append_data(rgb_frame)

    dataset.set_camera_extrinsics(sim.renderer.get_intrinsics())
    dataset.write_to_file()
    dataset.h5py_file.close()
    rgb_video_writer.close()


def generate(scene_name, num_trajectories=1, height=800, width=1008):
    sim = Simulator(
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
    height = height
    width = width
    sim.import_scene(scene)
    floor = sim.scene.get_random_floor()

    check_points = []
    for room_instance in sim.scene.room_ins_name_to_ins_id:
        lower, upper = sim.scene.get_aabb_by_room_instance(room_instance)  # Axis Aligned Bounding Box
        x_cord, y_cord, _ = (upper - lower) / 2 + lower
        check_points.append((x_cord, y_cord))
    check_points = obstacle_free_checkpoint(sim, floor, np.array(check_points))

    scene_trajectories = []
    for i in range(num_trajectories):
        scene_trajectories.append(check_points.copy())
        np.random.shuffle(check_points)

    for _, trajectory in enumerate(scene_trajectories):
        id = uuid.uuid1()
        first_iteration = True
        trajectory_waypoints = None
        knot_points = []
        for i in range(1, trajectory.shape[0]):
            current_position = trajectory[i - 1][:2]
            next_position = trajectory[i][:2]
            shortest_path_steps = np.array(sim.scene.get_shortest_path(floor, current_position, next_position, True)[0])

            if not first_iteration:
                knot_points.append([shortest_path_steps[0], shortest_path_steps[2]])
                trajectory_waypoints = np.append(trajectory_waypoints, shortest_path_steps[1:], axis=0)
            else:
                trajectory_waypoints = shortest_path_steps
            first_iteration = False
        knot_points = np.array(knot_points)
        splined_steps = get_splined_steps(trajectory_waypoints, knot_points)

        save_trajectory_data_locally(sim, scene_name, id, splined_steps, height, width)

    sim.disconnect()


def main():
    ig_dataset_path = igibson.ig_dataset_path
    ig_scenes_path = os.path.join(ig_dataset_path, "scenes")

    scene_list = os.listdir(ig_scenes_path)
    scene_list.remove("background")
    scene_list = [scene_list[0], scene_list[1]]

    num_workers = len(scene_list)

    with Pool(num_workers) as p:
        p.map(generate, scene_list)


if __name__ == "__main__":
    main()
