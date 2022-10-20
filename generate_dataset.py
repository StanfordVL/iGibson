import os

import cv2
import numpy as np
from scipy.interpolate import splev, splprep

import igibson
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


class GenerateWayPoints(object):
    def __init__(self, scene_name, num_trajectories):
        self.sim = Simulator(
            image_height=720,
            image_width=1024,
            mode="headless",
        )
        scene = InteractiveIndoorScene(
            scene_id=scene_name,
            not_load_object_categories=["door"],
            trav_map_type="no_door",
            trav_map_erosion=5,
            trav_map_resolution=0.1,
        )
        self.sim.import_scene(scene)
        self.floor = self.sim.scene.get_random_floor()

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

    def get_splined_steps(self, trajectory):
        spline_parameter, _ = splprep([trajectory[:, 0], trajectory[:, 1]], s=0.2)
        time_parameter = np.linspace(0, 1, num=len(trajectory) * 1)
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

        self.sim.step()
        cv2.imshow("test", cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        return frames

    def generate(self):
        scene_frames = []
        for trajectory in self.scene_trajectories:
            first_iteration = True
            trajectory_waypoints = None
            for i in range(1, len(trajectory)):
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

            trajectory_frames = []
            for j in range(1, len(splined_steps)):
                curr_step = splined_steps[j - 1]
                next_step = splined_steps[j]
                trajectory_frames.append(self.get_rgbd_frames(curr_step, next_step))
            # TODO: Don't do this. Stack overflow. Load to dataset instead
            # scene_frames.append(trajectory_frames)

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
            images_from_trajectories = waypoint_generator.generate()
            # TODO: Store frames to dataset in a parallel way


trajectories_per_scene = 1
dataset_generator = GenerateDataset(trajectories_per_scene)
dataset_generator.generate_waypoints()
