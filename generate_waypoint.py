import sys

import cv2
import dubins
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import splev, splprep

from debug_turns import plot_paths
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

np.set_printoptions(threshold=sys.maxsize)


IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1024
MAX_NUM_FRAMES = 10000
FRAME_BATCH_SIZE = 5

scene_name = "Ihlen_1_int"


class GenerateDataset(object):
    def __init__(self):
        self.sim = Simulator(
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
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

        self.check_points = self.obstacle_free_checkpoint(check_points)

        self.first_iteration = True
        self.current_camera_angle = None
        self.render = False
        self.total_trajectory = None

    def obstacle_free_checkpoint(self, checkpoints):
        traversable_checkpoints = np.copy(checkpoints)
        num_checkpoints = len(checkpoints)
        floor_map = self.sim.scene.floor_map[0]

        for i in range(num_checkpoints):
            position_in_map = self.sim.scene.world_to_map(checkpoints[i])
            if self.sim.scene.floor_map[0][position_in_map[0], position_in_map[1]] == 0:
                shorterst_path = np.array(
                    self.sim.scene.get_shortest_path(
                        self.floor, checkpoints[i], checkpoints[(i + 1) % num_checkpoints], True
                    )[0]
                )
                for point in shorterst_path:
                    new_position_in_map = self.sim.scene.world_to_map([point[0], point[1]])
                    if floor_map[new_position_in_map[0], new_position_in_map[1]] == 0:
                        continue
                    traversable_checkpoints[i] = point
                    break
        return np.array(traversable_checkpoints)

    def turn_camera_for_next_trajectory(self, current_step, next_step):
        next_steps = np.array(
            dubins.shortest_path(
                [current_step[0], current_step[1], np.pi / 2], [next_step[0], next_step[1], -np.pi / 2], 0.2
            ).sample_many(0.1)[0]
        )

        obstacle_free_path = []
        for step in next_steps:
            position_in_map = self.sim.scene.world_to_map([step[0], step[1]])
            if self.sim.scene.floor_map[0][position_in_map[0], position_in_map[1]] == 0:
                continue
            obstacle_free_path.append(step)
        return np.array(obstacle_free_path)[:, :2]

    def get_splined_steps(self):
        spline_parameter, u = splprep([self.total_trajectory[:, 0], self.total_trajectory[:, 1]], s=0.2)
        # print(u[1:] - u[:-1])
        time_parameter = np.linspace(0, 1, num=len(self.total_trajectory) * 5)
        smoothed_points = np.array(splev(time_parameter, spline_parameter))[:2]
        smoothed_points = np.dstack((smoothed_points[0], smoothed_points[1]))[0]
        return smoothed_points

    def render_image(self, step, next_step):
        x, y, z = step[0], step[1], 1
        tar_x, tar_y, tar_z = next_step[0], next_step[1], 1

        self.sim.renderer.set_camera([x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
        frames = self.sim.renderer.render(modes=("rgb"))

        self.sim.step()
        img = np.array(Image.fromarray((255 * frames[0]).astype(np.uint8)))
        cv2.imshow("test", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def generate(self):
        # source, target, camera_up
        check_points = self.check_points

        for i in range(1, len(check_points)):
            current_position = check_points[i - 1][:2]
            next_position = check_points[i][:2]

            shortest_path_steps = np.array(
                self.sim.scene.get_shortest_path(self.floor, current_position, next_position, True)[0]
            )

            if not self.first_iteration:
                self.total_trajectory = np.append(
                    self.total_trajectory,
                    self.turn_camera_for_next_trajectory(shortest_path_steps[0], shortest_path_steps[1]),
                    axis=0,
                )
                self.total_trajectory = np.append(self.total_trajectory, shortest_path_steps[2:-1, :], axis=0)
            else:
                self.total_trajectory = shortest_path_steps[:-1]

            self.first_iteration = False

        splined_steps = self.get_splined_steps()
        plot_paths(self.total_trajectory, splined_steps)
        for i in range(len(splined_steps) - 1):
            self.render_image(splined_steps[i], splined_steps[i + 1])

    def disconnect_simulator(self):
        self.sim.disconnect()


dataset_generator = GenerateDataset()
dataset_generator.generate()
dataset_generator.disconnect_simulator()
