import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
from PIL import Image

import numpy as np
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1024
MAX_NUM_FRAMES = 10000
FRAME_BATCH_SIZE = 5

frame_size = (1024, 720)
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
output_video = cv2.VideoWriter(
    "output_video_Rs_int.mp4", fourcc, 20.0, frame_size)


class GenerateDataset(object):
    def __init__(self):
        self.sim = Simulator(
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            mode='headless',
        )

        scene = InteractiveIndoorScene(
            scene_id="Rs_int", not_load_object_categories=["door"])
        self.sim.import_scene(scene)
        self.floor = self.sim.scene.get_random_floor()
        self.check_points = []

        for room_instance in self.sim.scene.room_ins_name_to_ins_id:
            lower, upper = self.sim.scene.get_aabb_by_room_instance(
                room_instance)  # Axis Aligned Bounding Box
            x_cord, y_cord, _ = (upper - lower)/2 + lower
            self.check_points.append((x_cord, y_cord))

    def get_interpolated_steps(self, step1, step2):
        curr_x, curr_y = step1
        next_x, next_y = step2
        dist_to_next_step = distance.euclidean(
            (curr_x, curr_y), (next_x, next_y))
        path_length = int(100 * dist_to_next_step)
        interpolated_points = []
        if path_length == 0:
            return []
        x_delta = (next_x - curr_x) / path_length
        y_delta = (next_y - curr_y) / path_length

        for i in range(path_length):
            interpolated_points.append(
                [curr_x + x_delta*i, curr_y + y_delta*i])
        return np.array(interpolated_points)

    def render_image(self, step, next_step):
        x, y, z = step[0], step[1], 1
        tar_x, tar_y, tar_z = next_step[0], next_step[1], 1
        self.sim.renderer.set_camera(
            [x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
        frames = self.sim.renderer.render(modes=("rgb", "3d"))

        # Render 3d points as depth map
        depth = np.linalg.norm(frames[1][:, :, :3], axis=2)
        depth /= depth.max()
        frames[1][:, :, :3] = depth[..., None]

        self.sim.step()
        img = np.array(Image.fromarray((255 * frames[0]).astype(np.uint8)))
        cv2.imshow("test", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        output_video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def generate(self):
        # source, target, camera_up
        check_points = self.check_points
        shortest_path = []

        # Here we find the shortest path twice as an alternative to using the spline function
        for i in range(1, len(check_points)):
            steps = self.sim.scene.get_shortest_path(
                self.floor, check_points[i-1][:2], check_points[i][:2], True)[0]
            for j in range(1, len(steps) - 1):
                interpolated_steps = self.sim.scene.get_shortest_path(
                    self.floor, steps[j-1][:2], steps[j][:2], True)[0]
                for step in interpolated_steps:
                    shortest_path.append(step)
        shortest_path = np.array(shortest_path)

        steps = []
        for i in range(len(shortest_path)-1):
            for step in self.get_interpolated_steps(shortest_path[i], shortest_path[i+1]):
                steps.append(step)
        steps = np.array(steps)

        for i in range(1, len(steps)-20):
            curr_step = steps[i]
            next_step = np.average(
                steps[i+1:i+10], axis=0, weights=np.arange(18, 0, -2))
            self.render_image(curr_step, next_step)

    def disconnect_simulator(self):
        self.sim.disconnect()


path = os.path.join(os.getcwd(), "data.hdf5")
dataset_generator = GenerateDataset()
dataset_generator.generate()
dataset_generator.disconnect_simulator()
output_video.release()
