import csv
import os

import cv2
import numpy as np
from PIL import Image

from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1024

frame_size = (1024, 720)
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
output_video = cv2.VideoWriter("output_video.mp4", fourcc, 20.0, frame_size)

sim = Simulator(image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, mode="headless")
scene_id = "Ihlen_1_int"
scene = InteractiveIndoorScene(scene_id="Ihlen_1_int", not_load_object_categories=["door"])
sim.import_scene(scene)
render = sim.renderer

path = "/data/ig_data/ig_dataset/scenes/Ihlen_1_int/misc/tour_cam_trajectory.txt"
with open(path) as tour_trajectory:
    csv_reader = csv.reader(tour_trajectory, delimiter=",")
    for view in csv_reader:
        view = np.asarray(view, dtype=float)
        step = view[:2]
        next_step = view[2:]

        x, y, z = step[0], step[1], 0.8
        tar_x, tar_y, tar_z = next_step[0], next_step[1], 0.8

        sim.renderer.set_camera([x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])
        sim.renderer.set_fov(90)
        frames = sim.renderer.render(modes=("rgb", "3d"))

        # Render 3d points as depth map
        depth = np.linalg.norm(frames[1][:, :, :3], axis=2)
        depth /= depth.max()
        frames[1][:, :, :3] = depth[..., None]

        sim.step()
        img = np.array(Image.fromarray((255 * frames[0]).astype(np.uint8)))
        output_video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imshow("test", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    output_video.release()
