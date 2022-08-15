import logging

import numpy as np
import pybullet as p
from PIL import Image

from scipy.spatial.transform import Rotation as R

from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene


def uniform_generator(scene):
    random_floor = scene.get_random_floor()
    camera_x, camera_y, camera_z = tuple(scene.get_random_point(random_floor)[1])
    camera_z = camera_z + np.random.uniform(low=1, high=2)
    camera_pos = np.array([camera_x, camera_y, camera_z])

    camera_yaw = np.random.uniform(-np.pi, np.pi)
    camera_pitch = np.random.uniform(low=-np.pi /18, high=np.pi / 4)

    r = R.from_euler("yz", [camera_pitch, camera_yaw])
    forward = r.apply([1, 0, 0])
    up = r.apply([0, 0, 1])

    camera_target = camera_pos + forward

    return camera_pos, camera_target, up

def gaussian_target_generator(scene):
    scene_objects = scene.get_objects()
    for object in scene_objects:
        for body_id in object.get_body_ids():
            position, orientation = p.getBasePositionAndOrientation(body_id)
            aabb = p.getAABB(body_id)
            print(aabb)
            #write code to
            return uniform_generator(scene)