import logging

import numpy as np
import pybullet as p
from PIL import Image

from scipy.spatial.transform import Rotation as R

from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

def texture_randomization(scene):
    scene.randomize_texture()

def object_randomization(scene):
    scene.randomize_objects()
def joint_randomization(scene):
    pass