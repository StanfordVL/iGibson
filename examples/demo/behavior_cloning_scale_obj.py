#!/usr/bin/env python
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
import os
import gibson2
import time
import random
import sys
from IPython import embed
import numpy as np
import pybullet as p


def scale_obj():
    p.connect(p.DIRECT)
    root_dir = '/cvgl2/u/chengshu/gibsonv2/gibson2/assets/models/mugs'
    scales = []
    for model in os.listdir(root_dir):
        # if model != '1eaf8db2dd2b710c7d5b1b70ae595e60':
        #     continue
        urdf = os.path.join(root_dir, model, model + '.urdf')
        body_id = p.loadURDF(urdf)
        lower, upper = p.getAABB(body_id)
        extent = np.array(upper) - np.array(lower)
        min_xy = np.min(extent[:2])
        scale = 0.08 / min_xy
        scales.append(scale)
        p.removeBody(body_id)
    print(np.mean(scales))
    p.disconnect()


def main():
    scale_obj()


if __name__ == "__main__":
    main()
