#!/usr/bin/env python
import os
import sys
import time
import random
import igibson
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.utils.assets_utils import get_ig_scene_path,get_cubicasa_scene_path,get_3dfront_scene_path
# human interaction demo


def test_import_igsdf(scene_name, scene_source):
    hdr_texture = os.path.join(
        igibson.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
    hdr_texture2 = os.path.join(
        igibson.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')

    if scene_source == "IG":
        scene_dir = get_ig_scene_path(scene_name)
    elif scene_source == "CUBICASA":
        scene_dir = get_cubicasa_scene_path(scene_name)
    else:
        scene_dir = get_3dfront_scene_path(scene_name)

    light_modulation_map_filename = os.path.join(
        scene_dir, 'layout', 'floor_lighttype_0.png')
    background_texture = os.path.join(
        igibson.ig_dataset_path, 'scenes', 'background',
        'urban_street_01.jpg')

    scene = InteractiveIndoorScene(
                    scene_name, 
                    texture_randomization=False, 
                    object_randomization=False,
                    scene_source=scene_source)

    settings = MeshRendererSettings(env_texture_filename=hdr_texture,
                                    env_texture_filename2=hdr_texture2,
                                    env_texture_filename3=background_texture,
                                    light_modulation_map_filename=light_modulation_map_filename,
                                    enable_shadow=True, msaa=True,
                                    light_dimming_factor=1.0)
    s = Simulator(mode='iggui', image_width=960,
                  image_height=720, device_idx=0, rendering_settings=settings)

    s.import_ig_scene(scene)
    fpss = []

    np.random.seed(0)
    _,(px,py,pz) = scene.get_random_point()
    s.viewer.px = px
    s.viewer.py = py
    s.viewer.pz = 1.7
    s.viewer.update()
    
    for i in range(3000):
        if i == 2500:
            logId = p.startStateLogging(loggingType=p.STATE_LOGGING_PROFILE_TIMINGS, fileName='trace_beechwood')
        start = time.time()
        s.step()
        end = time.time()
        print("Elapsed time: ", end - start)
        print("Frequency: ", 1 / (end - start))
        fpss.append(1 / (end - start))
    p.stopStateLogging(logId)
    s.disconnect()
    print("end")
    
    plt.plot(fpss)
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Open a scene with iGibson interactive viewer.')
    parser.add_argument('--scene', dest='scene_name', 
                        type=str, default='Rs_int',
                        help='The name of the scene to load')
    parser.add_argument('--source', dest='scene_source',
                        type=str, default='IG',
                        help='The name of the source dataset, among [IG,CUBICASA,THREEDFRONT]')
    args = parser.parse_args()
    test_import_igsdf(args.scene_name, args.scene_source)


if __name__ == "__main__":
    main()
