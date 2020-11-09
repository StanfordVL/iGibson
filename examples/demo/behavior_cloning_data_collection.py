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

# human interaction demo


def reset_fn(ycb_object):
    ycb_object_pos = np.random.uniform([-0.475, 2.9, 0.7], [-0.425, 3.0, 0.7])
    # ycb_object_pos = [-0.58, 2.9, 0.7]
    ycb_object.set_position_orientation(ycb_object_pos, [0, 0, 0, 1])


def test_import_igsdf():
    hdr_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
    hdr_texture2 = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
    light_modulation_map_filename = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
    background_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

    scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=False, object_randomization=False)
    # scene._set_first_n_objects(10)
    settings = MeshRendererSettings(env_texture_filename=hdr_texture,
                                    env_texture_filename2=hdr_texture2,
                                    env_texture_filename3=background_texture,
                                    light_modulation_map_filename=light_modulation_map_filename,
                                    enable_shadow=True, msaa=True,
                                    light_dimming_factor=1.0)
    s = Simulator(mode='iggui', image_width=960,
                  image_height=720, device_idx=0, rendering_settings=settings)
    s.import_ig_scene(scene)

    # ycb_object = YCBObject('025_mug')
    ycb_object = ArticulatedObject(
        '/cvgl2/u/chengshu/gibsonv2/gibson2/assets/models/mugs/1eaf8db2dd2b710c7d5b1b70ae595e60/1eaf8db2dd2b710c7d5b1b70ae595e60.urdf', scale=0.15)
    s.import_object(ycb_object, use_pbr=False, use_pbr_mapping=False)

    s.viewer.min_cam_z = 1.0
    s.viewer.px = -0.8
    s.viewer.py = 2.7
    s.viewer.pz = 1.08
    s.viewer.view_direction = np.array([0, 1, -0.5])
    s.viewer.reset_fn = lambda: reset_fn(ycb_object=ycb_object)

    for i in range(int(1e6)):
        # start = time.time()
        s.step()
        # end = time.time()
        demo_data = {
            'ee_pos': list(s.viewer.constraint_marker.get_position()),
            'obj_pos': list(ycb_object.get_position())
        }
        s.viewer.add_demo_data(demo_data)
        # embed()
        # print("Elapsed time: ", end - start)
        # print("Frequency: ", 1 / (end - start))
    s.disconnect()


def main():
    test_import_igsdf()


if __name__ == "__main__":
    main()
