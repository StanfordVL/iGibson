from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.task.task_base import iGTNTask
from gibson2.objects.articulated_object import ArticulatedObject
from tasknet.object import BaseObject
import os
import gibson2
import time
import random
import sys
import numpy as np
import pybullet as p


scene_name = 'Beechwood_0_int'
def test_import_igsdf():
    hdr_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
    hdr_texture2 = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
    light_modulation_map_filename = os.path.join(
        gibson2.ig_dataset_path, 'scenes', scene_name, 'layout', 'floor_lighttype_0.png')
    background_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

    scene = InteractiveIndoorScene(
        scene_name, texture_randomization=False, object_randomization=False)
    # scene._set_first_n_objects(5)
    settings = MeshRendererSettings(env_texture_filename=hdr_texture,
                                    env_texture_filename2=hdr_texture2,
                                    env_texture_filename3=background_texture,
                                    light_modulation_map_filename=light_modulation_map_filename,
                                    enable_shadow=True, msaa=True,
                                    light_dimming_factor=1.0)
    s = Simulator(mode='iggui', image_width=960,
                  image_height=720, device_idx=0, rendering_settings=settings)

    s.viewer.min_cam_z = 1.0

    s.import_ig_scene(scene)
    
    return s, scene


s, scene = test_import_igsdf()


sim_objects = []
dsl_objects = []

# List of object names to filename mapping
lunch_pack_folder = os.path.join(gibson2.assets_path, 'dataset', 'processed', 'pack_lunch')
lunch_pack_files = {
    'chips': os.path.join(lunch_pack_folder, 'food', 'snack', 'chips', 'chips0', 'rigid_body.urdf'),
    'fruit': os.path.join(lunch_pack_folder, 'food', 'fruit', 'pear', 'pear00', 'rigid_body.urdf'),
    'soda': os.path.join(lunch_pack_folder, 'drink', 'soda', 'soda23_mountaindew710mL', 'rigid_body.urdf'),
    'eggs': os.path.join(lunch_pack_folder, 'food', 'protein', 'eggs', 'eggs00_eggland', 'rigid_body.urdf'),
    'container': os.path.join(lunch_pack_folder, 'dish', 'casserole_dish', 'casserole_dish00', 'rigid_body.urdf')
}

item_scales = {
    'chips': 1,
    'fruit': 0.9,
    'soda': 0.8,
    'eggs': 0.5,
    'container': 0.5
}

# A list of start positions and orientations for the objects - determined by placing objects in VR
item_start_pos_orn = {
    'chips': [
        [(-5.39, -1.62, 1.42), (-0.14, -0.06, 0.71, 0.69)],
        [(-5.39, -1.62, 1.49), (-0.14, -0.06, 0.71, 0.69)],
        [(-5.12, -1.62, 1.42), (-0.14, -0.06, 0.71, 0.69)],
        [(-5.12, -1.62, 1.49), (-0.14, -0.06, 0.71, 0.69)],
    ],
    'fruit': [
        [(-4.8, -3.55, 0.97), (0, 0, 0, 1)],
        [(-4.8, -3.7, 0.97), (0, 0, 0, 1)],
        [(-4.8, -3.85, 0.97), (0, 0, 0, 1)],
        [(-4.8, -4.0, 0.97), (0, 0, 0, 1)],
    ],
    'soda': [
        [(-5.0, -3.55, 1.03), (0.68, -0.18, -0.18, 0.68)],
        [(-5.0, -3.7, 1.03), (0.68, -0.18, -0.18, 0.68)],
        [(-5.0, -3.85, 1.03), (0.68, -0.18, -0.18, 0.68)],
        [(-5.0, -4.0, 1.03), (0.68, -0.18, -0.18, 0.68)],
    ],
    'eggs': [
        [(-4.65, -1.58, 1.40), (0.72, 0, 0, 0.71)],
        [(-4.66, -1.58, 1.46), (0.72, 0, 0, 0.71)],
        [(-4.89, -1.58, 1.40), (0.72, 0, 0, 0.71)],
        [(-4.89, -1.58, 1.46), (0.72, 0, 0, 0.71)],
    ],
    'container': [
        [(-4.1, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.5, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.9, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-5.3, -1.82, 0.87), (0.71, 0, 0, 0.71)],
    ]
}

# Import all objects and put them in the correct positions
pack_items = list(lunch_pack_files.keys())
for item in pack_items:
    fpath = lunch_pack_files[item]
    start_pos_orn = item_start_pos_orn[item]
    item_scale = item_scales[item]
    for pos, orn in start_pos_orn:
        item_ob = ArticulatedObject(fpath, scale=item_scale)
        s.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)
        sim_objects.append(item_ob)
        dsl_objects.append(BaseObject(item))
        if item == 'container':
            p.changeDynamics(item_ob.body_id, -1, mass=8., lateralFriction=0.9)


igtn_task = iGTNTask('pack_lunch_demo', task_instance=2)
igtn_task.initialize_simulator(handmade_simulator=s,
                           handmade_sim_objs=sim_objects,
                           handmade_dsl_objs=dsl_objects)
while True:
    start_time = time.time()

    igtn_task.simulator.step()
    success, sorted_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', sorted_conditions['unsatisfied'])
    else:
        # break
        pass
    print('\n')

    frame_dur = time.time() - start_time
    print('FRAME DUR:', frame_dur)
    print('Fps: {}'.format(round(1/max(frame_dur, 0.00001), 2)))


