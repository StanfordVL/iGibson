""" This is a simple object picking and placing task
that can be used to benchmark the dexterity of the VR hand.
"""
import numpy as np
import os
import pybullet as p

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.objects.vr_objects import VrAgent
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.objects.articulated_object import URDFObject
from gibson2.utils.assets_utils import get_ig_model_path
from gibson2.object_states.factory import prepare_object_states


# Set to true to use viewer manipulation instead of VR
# Set to false by default so this benchmark task can be performed in VR
VIEWER_MANIP = False
# Set to true to print out render, physics and overall frame FPS
PRINT_FPS = True
# Set to true to use gripper instead of VR hands
USE_GRIPPER = False

# HDR files for PBR rendering
hdr_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

# VR rendering settings
vr_rendering_settings = MeshRendererSettings(optimized=True,
                                            fullscreen=False,
                                            env_texture_filename=hdr_texture,
                                            env_texture_filename2=hdr_texture2,
                                            env_texture_filename3=background_texture,
                                            light_modulation_map_filename=light_modulation_map_filename,
                                            enable_shadow=True,
                                            enable_pbr=True,
                                            msaa=True,
                                            light_dimming_factor=1.0)

if VIEWER_MANIP:
    s = Simulator(mode='iggui',
                image_width=512,
                image_height=512,
                rendering_settings=vr_rendering_settings,
                )
else:
    vr_settings = VrSettings(use_vr=True)
    s = Simulator(mode='vr',
                rendering_settings=vr_rendering_settings,
                vr_settings=vr_settings)

scene = InteractiveIndoorScene('Rs_int',
                          build_graph=True,
                          pybullet_load_texture=True)
scene._set_first_n_objects(3)
s.import_ig_scene(scene)

if not VIEWER_MANIP:
    vr_agent = VrAgent(s, use_gripper=USE_GRIPPER, normal_color=False)

model_path = os.path.join(get_ig_model_path('sink', 'sink_1'), 'sink_1.urdf')

sink = URDFObject(filename=model_path,
                 category='sink',
                 name='sink_1',
                 scale=np.array([0.8,0.8,0.8]),
                 abilities={'toggleable': {}, 'water_source': {}}
                 )

s.import_object(sink)
sink.set_position([1,1,0.8])

block = YCBObject(name='036_wood_block')
s.import_object(block)
block.set_position([1, 1, 1.8])
block.abilities = ["soakable", "cleaning_tool"]
prepare_object_states(block, abilities={"soakable": {}, "cleaning_tool": {}})
# assume block can soak water

model_path = os.path.join(get_ig_model_path('table', '19898'), '19898.urdf')
desk = URDFObject(filename=model_path,
                 category='table',
                 name='19898',
                 scale=np.array([0.8, 0.8, 0.8]),
                 abilities={'dustable': {}}
                 )

s.import_object(desk)
desk.set_position([1, -2, 0.4])

for _ in range(100):
    p.stepSimulation()

# Main simulation loop
while True:

    s.step(print_time=PRINT_FPS)

    if not VIEWER_MANIP:
        vr_agent.update()

s.disconnect()