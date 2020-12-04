""" VR playground containing various objects. This playground operates in
the Placida scene, which is from the set of old iGibson environments and does not use PBR.

Important: VR functionality and where to find it:

1) Most VR functions can be found in the gibson2/simulator.py
2) VR utility functions are found in gibson2/utils/vr_utils.py
3) The VR renderer and VR settings can be found in gibson2/render/mesh_renderer.py
4) The underlying VR C++ code can be found in vr_mesh_render.h and .cpp in gibson2/render/cpp
"""

import numpy as np
import os
import pybullet as p

from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand, VrGazeMarker
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path

sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

s = Simulator(mode='vr', 
            rendering_settings=MeshRendererSettings(optimized=True, fullscreen=False, enable_pbr=False), 
            vr_settings=VrSettings())
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

# VR objects automatically import themselves into simulator and perform setup
# use_constraints allows the body and hands to be controlled by PyBullet's constraint system
# This is turned off during full-state data replay
vr_body = VrBody(s, use_constraints=True)
r_hand = VrHand(s, hand='right', use_constraints=True)
l_hand = VrHand(s, hand='left', use_constraints=True)
gaze_marker = VrGazeMarker(s)

# Import objects to interact with
basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path)
s.import_object(basket)
basket.set_position([1, 0.2, 1])
p.changeDynamics(basket.body_id, -1, mass=5)

mass_list = [5, 10, 100, 500]
mustard_start = [1, -0.2, 1]
mustard_list = []
for i in range(len(mass_list)):
    mustard = YCBObject('006_mustard_bottle')
    mustard_list.append(mustard)
    s.import_object(mustard)
    mustard.set_position([mustard_start[0], mustard_start[1] - i * 0.2, mustard_start[2]])
    p.changeDynamics(mustard.body_id, -1, mass=mass_list[i])

s.optimize_vertex_and_texture()

# Since vr_height_offset is set, we will use the VR HMD true height plus this offset instead of the third entry of the start pos
s.set_vr_start_pos([0.5, 0, 0], vr_height_offset=0)

# Main simulation loop
while True:
    s.step()

    # Update VR objects
    gaze_marker.update_marker()
    r_hand.update_hand()
    l_hand.update_hand()
    vr_body.update_body()

s.disconnect()