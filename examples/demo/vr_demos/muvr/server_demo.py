""" Server-side demo for multi-user VR.

The server is responsible for running the physics simulation, sending
rendering data to the client, and retrieving VR data from the client for the physics simulation.
"""

import numpy as np
import os
import pybullet as p

from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path

# Key classes used for MUVR interaction
from igvr_server import IGVRServer, IGVRChannel

sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

# VR rendering settings
optimize = True
vr_mode = True

# Initialize simulator with specific rendering settings
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, 
            rendering_settings=MeshRendererSettings(optimized=optimize, fullscreen=False, enable_pbr=False),
            vr_eye_tracking=True, vr_mode=vr_mode)
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

# Eye tracking visual marker - a red marker appears in the scene to indicate gaze direction
gaze_marker = VisualMarker(radius=0.03)
s.import_object(gaze_marker)
gaze_marker.set_position([0,0,1.5])

# Start off with very few items in the scene
basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path)
s.import_object(basket)
basket.set_position([1, 0.2, 1])
p.changeDynamics(basket.body_id, -1, mass=5)

if optimize:
    s.optimize_vertex_and_texture()

# Start user close to counter for interaction
s.set_vr_offset([-2.0, 0.0, -0.4])

# Networking settings
# TODO: Change this to an IP address to communicate over a network
host = 'localhost'
port = 8887
vr_server = IGVRServer(localaddr=(host, port))
# Register iGibson renderer to work with server
vr_server.register_renderer(s.renderer)

# Main simulation loop
while True:
    s.step()

    # VR eye tracking data
    is_eye_data_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.get_eye_tracking_data()
    if is_eye_data_valid:
        # Move gaze marker based on eye tracking data
        updated_marker_pos = [origin[0] + dir[0], origin[1] + dir[1], origin[2] + dir[2]]
        gaze_marker.set_position(updated_marker_pos)

    # Under the hood, extracts object information from renderer and sends over to the client
    vr_server.send_frame()

s.disconnect()