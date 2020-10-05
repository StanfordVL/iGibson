""" A simple VR demo where you can look around and measure FPS. You can toggle VR mode on/off
to test out the VR rendering without the HMD."""

import os
import numpy as np
import pybullet as p

from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path

optimize = True
vrMode = True
s = Simulator(mode='vr', timestep = 1/90.0, vrFullscreen=False, optimized_renderer=optimize, vrMode=vrMode)
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

if not vrMode:
    camera_pose = np.array([0, 0, 1.2])
    view_direction = np.array([1, 0, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)

model = YCBObject('006_mustard_bottle')
s.import_object(model)
model.set_position([1,0,1])

if optimize:
    s.optimize_data()

# Start user close to counter for interaction
s.setVROffset([1.0, 0, -0.4])

# Measure FPS of VR system
while True:
    s.step(shouldPrintTime=True)

s.disconnect()