from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
import pybullet as p
import numpy as np
import os
import gibson2
import time

# Simple rendering test for VR without VR sytem (just rendering with GLFW)
optimize = True

s = Simulator(mode='iggui', image_width=700, image_height=700, msaa=True, optimize_render=optimize, vrMsaa=True, vrMode=False)
scene = BuildingScene('Bolton', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)
camera_pose = np.array([0, 0, 1.2])
view_direction = np.array([1, 1, 0])
s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])

if optimize:
    s.renderer.optimize_vertex_and_texture()

for i in range(1000):
    start = time.time()
    s.step()
    elapsed = time.time() - start
    if elapsed > 0:
        print(1/elapsed)
    else:
        print("Frame time practically 0")

s.disconnect()