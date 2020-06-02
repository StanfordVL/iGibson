from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
import pybullet as p
import numpy as np
import os
import gibson2
import time

# Simple rendering test for VR without VR sytem (just rendering with GLFW)
optimize = True

s = Simulator(mode='vr', image_width=700, image_height=700, msaa=False, optimize_render=optimize, vrFullscreen=True, vrMode=True)
scene = BuildingScene('Bolton', is_interactive=True)
scene.sleep = optimize
s.import_scene(scene)
camera_pose = np.array([0, 0, 0])
view_direction = np.array([1, 1, 0])
s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])

if optimize:
    s.renderer.optimize_vertex_and_texture()

frame_time_sum = 0
n = 1000
for i in range(n):
    start = time.time()
    s.step()
    elapsed = time.time() - start
    frame_time_sum += elapsed

av_fps = 1/(float(frame_time_sum)/float(n))
print("Average fps:")
print(av_fps)

s.disconnect()