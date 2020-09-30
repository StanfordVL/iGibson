from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
import pybullet as p
import numpy as np
import os
import gibson2
import time

# Simple rendering test for VR without VR sytem (just rendering with GLFW)
optimize = True

s = Simulator(mode='vr', timestep = 1/90.0, optimize_render=optimize)
scene = BuildingScene('Bolton', is_interactive=True)
scene.sleep = optimize
s.import_scene(scene)

if optimize:
    s.optimize_data()

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