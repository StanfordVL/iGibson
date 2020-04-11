from gibson2.core.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR
import cv2
import sys
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import VisualObject, InstanceGroup, MeshRenderer
from gibson2 import assets_path
import time
import os

renderer = MeshRendererVR(MeshRenderer)

model_path = assets_path + '\\datasets\\Rs_interactive\\Rs_interactive'
files = os.listdir(model_path)
files = [item for item in files if item.endswith('obj')]

for i,fn in enumerate(files):
    renderer.load_object(os.path.join(model_path, fn))
    renderer.add_instance(i)

camera_pose = np.array([0, 0, 1.2])
view_direction = np.array([1, 0, 0])
renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
renderer.set_fov(90)

while True:
    startFrame = time.time()
    renderer.render()

    endFrame = time.time()
    deltaT = endFrame - startFrame
    fps = 1/float(deltaT)

    print("Current fps: %f" % fps)

renderer.release()