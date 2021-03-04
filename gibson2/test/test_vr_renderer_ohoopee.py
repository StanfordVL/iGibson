from gibson2.core.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR
import cv2
import sys
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import VisualObject, InstanceGroup, MeshRenderer
from gibson2 import assets_path, dataset_path
import time

renderer = MeshRendererVR(MeshRenderer, msaa=True)
# Note that it is necessary to load the full path of an object!
renderer.load_object(dataset_path + '\\Ohoopee\\Ohoopee_mesh_texture.obj')
renderer.add_instance(0)

while True:
    startFrame = time.time()
    renderer.render()

    endFrame = time.time()
    deltaT = endFrame - startFrame
    fps = 1/float(deltaT)

    print("Current fps: %f" % fps)

renderer.release()