from gibson2.core.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR
import cv2
import sys
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import VisualObject, InstanceGroup, MeshRenderer
import time

renderer = MeshRendererVR(MeshRenderer)
# Note that it is necessary to load the full path of an object!
renderer.load_object("C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\gibson2\\assets\\datasets\\Ohoopee\\Ohoopee_mesh_texture.obj")
renderer.add_instance(0)

camera_pose = np.array([0, 0, 1.2])
view_direction = np.array([1, 0, 0])
renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
renderer.set_fov(90)

# Sets camera pose in Gibson world space
# TODO: Change to human height?
vr_camera_pose = np.array([0, 0, 1.2])
# TODO: Comment this out to enable free movement!
#renderer.set_vr_camera(vr_camera_pose)

while True:
    startFrame = time.time()
    renderer.render()

    endFrame = time.time()
    deltaT = endFrame - startFrame
    fps = 1/float(deltaT)

    #print("Current fps: %f" % fps)

renderer.release()