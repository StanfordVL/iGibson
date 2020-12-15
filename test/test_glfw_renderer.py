from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
import cv2
import sys
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import VisualObject, InstanceGroup, MeshRenderer
import time

renderer = MeshRenderer(width=512, height=512)
renderer.load_object("C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\gibson2\\assets\\datasets\\Ohoopee\\Ohoopee_mesh_texture.obj")
renderer.add_instance(0)

print(renderer.visual_objects, renderer.instances)
print(renderer.materials_mapping, renderer.mesh_materials)
camera_pose = np.array([0, 0, 1.2])
view_direction = np.array([1, 0, 0])
renderer.set_camera(camera_pose, camera_pose + view_direction, [1, 0, 0])
renderer.set_fov(90)

while True:
    frame = renderer.render(modes=('rgb'))
    cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
    q = cv2.waitKey(1)

renderer.release()