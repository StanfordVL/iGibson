from gibson2.core.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR
import cv2
import sys
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import VisualObject, InstanceGroup, MeshRenderer
from gibson2.core.render.viewer import ViewerVR

viewer = ViewerVR()
viewer.renderer = MeshRendererVR(MeshRenderer)

# Note that it is necessary to load the full path of an object!
viewer.renderer.load_object("C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\gibson2\\assets\\datasets\\Ohoopee\\Ohoopee_mesh_texture.obj")
viewer.renderer.add_instance(0)

while True:
    viewer.update(should_reset_vr_camera=True, vr_camera_pos=np.array([0,0,1.2]))

    hmdIsValid, hmdTrans, hmdRot = viewer.renderer.vrsys.getDataForVRDevice("hmd")
    lIsValid, lTrans, lRot = viewer.renderer.vrsys.getDataForVRDevice("left_controller")
    rIsValid, rTrans, rRot = viewer.renderer.vrsys.getDataForVRDevice("right_controller")

    print("Printing device data in order hmd, left_c, right_c:")
    print(hmdIsValid, hmdTrans, hmdRot)
    print(lIsValid, lTrans, lRot)
    print(rIsValid, rTrans, rRot)

viewer.renderer.release()