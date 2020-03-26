import pybullet as p
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.core.physics.scene import StadiumScene, BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config

# TODO: Users need to input different paths here - how to automate?
configs_folder = 'C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\examples\\configs\\'
config = parse_config(configs_folder + 'turtlebot_p2p_nav.yaml')

s = Simulator(mode='vr')
scene = StadiumScene()
s.import_scene(scene)
turtlebot = Turtlebot(config)
s.import_robot(turtlebot)
obj = YCBObject('006_mustard_bottle')

for i in range(2):
    s.import_object(obj)

obj = YCBObject('002_master_chef_can')
for i in range(2):
    s.import_object(obj)

# Runs simulation and measures fps
while True:
    # Always call before step
    eventList = s.pollVREvents()
    for event in eventList:
        deviceType, eventType = event
        print(deviceType, eventType)
        #if deviceType == "left_controller":
         #   print(eventType)
         #   print("Event happend on left controller!")

    # Set should_measure_fps to True to measure the current fps
    s.step(should_measure_fps=False)

    # Always call after step
    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
    #print('left controller data:')
    #print(lIsValid, lTrans, lRot)

s.disconnect()






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