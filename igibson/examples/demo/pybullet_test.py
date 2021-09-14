import os
import cv2
import glob
import numpy as np
import igibson
from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets
from IPython import embed
import pybullet as p
from termcolor import colored
import traceback
import pdb
import matplotlib.pyplot as plt
import pybullet_data


plt.ion()

def tup_to_np(tup, shape):
    return np.array(tup).reshape(shape)

if __name__ == '__main__':
    CID = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

 #    urdf_path = os.path.join(igibson.ig_dataset_path, "objects", "straight_chair", "219c603c479be977d5e0096fb2d3266a", "219c603c479be977d5e0096fb2d3266a.urdf")
 #    p.loadURDF(urdf_path)
 #    # obj_path = os.path.join(igibson.ig_dataset_path, "objects", "straight_chair", "219c603c479be977d5e0096fb2d3266a", "shape", "visual", "219c603c479be977d5e0096fb2d3266a_m1_vm.encrypted.obj")
 #    # visualShapeId = p.createVisualShape(
 # #    shapeType=p.GEOM_MESH,
 # #    fileName='random_urdfs/000/000.obj',
 # #    rgbaColor=None,
 # #    meshScale=[0.1, 0.1, 0.1])

 #    # collisionShapeId = p.createCollisionShape(
 #    # shapeType=p.GEOM_MESH,
 #    # fileName='random_urdfs/000/000_coll.obj',
 #    # meshScale=[0.1, 0.1, 0.1])

 #    viewMatrix = p.computeViewMatrix(
 #    cameraEyePosition=[0, 0, 3],
 #    cameraTargetPosition=[0, 0, 0],
 #    cameraUpVector=[0, 1, 0])
 #    projectionMatrix = p.computeProjectionMatrixFOV(
 #    fov=45.0,
 #    aspect=1.0,
 #    nearVal=0.1,
 #    farVal=3.1)

    # # width, height, rgbImg, depthImg, segImg
    # ret = p.getCameraImage(
    # width=224, 
    # height=224,
    # viewMatrix=viewMatrix,
    # projectionMatrix=projectionMatrix)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("r2d2.urdf")

    p.setGravity(0, 0, 0)
    camTargetPos = [0, 0, 0]
    cameraUp = [0, 0, 1]
    cameraPos = [1, 1, 1]

    pitch = -10.0

    roll = 0
    upAxisIndex = 2
    camDistance = 4
    pixelWidth = 320
    pixelHeight = 200
    nearPlane = 0.01
    farPlane = 100

    fov = 60

    yaw = 0

    p.stepSimulation()
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                            roll, upAxisIndex)
    aspect = pixelWidth / pixelHeight
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
    img_arr = p.getCameraImage(pixelWidth,
                                      pixelHeight,
                                      viewMatrix,
                                      projectionMatrix,
                                      shadow=1,
                                      lightDirection=[1, 1, 1],
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)



    w = img_arr[0]  #width of the image, in pixels
    h = img_arr[1]  #height of the image, in pixels
    rgb = img_arr[2]  #color data RGB
    dep = img_arr[3]  #depth data
    #print(rgb)
    print('width = %d height = %d' % (w, h))

    #note that sending the data using imshow to matplotlib is really slow, so we use set_data

    #plt.imshow(rgb,interpolation='none')

    #reshape is needed
    np_img_arr = np.reshape(rgb, (h, w, 4))
    # np_img_arr = np_img_arr * (1. / 255.)

    # img = [[1, 2, 3] * 50] * 100  #np.random.rand(200, 320)
    # image = plt.imshow(img, interpolation='none', animated=True, label="blah")
    # image.set_data(np_img_arr)

    # ax = plt.gca()
    # ax.plot([0])
    # #plt.draw()
    # #plt.show()
    # plt.pause(0.01)

    # frame = tup_to_np(ret[2], ret[:2] + (4,))

    filename = os.path.join(f"/home/frieda/iGibson/screenshots/pybullet_test.png")
    # step(simulator)
    # embed()
    frame = np_img_arr
    cv2.imwrite(filename, cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGBA2BGRA)) # (frame * 255).astype(np.uint8))
    while True:
        print('step')
        p.stepSimulation(1)