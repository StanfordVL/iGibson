import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
from gibson2.core.physics.interactive_objects import YCBObject, SoftObject
import pytest
import pybullet as p
import numpy as np
import time
import random

config = parse_config('../configs/turtlebot_p2p_nav.yaml')

s = Simulator(mode='gui', image_width=512, image_height=512, render_to_tensor=False)
scene = BuildingScene('Ohoopee')
s.import_scene(scene)
turtlebot = Turtlebot(config)
s.import_robot(turtlebot)

# soft bunny
# obj = SoftObject(filename = '/scr/yxjin/assets/models/bunny/bunny.obj', basePosition = [0.5, 0.1, 1.0], mass = 0.1, useNeoHookean = 1, NeoHookeanMu = 20, NeoHookeanLambda = 20, NeoHookeanDamping = 0.001, useSelfCollision = 1, frictionCoeff = .5, collisionMargin = 0.045)
# s.import_object(obj)

# cloth
obj = SoftObject(filename = '/scr/yxjin/assets/models/cloth/cloth_z_up_highres.obj', basePosition = [0.6, 0, 0.7], mass = 0.1, useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact = 1, collisionMargin = 0.045)
s.import_object(obj)
obj.addAnchor(nodeIndex=0)
obj.addAnchor(nodeIndex=2)

##### BENCHMART CODE START #####
# multiple objects
# for i in range(4):
#     # random quaternion (not UNIFORM!)
#     w = random.uniform(0, 1)
#     qsum = w
#     x = random.uniform(0, 1) * (1 - qsum)
#     qsum += x
#     y = random.uniform(0, 1) * (1 - qsum)
#     qsum += y
#     z = random.uniform(0, 1) * (1 - qsum)
#     bunny = SoftObject(filename = '/scr/yxjin/assets/models/bunny/bunny.obj', basePosition = [0.2, 0.2, 1.8+i*0.3], baseOrientation= [x, y, z, w], mass = 0.1, useNeoHookean = 1, NeoHookeanMu = 20, NeoHookeanLambda = 20, NeoHookeanDamping = 0.001, useSelfCollision = 1, frictionCoeff = .5)
#     s.import_object(bunny)

# for i in range(1):
#     cloth = SoftObject(filename = '/scr/yxjin/assets/models/cloth/cloth_z_up_highres.obj', basePosition = [0.2, 0.2, 0.7+0.3*i], mass = 0.1, useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact = 1)
#     s.import_object(cloth)
##### BENCHMART CODE END #####

# print('start timing ...')
# start_time = time.time()
for i in range(2000):
    turtlebot.apply_action([0.5,0.5])

    ##### BENCHMART CODE START #####
    # if i % 80 == 0 and i < 600:
    #     # random quaternion (not UNIFORM!)
    #     w = random.uniform(0, 1)
    #     qsum = w
    #     x = random.uniform(0, 1) * (1 - qsum)
    #     qsum += x
    #     y = random.uniform(0, 1) * (1 - qsum)
    #     qsum += y
    #     z = random.uniform(0, 1) * (1 - qsum)

    #     if i == 240 or i == 480:
    #         bunny = SoftObject(filename = '/scr/yxjin/assets/models/bunny/bunny.obj', basePosition = [0.2, 0.2, 1.5], baseOrientation= [x, y, z, w], mass = 0.1, useNeoHookean = 1, NeoHookeanMu = 20, NeoHookeanLambda = 20, NeoHookeanDamping = 0.001, useSelfCollision = 1, frictionCoeff = .5)
    #         s.import_object(bunny)
    #     elif i == 0 or i == 320:
    #         obj = YCBObject('003_cracker_box')
    #         s.import_object(obj)
    #     else:
    #         cloth = SoftObject(filename = '/scr/yxjin/assets/models/cloth/cloth_z_up_highres.obj', basePosition = [0.2, 0.2, 1.5], baseOrientation= [x, y, z, w], mass = 0.1, useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact = 1)
    #         s.import_object(cloth)
    ##### BENCHMART CODE END #####

    s.step()
    rgb = s.renderer.render_robot_cameras(modes=('rgb'))
# used_time = time.time() - start_time
# print("duration: %s seconds, fps: %s" % (used_time, 2000/used_time))

s.disconnect()