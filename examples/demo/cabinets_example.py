import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, JR2_Kinova, Fetch
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import EmptyScene
from gibson2.core.physics.interactive_objects import InteractiveObj, BoxShape, YCBObject
from gibson2.utils.utils import parse_config
from gibson2.core.render.profiler import Profiler

import pytest
import pybullet as p
import numpy as np

config = parse_config('../configs/jr_interactive_nav.yaml')
s = Simulator(mode='headless', timestep=1 / 240.0)
scene = EmptyScene()
s.import_scene(scene)
jr = JR2_Kinova(config)
s.import_robot(jr)
jr.robot_body.reset_position([0,0,0])
jr.robot_body.reset_orientation([0,0,1,0])
fetch = Fetch(config)
s.import_robot(fetch)
fetch.robot_body.reset_position([0,1,0])
fetch.robot_body.reset_orientation([0,0,1,0])
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0007/part_objs/cabinet_0007.urdf')
s.import_interactive_object(obj)
obj.set_position([-2,0,0.5])
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0007/part_objs/cabinet_0007.urdf')
s.import_interactive_object(obj)
obj.set_position([-2,2,0.5])
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0004/part_objs/cabinet_0004.urdf')
s.import_interactive_object(obj)
obj.set_position([-2.1, 1.6, 2])
obj = InteractiveObj(filename='/data4/mdv0/cabinet/0004/part_objs/cabinet_0004.urdf')
s.import_interactive_object(obj)
obj.set_position([-2.1, 0.4, 2])
obj = BoxShape([-2.05,1,0.5], [0.35,0.6,0.5])
s.import_interactive_object(obj)
obj = BoxShape([-2.45,1,1.5], [0.01,2,1.5])
s.import_interactive_object(obj)
p.createConstraint(0,-1,obj.body_id, -1, p.JOINT_FIXED, [0,0,1], [-2.55,1,1.5], [0,0,0])
obj = YCBObject('003_cracker_box')
s.import_object(obj)
p.resetBasePositionAndOrientation(obj.body_id, [-2,1,1.2], [0,0,0,1])
obj = YCBObject('003_cracker_box')
s.import_object(obj)
p.resetBasePositionAndOrientation(obj.body_id, [-2,2,1.2], [0,0,0,1])

for i in range(1000):
    with Profiler("Simulation: step"):
        s.step()

s.disconnect()