import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, JR2_Kinova, Fetch
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import EmptyScene
from gibson2.core.physics.interactive_objects import InteractiveObj, BoxShape, YCBObject
from gibson2.utils.utils import parse_config
import gibson2
import os
import pytest
import pybullet as p
import numpy as np

config = parse_config('../configs/fetch_p2p_nav.yaml')
s = Simulator(mode='gui')
scene = EmptyScene()
s.import_scene(scene)
fetch = Fetch(config)
s.import_robot(fetch)
fetch.set_position_orientation([0,1,0], [0,0,1,0])
fetch.keep_still()

cabinet_0007 = os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')
cabinet_0004 = os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')

obj = InteractiveObj(filename=cabinet_0007)
s.import_interactive_object(obj)
obj.set_position([-2,0,0.5])
obj = InteractiveObj(filename=cabinet_0007)
s.import_interactive_object(obj)
obj.set_position([-2,2,0.5])
obj = InteractiveObj(filename=cabinet_0004)
s.import_interactive_object(obj)
obj.set_position([-2.1, 1.6, 2])
obj = InteractiveObj(filename=cabinet_0004)
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
    s.step()

s.disconnect()
