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
from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, set_point, create_box, stable_z, control_joints

import time
import numpy as np
import gibson2

config = parse_config('../configs/jr_p2p_nav.yaml')
print(config)
s = Simulator(mode='gui', timestep=1 / 240.0, image_width=800, image_height=500)
scene = EmptyScene()
s.import_scene(scene)
# jr = JR2_Kinova(config)
# s.import_robot(jr)
# jr.robot_body.reset_position([0,0,0.5])
# jr.robot_body.reset_orientation([0,0,1,0])

obstacles = []
interactive_objs = []
obj = InteractiveObj(filename=gibson2.assets_path + '/models/cabinet2/cabinet_0007.urdf')
ids = s.import_articulated_object(obj, class_id=30)
obstacles.append(ids)
interactive_objs.append(obj)


obj.set_position([-2,0,0.5])
obj = InteractiveObj(filename=gibson2.assets_path + '/models/cabinet2/cabinet_0007.urdf')
ids = s.import_articulated_object(obj, class_id=60)
obstacles.append(ids)
interactive_objs.append(obj)

obj.set_position([-2,2,0.5])
obj = InteractiveObj(filename=gibson2.assets_path + '/models/cabinet/cabinet_0004.urdf')
ids = s.import_articulated_object(obj, class_id=90)
obstacles.append(ids)
interactive_objs.append(obj)


obj.set_position([-2.1, 1.6, 2])
obj = InteractiveObj(filename=gibson2.assets_path + '/models/cabinet/cabinet_0004.urdf')
ids = s.import_articulated_object(obj, class_id=120)
obstacles.append(ids)
obj.set_position([-2.1, 0.4, 2])
interactive_objs.append(obj)


obj = BoxShape([-2.05,1,0.5], [0.35,0.6,0.5])
ids = s.import_articulated_object(obj, class_id=150)
obstacles.append(ids)

obj = BoxShape([-2.45,1,1.5], [0.01,2,1.5])
ids = s.import_articulated_object(obj, class_id=180)
obstacles.append(ids)

p.createConstraint(0,-1,obj.body_id, -1, p.JOINT_FIXED, [0,0,1], [-2.55,1,1.5], [0,0,0])
obj = YCBObject('003_cracker_box')
s.import_object(obj, class_id=210)
p.resetBasePositionAndOrientation(obj.body_id, [-2,1,1.2], [0,0,0,1])
obj = YCBObject('003_cracker_box')
s.import_object(obj, class_id=240)
p.resetBasePositionAndOrientation(obj.body_id, [-2,2,1.2], [0,0,0,1])

print(interactive_objs)

#from IPython import embed; embed()
i = 0
while True:
    if i % 2000 == 0:
        for interactive_obj in interactive_objs:
            body_id = interactive_obj.body_id
            for joint_id in range(p.getNumJoints(body_id)):
                jointIndex, jointName, jointType, _, _, _, _, _, \
                jointLowerLimit, jointUpperLimit, _,_,_,_,_,_,_ = p.getJointInfo(body_id, joint_id)

                if jointType == p.JOINT_REVOLUTE or jointType == p.JOINT_PRISMATIC:
                    joint_pos = np.random.uniform(jointLowerLimit, jointUpperLimit)
                    p.resetJointState(body_id, jointIndex, targetValue=joint_pos, targetVelocity=0)
    s.step()
    i += 1