import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, JR2_Kinova, Fetch
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import EmptyScene
from gibson2.core.physics.interactive_objects import InteractiveObj, BoxShape, YCBObject, VisualMarker
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
import cv2
import pickle


config = parse_config('../configs/jr_p2p_nav.yaml')
print(config)
s = Simulator(mode='headless', timestep=1 / 240.0, image_width=800, image_height=500)
scene = EmptyScene()
s.import_scene(scene, load_sem_map=False)
fetch = Fetch(config)
s.import_robot(fetch)
fetch.robot_body.reset_position([0,1,0])
fetch.robot_body.reset_orientation([0,0,1,0])

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

interactive_obj_ids = [interactive_obj.body_id for interactive_obj in interactive_objs]
debug_line_id = None
i = 0
num_data = 0
data = []
while num_data < 100:
    if i % 300 == 0:
        for interactive_obj in interactive_objs:
            body_id = interactive_obj.body_id
            for joint_id in range(p.getNumJoints(body_id)):
                jointIndex, jointName, jointType, _, _, _, _, _, \
                jointLowerLimit, jointUpperLimit, _,_,_,_,_,_,_ = p.getJointInfo(body_id, joint_id)
                if jointType == p.JOINT_REVOLUTE or jointType == p.JOINT_PRISMATIC:
                    joint_pos = np.random.uniform(jointLowerLimit, jointUpperLimit)
                    p.resetJointState(body_id, jointIndex, targetValue=joint_pos, targetVelocity=0)
        #fetch.robot_body.reset_position([0,1,0])
        #fetch.robot_body.reset_orientation([0,0,1,0])

    fetch.apply_action([0,0,0,0,0,0,0,0,0,0])
    s.step()
    frames = s.renderer.render_robot_cameras(modes=('3d'))
    #frame = cv2.cvtColor(np.concatenate(frames[:2], axis=1), cv2.COLOR_RGB2BGR)
    #cv2.imshow('ControlView', frame)
    #q = cv2.waitKey(1)
    #print(_mouse_ix, _mouse_iy, down )
    if i % 10 == 0:
        action = (np.random.random(size=(2,)) * 500).astype(np.int)
    position_cam = frames[0][action[0], action[1]]    
    position_world = np.linalg.inv(s.renderer.V).dot(position_cam)
    #marker.set_position(position_world[:3])
    position_eye = fetch.eyes.get_position()
    res = p.rayTest(position_eye, position_world[:3])
    
    if i % 10 == 4:
        frames = s.renderer.render_robot_cameras(modes=('scene_flow'))

    if i % 10 == 5:
        # sample frame from this frame:
        frames = s.renderer.render_robot_cameras(modes=('rgb', '3d', 'scene_flow'))
        data.append((action, [frames[0][:,:,:3], frames[1][:,:,:3], frames[2][:,:,:3]]))
        num_data += 1



    if debug_line_id is not None:
        debug_line_id = p.addUserDebugLine(position_eye, position_world[:3], lineWidth=3, replaceItemUniqueId=debug_line_id)
    else:
        debug_line_id = p.addUserDebugLine(position_eye, position_world[:3], lineWidth=3)
    object_id, link_id, _, hit_pos, hit_normal = res[0]
    if object_id in interactive_obj_ids:
        p.applyExternalForce(object_id, link_id, -np.array(hit_normal)*1000, hit_pos, p.WORLD_FRAME)
    i += 1

with open('generated_data/test100.pkl', 'wb') as f:
    pickle.dump(data, f)