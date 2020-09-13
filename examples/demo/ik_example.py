from gibson2.robots.robot_locomotors import Fetch
from gibson2.simulator import Simulator
from gibson2.scenes.scene import EmptyScene
from gibson2.utils.utils import parse_config
from gibson2.render.profiler import Profiler

import pybullet as p
from gibson2.external.pybullet_tools.utils import set_joint_positions, joints_from_names, get_joint_positions, \
    get_max_limits, get_min_limits, get_sample_fn

import numpy as np


def main():
    config = parse_config('../configs/fetch_p2p_nav.yaml')
    s = Simulator(mode='gui', timestep=1 / 240.0)
    scene = EmptyScene()
    s.import_scene(scene)
    fetch = Fetch(config)
    s.import_robot(fetch)

    robot_id = fetch.robot_ids[0]

    arm_joints = joints_from_names(robot_id, ['torso_lift_joint','shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint',
                                              'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint'])
    #finger_joints = joints_from_names(robot_id, ['l_gripper_finger_joint', 'r_gripper_finger_joint'])
    fetch.robot_body.reset_position([0, 0, 0])
    fetch.robot_body.reset_orientation([0, 0, 1, 0])
    x,y,z = fetch.get_end_effector_position()
    #set_joint_positions(robot_id, finger_joints, [0.04,0.04])
    print(x,y,z)

    visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius = 0.02)
    marker = p.createMultiBody(baseVisualShapeIndex = visual_marker)

    #max_limits = [0,0] + get_max_limits(robot_id, arm_joints) + [0.05,0.05]
    #min_limits = [0,0] + get_min_limits(robot_id, arm_joints) + [0,0]
    #rest_position = [0,0] + list(get_joint_positions(robot_id, arm_joints)) + [0.04,0.04]
    max_limits = [0,0] + get_max_limits(robot_id, arm_joints)
    min_limits = [0,0] + get_min_limits(robot_id, arm_joints)
    rest_position = [0,0] + list(get_joint_positions(robot_id, arm_joints))
    joint_range = list(np.array(max_limits) - np.array(min_limits))
    joint_range = [item + 1 for item in joint_range]
    jd = [0.1 for item in joint_range]
    print(max_limits)
    print(min_limits)

    def accurateCalculateInverseKinematics(robotid, endEffectorId, targetPos, threshold, maxIter):
        sample_fn = get_sample_fn(robotid, arm_joints)
        set_joint_positions(robotid, arm_joints, sample_fn())
        it = 0
        while it < maxIter:
            jointPoses = p.calculateInverseKinematics(robotid, endEffectorId, targetPos,
                                                      lowerLimits = min_limits,
                                                      upperLimits = max_limits,
                                                      jointRanges = joint_range,
                                                      restPoses = rest_position,
                                                      jointDamping = jd)
            set_joint_positions(robotid, arm_joints, jointPoses[2:10])
            ls = p.getLinkState(robotid, endEffectorId)
            newPos = ls[4]

            dist = np.linalg.norm(np.array(targetPos) - np.array(newPos))
            if dist < threshold:
                break

            it += 1

        print ("Num iter: " + str(it) + ", threshold: " + str(dist))
        return jointPoses

    while True:
        with Profiler("Simulation step"):
            fetch.robot_body.reset_position([0, 0, 0])
            fetch.robot_body.reset_orientation([0, 0, 1, 0])
            threshold = 0.01
            maxIter = 100
            joint_pos = accurateCalculateInverseKinematics(robot_id, fetch.parts['gripper_link'].body_part_index, [x, y, z],
                                                           threshold, maxIter)[2:10]

            #set_joint_positions(robot_id, finger_joints, [0.04, 0.04])
            s.step()
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN)):
                    x += 0.01
                if (k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN)):
                    x -= 0.01
                if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
                    y += 0.01
                if (k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN)):
                    y -= 0.01
                if (k == ord('z') and (v & p.KEY_IS_DOWN)):
                    z += 0.01
                if (k == ord('x') and (v & p.KEY_IS_DOWN)):
                    z -= 0.01
            p.resetBasePositionAndOrientation(marker, [x,y,z], [0,0,0,1])

    s.disconnect()


if __name__ == '__main__':
    main()

