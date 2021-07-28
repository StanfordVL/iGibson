from igibson.robots.fetch_robot import Fetch
from igibson.simulator import Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.utils.utils import parse_config
from igibson.render.profiler import Profiler

import pybullet as p
from igibson.external.pybullet_tools.utils import set_joint_positions, joints_from_names, get_joint_positions, \
    get_max_limits, get_min_limits, get_sample_fn

import numpy as np
import igibson

def main():
    config = parse_config(os.path.join(igibson.example_config_path, 'fetch_reaching.yaml'))
    s = Simulator(mode='gui', physics_timestep=1 / 240.0)
    scene = EmptyScene()
    s.import_scene(scene)
    fetch = Fetch(config)
    s.import_robot(fetch)

    robot_id = fetch.robot_ids[0]

    arm_joints = joints_from_names(robot_id, [
        'torso_lift_joint',
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'upperarm_roll_joint',
        'elbow_flex_joint',
        'forearm_roll_joint',
        'wrist_flex_joint',
        'wrist_roll_joint'])

    fetch.robot_body.reset_position([0, 0, 0])
    fetch.robot_body.reset_orientation([0, 0, 1, 0])
    x, y, z = fetch.get_end_effector_position()

    visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.02)
    marker = p.createMultiBody(baseVisualShapeIndex=visual_marker)

    max_limits = [0, 0] + get_max_limits(robot_id, arm_joints)
    min_limits = [0, 0] + get_min_limits(robot_id, arm_joints)
    rest_position = [0, 0] + list(get_joint_positions(robot_id, arm_joints))
    joint_range = list(np.array(max_limits) - np.array(min_limits))
    joint_range = [item + 1 for item in joint_range]
    jd = [0.1 for item in joint_range]

    def accurateCalculateInverseKinematics(robotid, endEffectorId, targetPos, threshold, maxIter):
        sample_fn = get_sample_fn(robotid, arm_joints)
        set_joint_positions(robotid, arm_joints, sample_fn())
        it = 0
        while it < maxIter:
            jointPoses = p.calculateInverseKinematics(
                robotid,
                endEffectorId,
                targetPos,
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=jd)
            set_joint_positions(robotid, arm_joints, jointPoses[2:10])
            ls = p.getLinkState(robotid, endEffectorId)
            newPos = ls[4]

            dist = np.linalg.norm(np.array(targetPos) - np.array(newPos))
            if dist < threshold:
                break

            it += 1

        print("Num iter: " + str(it) + ", residual: " + str(dist))
        return jointPoses

    while True:
        with Profiler("Simulation step"):
            fetch.robot_body.reset_position([0, 0, 0])
            fetch.robot_body.reset_orientation([0, 0, 1, 0])
            threshold = 0.01
            maxIter = 100
            joint_pos = accurateCalculateInverseKinematics(
                robot_id,
                fetch.parts['gripper_link'].body_part_index,
                [x, y, z],
                threshold,
                maxIter)[2:10]

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
            p.resetBasePositionAndOrientation(marker, [x, y, z], [0, 0, 0, 1])

    s.disconnect()


if __name__ == '__main__':
    main()
