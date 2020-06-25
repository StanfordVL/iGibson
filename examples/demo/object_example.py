from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, Object
import gibson2
import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import gibson2.external.pybullet_tools.transformations as T


class FreeGripper(Object):
    def __init__(self, filename, init_pose, scale=1., joint_min=(0.00, 0.00), joint_max=(1., 1.)):
        super(FreeGripper, self).__init__()
        self.filename = filename
        self.scale = scale
        self.pose = init_pose
        self.joint_min = joint_min
        self.joint_max = joint_max
        assert len(joint_min) == len(self.motor_joints)
        assert len(joint_max) == len(self.motor_joints)

    def _load(self):
        body_id = p.loadURDF(self.filename, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]
        p.resetBasePositionAndOrientation(body_id, self.pose[0], self.pose[1])

        self.cid = p.createConstraint(
            parentBodyUniqueId=body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self.pose[0],
        )

        for i, jointIndex in enumerate(self.motor_joints):
            p.resetJointState(body_id, jointIndex, self.joint_max[i])

        for l in range(p.getNumJoints(body_id)):
            p.changeDynamics(body_id, l, lateralFriction=10)
        return body_id

    @property
    def motor_joints(self):
        return [0, 2]

    def gripper_set_torque(self, force):
        for joint_index in self.motor_joints:
            p.setJointMotorControl2(self.body_id, joint_index, p.TORQUE_CONTROL, force=force)

    def gripper_set_position(self, position):
        for i, joint_idx in enumerate(self.motor_joints):
            position = max(self.joint_min[i], position)
            position = min(self.joint_max[i], position)
            p.setJointMotorControl2(self.body_id, joint_idx, p.POSITION_CONTROL, targetPosition=position, force=5000)

    def reset_position_orientation(self, pos, orn):
        self.pose = (pos, orn)
        p.changeConstraint(self.cid, pos, orn)

    def set_position(self, pos):
        _, old_orn = p.getBasePositionAndOrientation(self.body_id)
        self.reset_position_orientation(pos, old_orn)


def main():
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    cabinet_0007 = os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')
    cabinet_0004 = os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')

    obj1 = InteractiveObj(filename=cabinet_0007)
    obj1.load()
    obj1.set_position([0,0,0.5])

    for jointIndex in range(p.getNumJoints(obj1.body_id)):
        # p.resetJointState(obj1, jointIndex, 0)
        friction = 0
        p.setJointMotorControl2(obj1.body_id, jointIndex, p.POSITION_CONTROL, force=friction)

    for l in range(p.getNumJoints(obj1.body_id)):
        p.changeDynamics(obj1.body_id, l, lateralFriction=10)

    obj2 = InteractiveObj(filename=cabinet_0004)
    obj2.load()
    obj2.set_position([0,0,2])

    obj3 = YCBObject('003_cracker_box')
    obj3.load()
    obj3.set_position_orientation([0,0,1.2], [0, 0, 0, 1])
    
    gripper = FreeGripper(
        filename=os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper.urdf'),
        init_pose=([0, 0.3, 1.2], [0, 0, 0, 1])
    )
    gripper.load()
    # gripper.set_position([0, 0.3, 1.2])

    pos = np.array([0, 0.3, 1.2])
    jpos = 0

    rot = T.quaternion_multiply(T.quaternion_about_axis(np.pi, [0, 0, 1]), np.array([0, 0, 0, 1]))
    gripper.reset_position_orientation(pos, rot)

    rot_yaw_pos = T.quaternion_about_axis(0.01, [0, 0, 1])
    rot_yaw_neg = T.quaternion_about_axis(0.01, [0, 0, 1])
    rot_pitch_pos = T.quaternion_about_axis(0.01, [1, 0, 0])
    rot_pitch_neg = T.quaternion_about_axis(-0.01, [1, 0, 0])

    prev_key = None
    for i in range(24000):  # at least 100 seconds
        prev_rot = rot.copy()
        prev_pos = pos.copy()
        prev_jpos = jpos
        keys = p.getKeyboardEvents()

        p.stepSimulation()
        if ord('c') in keys and prev_key != keys:
            jpos = 1 - jpos

        if p.B3G_LEFT_ARROW in keys:
            pos[1] -= 0.005
        if p.B3G_RIGHT_ARROW in keys:
            pos[1] += 0.005

        if p.B3G_UP_ARROW in keys:
            pos[0] -= 0.005
        if p.B3G_DOWN_ARROW in keys:
            pos[0] += 0.005

        if p.B3G_PAGE_UP in keys:
            pos[2] += 0.005
        if p.B3G_PAGE_DOWN in keys:
            pos[2] -= 0.005

        if p.B3G_HOME in keys:
            rot = T.quaternion_multiply(rot_yaw_pos, rot)
        if p.B3G_END in keys:
            rot = T.quaternion_multiply(rot_yaw_neg, rot)

        if p.B3G_INSERT in keys:
            rot = T.quaternion_multiply(rot_pitch_pos, rot)
        if p.B3G_DELETE in keys:
            rot = T.quaternion_multiply(rot_pitch_neg, rot)

        if not np.all(prev_pos == pos) or not np.all(prev_rot == rot):
            gripper.reset_position_orientation(pos, rot)
        if prev_jpos != jpos:
            gripper.gripper_set_position(jpos)
        # print(jpos)
        time.sleep(1./240.)
        prev_key = keys

    p.disconnect()


if __name__ == '__main__':
    main()
