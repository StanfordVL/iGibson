from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, Object
import gibson2
import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import gibson2.external.pybullet_tools.transformations as T
import gibson2.external.pybullet_tools.utils as PBU
from contextlib import contextmanager


@contextmanager
def world_saved():
    saved_world = PBU.WorldSaver()
    yield
    saved_world.restore()


def plan_pose(robot, eef_link_name, target_pose, obstacles=(), num_attempts=10):
    eef_link = PBU.link_from_name(robot, eef_link_name)
    conf = p.calculateInverseKinematics(robot, eef_link, target_pose[0], target_pose[1])

    # plan collision-free path
    with world_saved():
        path = PBU.plan_joint_motion(robot, PBU.get_movable_joints(robot), conf, obstacles=obstacles)
    return path


class FreeGripper(Object):
    def __init__(
            self,
            filename,
            init_pose,
            scale=1.,
            eef_link_name="eef_link",
            joint_min=(0.00, 0.00),
            joint_max=(1., 1.)
    ):
        super(FreeGripper, self).__init__()
        self.filename = filename
        self.scale = scale
        self.init_pose = init_pose
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.eef_link_name = eef_link_name
        self.eef_link_index = None
        assert len(joint_min) == len(self.motor_joints)
        assert len(joint_max) == len(self.motor_joints)
        self.grasp_cid = None

    def _load(self):
        body_id = p.loadURDF(self.filename, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]
        p.resetBasePositionAndOrientation(body_id, self.init_pose[0], self.init_pose[1])

        # get end-effector link index
        self.eef_link_index = PBU.link_from_name(body_id, self.eef_link_name)
        # create constraint for actuation
        self.cid = p.createConstraint(
            parentBodyUniqueId=body_id,
            parentLinkIndex=self.eef_link_index,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=self.init_pose[0],
        )

        # set to open
        for i, jointIndex in enumerate(self.motor_joints):
            p.resetJointState(body_id, jointIndex, self.joint_max[i])

        # change friction to simulate rubber
        for l in range(p.getNumJoints(body_id)):
            p.changeDynamics(body_id, l, lateralFriction=10)

        return body_id

    @property
    def motor_joints(self):
        return [0, 2]

    def grasp(self, force=100.):
        self.gripper_set_positions(self.joint_min, force=force)
        # self._magic_grasp(target_id, target_link=-1, joint_type=p.JOINT_FIXED)

    def ungrasp(self):
        self.gripper_set_positions(self.joint_max)
        # self._magic_ungrasp()

    def _magic_grasp(self, target_id, target_link=-1, joint_type=p.JOINT_FIXED):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(target_id)
        gripper_pos, gripper_orn = p.getBasePositionAndOrientation(self.body_id)
        grasp_pose = p.multiplyTransforms(*p.invertTransform(gripper_pos, gripper_orn), obj_pos, obj_orn)

        self.grasp_cid = p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=target_id,
            childLinkIndex=target_link,
            jointType=joint_type,
            jointAxis=(0, 0, 0),
            parentFramePosition=grasp_pose[0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=grasp_pose[1],
            childFrameOrientation=(0, 0, 0, 1),
        )

    def _magic_ungrasp(self):
        if self.grasp_cid is not None:
            p.removeConstraint(self.grasp_cid)
            self.grasp_cid = None

    def gripper_set_forces(self, forces):
        assert len(forces) == len(self.motor_joints)
        for i, joint_index in enumerate(self.motor_joints):
            p.setJointMotorControl2(self.body_id, joint_index, p.TORQUE_CONTROL, force=forces[i])

    def gripper_set_positions(self, positions, force=100.):
        assert len(positions) == len(self.motor_joints)
        for i, joint_idx in enumerate(self.motor_joints):
            pos = positions[i]
            pos = max(self.joint_min[i], pos)
            pos = min(self.joint_max[i], pos)
            p.setJointMotorControl2(self.body_id, joint_idx, p.POSITION_CONTROL, targetPosition=pos, force=force)

    def set_eef_position_orientation(self, pos, orn):
        self.init_pose = (pos, orn)
        p.changeConstraint(self.cid, pos, orn)

    def set_eef_position(self, pos):
        _, old_orn = p.getBasePositionAndOrientation(self.body_id)
        self.set_eef_position_orientation(pos, old_orn)

    def get_eef_position_orientation(self):
        return p.getLinkState(self.body_id, self.eef_link_index)[:2]

    def get_eef_position(self):
        return self.get_eef_position_orientation()[0]

    def get_eef_orientation(self):
        return self.get_eef_position_orientation()[1]


def main():
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)
    PBU.set_camera(45, -40, 2, (0, 0, 0))

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    cabinet_0007 = os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')
    cabinet_0004 = os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')

    obj1 = InteractiveObj(filename=cabinet_0007)
    obj1.load()
    obj1.set_position([0,0,0.5])

    for jointIndex in range(p.getNumJoints(obj1.body_id)):
        friction = 0
        p.setJointMotorControl2(obj1.body_id, jointIndex, p.POSITION_CONTROL, force=friction)

    for l in range(p.getNumJoints(obj1.body_id)):
        p.changeDynamics(obj1.body_id, l, lateralFriction=10)

    obj2 = InteractiveObj(filename=cabinet_0004)
    obj2.load()
    obj2.set_position([0,0,2])

    obj3 = YCBObject('005_tomato_soup_can')
    obj3.load()
    obj3.set_position_orientation([0,0,1.2], [0, 0, 0, 1])
    for l in range(p.getNumJoints(obj3.body_id)):
        p.changeDynamics(obj3.body_id, l, lateralFriction=10)
    
    gripper = FreeGripper(
        filename=os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper.urdf'),
        init_pose=([0, 0.3, 1.2], [0, 0, 0, 1])
    )
    gripper.load()

    # path planning
    cube_gripper = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/grippers/cube_gripper/gripper.urdf'))
    cube_gripper.load()
    cube_gripper.set_position_orientation([0, 0.3, 1.2], [0, 0, 0, 1])

    # disable collision between the shadow gripper and the real gripper
    for gl in PBU.get_all_links(gripper.body_id):
        for cgl in PBU.get_all_links(cube_gripper.body_id):
            p.setCollisionFilterPair(gripper.body_id, cube_gripper.body_id, gl, cgl, 0)

    target_pose = ([0.4579213,  0.0072391,  0.71218301], T.quaternion_multiply(T.quaternion_about_axis(np.pi, axis=[0, 0, 1]), T.quaternion_about_axis(np.pi / 2, axis=[1, 0, 0])))

    path = plan_pose(cube_gripper.body_id, "eef_link", target_pose=target_pose, obstacles=[obj3.body_id, obj1.body_id, obj2.body_id])
    for conf in path:
        for i, l in enumerate(PBU.get_movable_joints(cube_gripper.body_id)):
            p.resetJointState(cube_gripper.body_id, l, conf[i])
        target_eef_pose = p.getLinkState(cube_gripper.body_id, linkIndex=PBU.link_from_name(cube_gripper.body_id, "eef_link"))[0:2]
        print(target_eef_pose)
        for i in range(2):
            gripper.set_eef_position_orientation(target_eef_pose[0], target_eef_pose[1])

            p.stepSimulation()
            time.sleep(1./240.)

    for i in range(10):
        gripper.set_eef_position([0.3879213,  0.0072391,  0.71218301])
        p.stepSimulation()
        time.sleep(1. / 240.)

    for i in range(100):
        gripper.grasp()
        p.stepSimulation()
        time.sleep(1. / 240.)

    pos = np.array(gripper.get_eef_position())
    for i in range(100):
        pos[0] += 0.002
        gripper.set_eef_position(pos)
        p.stepSimulation()
        time.sleep(1. / 240.)


    pos = np.array(gripper.get_eef_position())
    rot = np.array(gripper.get_eef_orientation())
    jpos = 0
    grasped = False

    # rot = T.quaternion_about_axis(np.pi, [0, 0, 1])
    # gripper.set_position_orientation(pos, rot)

    rot_yaw_pos = T.quaternion_about_axis(0.01, [0, 0, 1])
    rot_yaw_neg = T.quaternion_about_axis(-0.01, [0, 0, 1])
    rot_pitch_pos = T.quaternion_about_axis(0.01, [1, 0, 0])
    rot_pitch_neg = T.quaternion_about_axis(-0.01, [1, 0, 0])

    prev_key = None
    init_t = time.time()
    for i in range(24000):  # at least 100 seconds
        prev_rot = rot.copy()
        prev_pos = pos.copy()
        prev_jpos = jpos
        keys = p.getKeyboardEvents()
        # print((time.time() - init_t) / float(i + 1))
        print(pos, rot)

        p.stepSimulation()
        if ord('c') in keys and prev_key != keys:
            if grasped:
                gripper.ungrasp()
            else:
                gripper.grasp(obj3.body_id)
            grasped = not grasped

        if p.B3G_ALT in keys and p.B3G_LEFT_ARROW in keys:
            rot = T.quaternion_multiply(rot_yaw_pos, rot)
        if p.B3G_ALT in keys and p.B3G_RIGHT_ARROW in keys:
            rot = T.quaternion_multiply(rot_yaw_neg, rot)

        if p.B3G_ALT in keys and p.B3G_UP_ARROW in keys:
            rot = T.quaternion_multiply(rot_pitch_pos, rot)
        if p.B3G_ALT in keys and p.B3G_DOWN_ARROW in keys:
            rot = T.quaternion_multiply(rot_pitch_neg, rot)

        if p.B3G_ALT not in keys and p.B3G_LEFT_ARROW in keys:
            pos[1] -= 0.005
        if p.B3G_ALT not in keys and p.B3G_RIGHT_ARROW in keys:
            pos[1] += 0.005

        if p.B3G_ALT not in keys and p.B3G_UP_ARROW in keys:
            pos[0] -= 0.005
        if p.B3G_ALT not in keys and p.B3G_DOWN_ARROW in keys:
            pos[0] += 0.005

        if p.B3G_PAGE_UP in keys:
            pos[2] += 0.005
        if p.B3G_PAGE_DOWN in keys:
            pos[2] -= 0.005

        if not np.all(prev_pos == pos) or not np.all(prev_rot == rot):
            gripper.set_eef_position_orientation(pos, rot)
        if prev_jpos != jpos:
            gripper.gripper_set_positions(jpos)

        # p.saveBullet("/Users/danfeixu/workspace/igibson/states/{}.bullet".format(i))
        # print(jpos)
        time.sleep(1./240.)
        prev_key = keys

    p.disconnect()


if __name__ == '__main__':
    main()
