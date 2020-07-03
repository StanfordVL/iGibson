import numpy as np
import os
import time

import pybullet as p
import pybullet_data
import gibson2

from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject
import gibson2.external.pybullet_tools.transformations as T

from gibson2.envs.kitchen.camera import Camera
from gibson2.envs.kitchen.robots import Arm, ConstraintActuatedRobot, PlannerRobot, Robot, Gripper
from gibson2.envs.kitchen.env_utils import ObjectBank, set_friction, set_articulated_object_dynamics, pose_to_array,\
    PlannerObjectBank
import gibson2.external.pybullet_tools.utils as PBU


class BaseKitchenEnv(object):
    MAX_DPOS = 0.1
    MAX_DROT = np.pi / 8

    def __init__(
            self,
            robot_base_pose,
            num_sim_per_step,
            use_gui=False,
            use_planner=False,
            hide_planner=True,
            sim_time_step=1./240.,
    ):
        self._hide_planner = hide_planner
        self._robot_base_pose = robot_base_pose
        self._num_sim_per_step = num_sim_per_step
        self._sim_time_step = sim_time_step
        self._use_gui = use_gui
        self.objects = ObjectBank()
        self.object_visuals = []
        self.planner = None

        self._setup_simulation()
        self._create_robot()
        self._create_env()
        self._create_sensors()
        self._create_env_extras()
        if use_planner:
            self._create_planner()

        self.initial_world = PBU.WorldSaver()
        assert isinstance(self.robot, Robot)

    @property
    def action_dimension(self):
        """Action dimension"""
        return 7  # [x, y, z, ai, aj, ak, g]

    @property
    def name(self):
        """Environment name"""
        return "BasicKitchen"

    def _setup_simulation(self):
        if self._use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self._sim_time_step)
        PBU.set_camera(45, -40, 2, (0, 0, 0))

    def _create_robot(self):
        gripper = Gripper(
            joint_names=("left_gripper_joint", "right_gripper_joint"),
            finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip")
        )
        gripper.load(os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper.urdf'))
        robot = ConstraintActuatedRobot(
            eef_link_name="eef_link", init_base_pose=self._robot_base_pose, gripper=gripper)

        self.robot = robot

    def _create_planner(self):
        shadow_gripper = Gripper(
            joint_names=("left_gripper_joint", "right_gripper_joint"),
            finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip")
        )
        shadow_gripper.load(
            os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper_plannable.urdf'),
            scale=1.2  # make the planner robot slightly larger than the real gripper to allow imprecise plan
        )
        arm = Arm(joint_names=("txj", "tyj", "tzj", "rxj", "ryj", "rzj"))
        arm.load(body_id=shadow_gripper.body_id)
        planner = PlannerRobot(
            eef_link_name="eef_link",
            init_base_pose=self._robot_base_pose,
            gripper=shadow_gripper,
            arm=arm,
            plannable_joint_names=arm.joint_names,
            # plan_objects=PlannerObjectBank.create_from(
            #     self.objects, scale=1.2, rgba_alpha=0. if self._hide_planner else 0.7)
        )
        planner.setup(self.robot, self.objects.body_ids, hide_planner=self._hide_planner)
        self.planner = planner

    def _create_env(self):
        floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        p.loadMJCF(floor)

        drawer = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf'))
        drawer.load()
        drawer.set_position([0, 0, 0.5])
        set_articulated_object_dynamics(drawer.body_id)
        self.objects.add_object("drawer", drawer)

        cabinet = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf'))
        cabinet.load()
        cabinet.set_position([0, 0, 2])
        set_articulated_object_dynamics(cabinet.body_id)
        self.objects.add_object("cabinet", cabinet)

        can = YCBObject('005_tomato_soup_can')
        can.load()
        z = PBU.stable_z(can.body_id, drawer.body_id)
        can.set_position_orientation([0, 0, z], [0, 0, 0, 1])
        p.changeDynamics(can.body_id, -1, mass=1.0)
        set_friction(can.body_id)
        self.objects.add_object("can", can)

    def _create_env_extras(self):
        pass
        # for _ in range(10):
        #     self.object_visuals.append(self.objects.create_virtual_copy(scale=1., rgba_alpha=0.3))

    def _create_sensors(self):
        self.camera = Camera(
            height=256, width=256, fov=60, near=0.01, far=100., renderer=p.ER_TINY_RENDERER
        )
        self.camera.set_pose_ypr((0, 0, 0.5), distance=2.0, yaw=45, pitch=-45)

    def reset(self):
        self.initial_world.restore()
        self.robot.reset_base_position_orientation(*self._robot_base_pose)
        self.robot.reset()
        z = PBU.stable_z(self.objects["can"].body_id, self.objects["drawer"].body_id)
        self.objects["can"].set_position_orientation([0, 0, z], [0, 0, 0, 1])
        return self.get_observation()

    @property
    def sim_state(self):
        return np.zeros(3)  # TODO: implement this

    def step(self, action, sleep_per_sim_step=0.0):
        assert len(action) == self.action_dimension
        action = action.copy()
        pos = action[:3]  # delta position
        orn = action[3:6]  # delta rotation in euler angle
        gri = action[-1]  # grasp or not
        pos *= self.MAX_DPOS
        orn *= self.MAX_DROT

        orn = T.quaternion_from_euler(*orn)
        self.robot.set_relative_eef_position_orientation(pos, orn)
        if gri > 0:
            self.robot.gripper.grasp()
        else:
            self.robot.gripper.ungrasp()
        for _ in range(self._num_sim_per_step):
            p.stepSimulation()
            time.sleep(sleep_per_sim_step)

        return self.get_observation(), self.get_reward(), self.is_done(), {}

    def get_reward(self):
        return float(self.is_success())

    def render(self, mode):
        """Render"""
        rgb, depth, obj_map, link_map = self.camera.capture_frame()
        return rgb

    def get_observation(self):
        # get proprio
        proprio = []
        gpose = self.robot.get_eef_position_orientation()
        proprio.append(np.array(self.robot.gripper.get_joint_positions()))
        proprio.append(pose_to_array(gpose))
        proprio = np.hstack(proprio).astype(np.float32)

        # get object info
        object_states = self.objects.serialize()
        rel_link_poses = np.zeros_like(object_states["link_poses"])
        for i, lpose in enumerate(object_states["link_poses"]):
            rel_link_poses[i] = pose_to_array(PBU.multiply((lpose[:3], lpose[3:]), PBU.invert(gpose)))

        return {
            "proprio": proprio,
            "link_poses": object_states["link_poses"],
            "link_relative_poses": rel_link_poses,
            "link_positions": object_states["link_poses"][:, :3],
            "link_relative_positions": rel_link_poses[:, :3]
        }

    def set_goal(self, **kwargs):
        """Set env target with external specification"""
        pass

    def is_done(self):
        """Check if the agent is done (not necessarily successful)."""
        return False

    def is_success(self):
        """Check if the task condition is reached."""
        can_position = self.objects["can"].get_position()
        drawer_aabb = PBU.get_aabb(self.objects["drawer"].body_id, 2)
        return PBU.aabb_contains_point(can_position, drawer_aabb)
