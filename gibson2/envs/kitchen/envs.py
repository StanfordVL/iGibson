import numpy as np
import os
import time

import pybullet as p
import pybullet_data
import gibson2

from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, Object
import gibson2.external.pybullet_tools.transformations as T
from gibson2.envs.kitchen.transform_utils import quat2col

from gibson2.envs.kitchen.camera import Camera
from gibson2.envs.kitchen.robots import Arm, ConstraintActuatedRobot, PlannerRobot, Robot, Gripper
from gibson2.envs.kitchen.env_utils import ObjectBank, set_friction, set_articulated_object_dynamics, pose_to_array, \
    change_object_rgba, action_to_delta_pose_axis_vector, action_to_delta_pose_euler, objects_center_in_container
import gibson2.external.pybullet_tools.utils as PBU
import gibson2.envs.kitchen.plan_utils as PU
import gibson2.envs.kitchen.skills as skills
from gibson2.envs.kitchen.objects import Faucet, Platform


def env_factory(name, **kwargs):
    if name.endswith("Skill"):
        name = name[:-5]
        kwargs["use_skills"] = True
        kwargs["use_planner"] = True
        return EnvSkillWrapper(eval(name)(**kwargs))
    else:
        return eval(name)(**kwargs)


class BaseEnv(object):
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
            obs_image=False,
            obs_depth=False,
            obs_segmentation=False,
            camera_width=256,
            camera_height=256,
            use_skills=False,
    ):
        self._hide_planner = hide_planner
        self._robot_base_pose = robot_base_pose
        self._num_sim_per_step = num_sim_per_step
        self._sim_time_step = sim_time_step
        self._use_gui = use_gui
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._obs_image = obs_image
        self._obs_depth = obs_depth
        self._obs_segmentation = obs_segmentation

        self.objects = ObjectBank()
        self.interactive_objects = ObjectBank()
        self.fixtures = ObjectBank()
        self.object_visuals = []
        self.planner = None
        self.skill_lib = None
        self._task_spec = np.array([0])

        self._setup_simulation()
        self._create_robot()
        self._create_env()
        self._create_sensors()
        self._create_env_extras()
        if use_planner:
            self._create_planner()
        if use_skills:
            assert use_planner
            self._create_skill_lib()

        self.initial_world = PBU.WorldSaver()
        assert isinstance(self.robot, Robot)

    @property
    def action_dimension(self):
        """Action dimension"""
        return 7  # [x, y, z, ai, aj, ak, g]

    @property
    def task_spec(self):
        return self._task_spec.copy()

    def _create_skill_lib(self):
        return None

    def _setup_simulation(self):
        if self._use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self._sim_time_step)

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
        planner.setup(self.robot, hide_planner=self._hide_planner)
        self.planner = planner

    def _create_env(self):
        self._create_fixtures()
        self._create_objects()

    def _create_fixtures(self):
        raise NotImplementedError

    def _create_objects(self):
        raise NotImplementedError

    def _reset_objects(self):
        raise NotImplementedError

    def _sample_task(self):
        pass

    def _create_env_extras(self):
        pass
        # for _ in range(10):
        #     self.object_visuals.append(self.objects.create_virtual_copy(scale=1., rgba_alpha=0.3))

    def _create_sensors(self):
        PBU.set_camera(45, -45, 2, (0, 0, 0))
        self.camera = Camera(
            height=self._camera_width,
            width=self._camera_height,
            fov=60,
            near=0.01,
            far=10.,
            renderer=p.ER_TINY_RENDERER
        )
        self.camera.set_pose_ypr((0, 0, 0.5), distance=2.0, yaw=45, pitch=-45)

    def reset(self):
        self.initial_world.restore()
        self.robot.reset_base_position_orientation(*self._robot_base_pose)
        self.robot.reset()
        self._reset_objects()
        self._sample_task()
        if self.skill_lib is not None:
            self.skill_lib.reset()
        for o in self.interactive_objects.object_list:
            o.reset()
        return self.get_observation()

    def reset_to(self, serialized_world_state, return_obs=True):
        exclude = []
        if self.planner is not None:
            exclude.append(self.planner.body_id)
        state = PBU.WorldSaver(exclude_body_ids=exclude)
        state.deserialize(serialized_world_state)
        state.restore()
        if return_obs:
            return self.get_observation()

    @property
    def serialized_world_state(self):
        exclude = []
        if self.planner is not None:
            exclude.append(self.planner.body_id)
        return PBU.WorldSaver(exclude_body_ids=exclude).serialize()

    def step(self, action, sleep_per_sim_step=0.0, return_obs=True):
        assert len(action) == self.action_dimension
        action = action.copy()
        gri = action[-1]
        pos, orn = action_to_delta_pose_euler(action[:6], max_dpos=self.MAX_DPOS, max_drot=self.MAX_DROT)
        # pos, orn = action_to_delta_pose_axis_vector(action[:6], max_dpos=self.MAX_DPOS, max_drot=self.MAX_DROT)
        # print(np.linalg.norm(pos), np.linalg.norm(T.euler_from_quaternion(orn)))
        self.robot.set_relative_eef_position_orientation(pos, orn)
        if gri > 0:
            self.robot.gripper.grasp()
        else:
            self.robot.gripper.ungrasp()

        for o in self.interactive_objects.object_list:
            o.step(self.objects.object_list)

        for _ in range(self._num_sim_per_step):
            p.stepSimulation()
            time.sleep(sleep_per_sim_step)

        if not return_obs:
            return self.get_reward(), self.is_done(), {}

        return self.get_observation(), self.get_reward(), self.is_done(), {}

    def get_reward(self):
        return float(self.is_success())

    def render(self, mode):
        """Render"""
        rgb, depth, obj_map, link_map = self.camera.capture_frame()
        return rgb

    def _get_pixel_observation(self, camera):
        obs = {}
        rgb, depth, seg_obj, seg_link = camera.capture_frame()
        if self._obs_image:
            obs["images"] = rgb
        if self._obs_depth:
            obs["depth"] = depth
        if self._obs_segmentation:
            obs["segmentation_objects"] = seg_obj
            obs["segmentation_links"] = seg_link
        return obs

    def _get_state_observation(self):
        # get object info
        gpose = self.robot.get_eef_position_orientation()
        object_states = self.objects.serialize()
        rel_link_poses = np.zeros_like(object_states["link_poses"])
        for i, lpose in enumerate(object_states["link_poses"]):
            rel_link_poses[i] = pose_to_array(PBU.multiply((lpose[:3], lpose[3:]), PBU.invert(gpose)))
        return {
            "link_poses": object_states["link_poses"],
            "link_relative_poses": rel_link_poses,
            "link_positions": object_states["link_poses"][:, :3],
            "link_relative_positions": rel_link_poses[:, :3]
        }

    def _get_proprio_observation(self):
        proprio = []
        proprio.append(np.array(self.robot.gripper.get_joint_positions()))
        gpos, gorn = self.robot.get_eef_position_orientation()
        gorn = quat2col(gorn)
        gpose = np.concatenate([gpos, gorn])
        proprio.append(gpose)

        gvel = np.concatenate(self.robot.get_eef_velocity(), axis=0)
        proprio.append(gvel)
        # proprio.append(pose_to_array(self.robot.get_eef_position_orientation()))

        proprio = np.hstack(proprio).astype(np.float32)
        return {
            "proprio": proprio
        }

    def get_observation(self):
        obs = {}
        obs.update(self._get_proprio_observation())
        obs.update(self._get_state_observation())
        if self._obs_image or self._obs_depth or self._obs_segmentation:
            obs.update(self._get_pixel_observation(self.camera))
        return obs

    @property
    def obstacles(self):
        return self.objects.body_ids + self.fixtures.body_ids

    def set_goal(self, **kwargs):
        """Set env target with external specification"""
        pass

    def is_done(self):
        """Check if the agent is done (not necessarily successful)."""
        return False

    def is_success(self):
        return False

    @property
    def name(self):
        """Environment name"""
        return self.__class__.__name__


class EnvSkillWrapper(object):
    def __init__(self, env):
        self.env = env
        self.skill_lib = env.skill_lib

    @property
    def action_dimension(self):
        return self.skill_lib.action_dimension + len(self.env.objects)

    def step(self, actions, sleep_per_sim_step=0.0):
        skill_params = actions[:self.skill_lib.action_dimension]
        object_index = int(np.argmax(actions[self.skill_lib.action_dimension:]))
        object_id = self.env.objects.body_ids[object_index]
        path = self.skill_lib.plan(params=skill_params, target_object_id=object_id)
        PU.execute_planned_path(self.env, path, sleep_per_sim_step=sleep_per_sim_step)
        return self.get_observation(), self.get_reward(), self.is_done(), {}

    def __getattr__(self, attr):
        """
        This method is a fallback option on any methods the original dataset might support.
        """

        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if not isinstance(result, np.ndarray) and result == self.env:
                    return self
                return result

            return hooked
        else:
            return orig_attr


class BasicKitchenEnv(BaseEnv):
    def __init__(self, **kwargs):
        kwargs["robot_base_pose"] = ([0.5, 0.3, 1.2], [0, 0, 1, 0])
        super(BasicKitchenEnv, self).__init__(**kwargs)

    def _create_fixtures(self):
        floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        p.loadMJCF(floor)

    def _create_objects(self):
        drawer = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf'))
        drawer.load()
        drawer.set_position([0, 0, 0.5])
        set_articulated_object_dynamics(drawer.body_id)
        self.objects.add_object("drawer", drawer)

        # cabinet = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf'))
        # cabinet.load()
        # cabinet.set_position([0, 0, 2])
        # set_articulated_object_dynamics(cabinet.body_id)
        # self.objects.add_object("cabinet", cabinet)

        can = YCBObject('005_tomato_soup_can')
        can.load()
        p.changeDynamics(can.body_id, -1, mass=1.0)
        set_friction(can.body_id)

        z = PBU.stable_z(can.body_id, self.objects["drawer"].body_id)
        can.set_position_orientation([0, 0, z], [0, 0, 0, 1])

        self.objects.add_object("can", can)


class BasicKitchenCanInDrawer(BasicKitchenEnv):
    def is_success(self):
        """Check if the task condition is reached."""
        can_position = self.objects["can"].get_position()
        drawer_aabb = PBU.get_aabb(self.objects["drawer"].body_id, 2)
        return PBU.aabb_contains_point(can_position, drawer_aabb)

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["can"].body_id, self.objects["drawer"].body_id)
        self.objects["can"].set_position_orientation([0, 0, z], [0, 0, 0, 1])


class BasicKitchenLiftCan(BasicKitchenEnv):
    def is_success(self):
        """Check if the task condition is reached."""
        can_position = self.objects["can"].get_position()
        surface_z = PBU.stable_z(self.objects["can"].body_id, self.objects["drawer"].body_id)
        return can_position[2] - surface_z > 0.1

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["can"].body_id, self.objects["drawer"].body_id)
        rand_pos = PU.sample_positions_in_box([-0.1, 0.1], [-0.1, 0.1], [z, z])
        self.objects["can"].set_position_orientation([rand_pos[0], rand_pos[1], z], [0, 0, 0, 1])


class TableTop(BaseEnv):
    def __init__(self, **kwargs):
        kwargs["robot_base_pose"] = ([0.5, 0.3, 1.2], [0, 0, 1, 0])
        super(TableTop, self).__init__(**kwargs)

    def _create_sensors(self):
        PBU.set_camera(45, -45, 0.8, (0, 0, 0.7))
        self.camera = Camera(
            height=self._camera_width,
            width=self._camera_height,
            fov=60,
            near=0.01,
            far=10.,
            renderer=p.ER_TINY_RENDERER
        )
        self.camera.set_pose_ypr((0, 0, 0.7), distance=0.8, yaw=45, pitch=-45)

    def _create_fixtures(self):
        p.loadMJCF(os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml"))
        table_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
            useFixedBase=True,
            basePosition=(0, 0, 0.0)
        )
        table = Object()
        table.loaded = True
        table.body_id = table_id
        self.fixtures.add_object("table", table)


class TableTopPour(TableTop):
    def _create_objects(self):
        bowl = YCBObject('024_bowl')
        bowl.load()
        p.changeDynamics(bowl.body_id, -1, mass=10.0)
        set_friction(bowl.body_id)
        self.objects.add_object("bowl_red", bowl)

        bowl = YCBObject('024_bowl')
        bowl.load()
        p.changeDynamics(bowl.body_id, -1, mass=1.0)
        change_object_rgba(bowl.body_id, (0, 0, 1, 1))
        set_friction(bowl.body_id)
        self.objects.add_object("bowl_blue", bowl)

        mug = YCBObject('025_mug')
        mug.load()
        p.changeDynamics(mug.body_id, -1, mass=1.0)
        set_friction(mug.body_id)
        self.objects.add_object("mug_red", mug)

        mug = YCBObject('025_mug')
        mug.load()
        p.changeDynamics(mug.body_id, -1, mass=1.0)
        change_object_rgba(mug.body_id, (0, 0, 1, 1))
        set_friction(mug.body_id)
        self.objects.add_object("mug_blue", mug)

        self.blue_beads_ids = [PBU.create_sphere(0.005, mass=0.1, color=(0, 0, 1, 0.8)) for _ in range(50)]
        self.red_beads_ids = [PBU.create_sphere(0.005, mass=0.1, color=(1, 0, 0, 0.8)) for _ in range(50)]

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["bowl_red"].body_id, self.fixtures["table"].body_id)
        self.objects["bowl_red"].set_position_orientation(
            PU.sample_positions_in_box([-0.25, -0.15], [-0.05, 0.05], [z, z]), PBU.unit_quat())
        self.objects["bowl_blue"].set_position_orientation(
            PU.sample_positions_in_box([-0.25, -0.15], [0.25, 0.35], [z, z]), PBU.unit_quat())
        # [0.15, 0.25], [-0.05, 0.05]
        z = PBU.stable_z(self.objects["mug_red"].body_id, self.fixtures["table"].body_id)
        self.objects["mug_red"].set_position_orientation(
            PU.sample_positions_in_box([0.15, 0.25], [-0.05, 0.05], [z, z]), PBU.unit_quat())
        self.objects["mug_blue"].set_position_orientation(
            PU.sample_positions_in_box([0.15, 0.25], [0.25, 0.35], [z, z]), PBU.unit_quat())

        beads_pos = self.objects["mug_red"].get_position()
        for i, bid in enumerate(self.red_beads_ids):
            p.resetBasePositionAndOrientation(bid, beads_pos + np.array([0, 0, z + 0.1 + i * 0.01]), PBU.unit_quat())
        beads_pos = self.objects["mug_blue"].get_position()
        for i, bid in enumerate(self.blue_beads_ids):
            p.resetBasePositionAndOrientation(bid, beads_pos + np.array([0, 0, z + 0.1 + i * 0.01]), PBU.unit_quat())

    def is_success(self):
        num_contained = 0
        bowl_aabb = PBU.get_aabb(self.objects["bowl_red"].body_id, -1)
        for bid in self.red_beads_ids:
            if PBU.aabb_contains_point(p.getBasePositionAndOrientation(bid)[0], bowl_aabb):
                num_contained += 1
        return float(num_contained) / len(self.red_beads_ids) > 0.5


class TableTopArrange(TableTop):
    def _create_objects(self):
        can = YCBObject('005_tomato_soup_can')
        can.load()
        p.changeDynamics(can.body_id, -1, mass=1.0)
        set_friction(can.body_id)
        self.objects.add_object("can", can)

        target_id = PBU.create_box(0.15, 0.15, 0.01, mass=100, color=(0, 1, 0, 1))
        target = Object()
        target.body_id = target_id
        target.loaded = True
        self.objects.add_object("target", target)

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["can"].body_id, self.fixtures["table"].body_id)
        self.objects["can"].set_position_orientation(
            PU.sample_positions_in_box([-0.1, 0.1], [-0.3, -0.1], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["target"].body_id, self.fixtures["table"].body_id)
        self.objects["target"].set_position_orientation(
            PU.sample_positions_in_box([-0.05, 0.05], [0.3, 0.2], [z, z]), PBU.unit_quat())

    def is_success(self):
        on_top = PBU.is_placement(self.objects["can"].body_id, self.objects["target"].body_id, below_epsilon=1e-2)
        return on_top


class TableTopArrangeHard(TableTop):
    def __init__(self, **kwargs):
        kwargs["robot_base_pose"] = ([0.0, 0.3, 1.4], T.quaternion_from_euler(0, np.pi / 2, 0))
        super(TableTop, self).__init__(**kwargs)

    def _sample_task(self):
        self._task_spec = np.array([np.random.randint(0, 3), np.random.randint(3, 5)])

    def set_goal(self, task_specs):
        """Set env target with external specification"""
        assert len(task_specs) == 2
        self._task_spec = np.array(task_specs)
        assert 0 <= self._task_spec[0] <= 2
        assert 3 <= self._task_spec[1] <= 4

    def _create_objects(self):
        o = YCBObject('005_tomato_soup_can')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.0)
        set_friction(o.body_id)
        self.objects.add_object("can", o)

        o = YCBObject('025_mug')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.0)
        set_friction(o.body_id)
        self.objects.add_object("mug", o)

        o = YCBObject('006_mustard_bottle')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.0)
        set_friction(o.body_id)
        self.objects.add_object("bottle", o)

        target_id = PBU.create_box(0.15, 0.15, 0.01, mass=100, color=(0, 1, 0, 1))
        target = Object()
        target.body_id = target_id
        target.loaded = True
        self.objects.add_object("target1", target)

        target_id = PBU.create_box(0.15, 0.15, 0.01, mass=100, color=(0, 0, 1, 1))
        target = Object()
        target.body_id = target_id
        target.loaded = True
        self.objects.add_object("target2", target)

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["can"].body_id, self.fixtures["table"].body_id)
        self.objects["can"].set_position_orientation(
            PU.sample_positions_in_box([-0.3, -0.2], [-0.2, -0.1], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["mug"].body_id, self.fixtures["table"].body_id)
        self.objects["mug"].set_position_orientation(
            PU.sample_positions_in_box([-0.1, 0.0], [-0.2, -0.1], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["bottle"].body_id, self.fixtures["table"].body_id)
        self.objects["bottle"].set_position_orientation(
            PU.sample_positions_in_box([0.1, 0.2], [-0.2, -0.1], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["target1"].body_id, self.fixtures["table"].body_id)
        self.objects["target1"].set_position_orientation(
            PU.sample_positions_in_box([0.1, 0.2], [0.3, 0.2], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["target2"].body_id, self.fixtures["table"].body_id)
        self.objects["target2"].set_position_orientation(
            PU.sample_positions_in_box([-0.2, -0.1], [0.3, 0.2], [z, z]), PBU.unit_quat())

    def is_success(self):
        src_object_id = self.objects.body_ids[int(self.task_spec[0])]
        tgt_object_id = self.objects.body_ids[int(self.task_spec[1])]
        return PBU.is_center_stable(src_object_id, tgt_object_id, above_epsilon=0.04, below_epsilon=0.02)

    def _create_skill_lib(self):
        lib_skills = (
            skills.GraspDistDiscreteOrn(lift_distance=0.1),
            skills.PlacePosDiscreteOrn(retract_distance=0.1)
        )
        self.skill_lib = skills.SkillLibrary(self.planner, obstacles=self.obstacles, skills=lib_skills)


class KitchenCoffee(TableTop):
    def __init__(self, **kwargs):
        super(KitchenCoffee, self).__init__(**kwargs)

    def _create_sensors(self):
        PBU.set_camera(45, -60, 0.8, (0, 0, 0.7))
        self.camera = Camera(
            height=self._camera_width,
            width=self._camera_height,
            fov=60,
            near=0.01,
            far=10.,
            renderer=p.ER_TINY_RENDERER
        )
        self.camera.set_pose_ypr((0, 0, 0.7), distance=0.8, yaw=45, pitch=-60)

    def _create_objects(self):
        o = YCBObject('025_mug')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.0)
        set_friction(o.body_id)
        self.objects.add_object("mug", o)

        o = Faucet(num_beads=10, dispense_freq=1, beads_color=(111 / 255, 78 / 255, 55 / 255, 1))
        o.load()
        self.objects.add_object("faucet_coffee", o)
        self.interactive_objects.add_object("faucet_coffee", o)

        o = Faucet(num_beads=10, dispense_freq=1, beads_color=(1, 1, 1, 1))
        o.load()
        self.objects.add_object("faucet_milk", o)
        self.interactive_objects.add_object("faucet_milk", o)

        o = YCBObject('024_bowl')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=10.0)
        self.objects.add_object("bowl", o)

    def _create_skill_lib(self):
        lib_skills = (
            skills.GraspDistDiscreteOrn(lift_height=0.1, lift_speed=0.01),
            skills.PlacePosDiscreteOrn(retract_distance=0.1),
            skills.PourPosAngle(pour_angle_speed=np.pi / 32)
        )
        self.skill_lib = skills.SkillLibrary(self.planner, obstacles=self.obstacles, skills=lib_skills)

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["mug"].body_id, self.fixtures["table"].body_id)
        self.objects["mug"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.3], [-0.2, -0.1], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["faucet_coffee"].body_id, self.fixtures["table"].body_id)
        pos = PU.sample_positions_in_box([0.2, 0.3], [0.1, 0.2], [z, z])
        coffee_pos = pos + np.array([0, 0.075, 0])
        milk_pos = pos + np.array([0, -0.075, 0])
        self.objects["faucet_coffee"].set_position_orientation(coffee_pos, PBU.unit_quat())
        self.objects["faucet_milk"].set_position_orientation(milk_pos, PBU.unit_quat())

        z = PBU.stable_z(self.objects["bowl"].body_id, self.fixtures["table"].body_id)
        self.objects["bowl"].set_position_orientation(
            PU.sample_positions_in_box([-0.3, -0.2], [-0.05, 0.05], [z, z]), PBU.unit_quat())

    def _sample_task(self):
        self._task_spec = np.random.randint(0, 2, size=1)

    def set_goal(self, task_specs):
        """Set env target with external specification"""
        assert len(task_specs) == 1
        self._task_spec = np.array(task_specs)
        assert 0 <= self._task_spec[0] <= 1

    def get_demo_suboptimal(self, noise=None):
        self.reset()
        buffer = PU.Buffer()
        skill_seq = []
        params = self.skill_lib.get_serialized_skill_params(
            "grasp_dist_discrete_orn", grasp_orn_name="back", grasp_distance=0.05)
        skill_seq.append((params, self.objects["mug"].body_id))

        place_delta = np.array((0, 0, 0.01))
        place_delta[0] += (np.random.rand(1) - 0.5) * 0.05

        # place_delta[1] += 0.075 + (np.random.rand(1) - 0.5) * 0.3
        # params = self.skill_lib.get_serialized_skill_params(
        #     "place_pos_discrete_orn", place_orn_name="front", place_pos=place_delta)
        # skill_seq.append((params, self.objects["faucet_milk"].body_id))

        target_faucet = np.random.choice(["faucet_milk", "faucet_coffee"])
        # place_delta[1] += (np.random.rand(1) - 0.5) * 0.15
        place_delta[1] += (np.random.rand(1) - 0.2) * 0.15 * (-1 if target_faucet == 'faucet_milk' else 1)
        params = self.skill_lib.get_serialized_skill_params(
            "place_pos_discrete_orn", place_orn_name="front", place_pos=place_delta)
        skill_seq.append((params, self.objects[target_faucet].body_id))

        params = self.skill_lib.get_serialized_skill_params(
            "grasp_dist_discrete_orn", grasp_orn_name="back", grasp_distance=0.05)
        skill_seq.append((params, self.objects["mug"].body_id))

        pour_delta = np.array([0, 0, 0.3])
        pour_delta[:2] += (np.random.rand(2) * 0.1 + 0.03) * np.random.choice([-1, 1], size=2)
        pour_angle = float(np.random.rand(1) * np.pi * 0.75) + np.pi * 0.25

        params = self.skill_lib.get_serialized_skill_params(
            "pour_pos_angle", pour_pos=np.array(pour_delta), pour_angle=pour_angle)
        skill_seq.append((params, self.objects["bowl"].body_id))

        skill_step = 0
        for skill_param, object_id in skill_seq:
            traj, skill_step = PU.execute_skill(
                self, self.skill_lib, skill_param,
                target_object_id=object_id,
                skill_step=skill_step,
                noise=noise
            )
            buffer.append(**traj)
        return buffer.aggregate()

    def get_demo_expert(self, noise=None):
        self.reset()
        buffer = PU.Buffer()
        skill_seq = []
        params = self.skill_lib.get_serialized_skill_params(
            "grasp_dist_discrete_orn", grasp_orn_name="back", grasp_distance=0.05)
        skill_seq.append((params, self.objects["mug"].body_id))

        target_faucet = "faucet_milk" if self.task_spec[0] == 1 else "faucet_coffee"
        place_delta = np.array((0, 0, 0.01))
        place_delta[:2] += (np.random.rand(2) - 0.5) * 0.1
        params = self.skill_lib.get_serialized_skill_params(
            "place_pos_discrete_orn", place_orn_name="front", place_pos=place_delta)
        skill_seq.append((params, self.objects[target_faucet].body_id))

        params = self.skill_lib.get_serialized_skill_params(
            "grasp_dist_discrete_orn", grasp_orn_name="back", grasp_distance=0.05)
        skill_seq.append((params, self.objects["mug"].body_id))

        pour_delta = np.array([0, 0, 0.3])
        pour_delta[:2] += (np.random.rand(2) - 0.5) * 0.1
        pour_angle = float(np.random.rand(1) * np.pi / 2) + np.pi / 2

        params = self.skill_lib.get_serialized_skill_params(
            "pour_pos_angle", pour_pos=np.array(pour_delta), pour_angle=pour_angle)
        skill_seq.append((params, self.objects["bowl"].body_id))

        skill_step = 0
        for skill_param, object_id in skill_seq:
            traj, skill_step = PU.execute_skill(
                self, self.skill_lib, skill_param,
                target_object_id=object_id,
                skill_step=skill_step,
                noise=noise
            )
            buffer.append(**traj)
        return buffer.aggregate()

    def is_success(self):
        beads = self.objects["faucet_milk"].beads if self.task_spec[0] == 1 else self.objects["faucet_coffee"].beads
        return len(objects_center_in_container(beads, self.objects["bowl"].body_id)) >= 3

    def is_success_subtasks(self):
        beads = self.objects["faucet_milk"].beads if self.task_spec[0] == 1 else self.objects["faucet_coffee"].beads
        num_beads_in_mug = len(objects_center_in_container(beads, self.objects["mug"].body_id))
        num_beads_in_bowl = len(objects_center_in_container(beads, self.objects["bowl"].body_id))
        return {
            "fill_mug": num_beads_in_mug >= 3,
            "fill_bowl": num_beads_in_bowl >= 3
        }
