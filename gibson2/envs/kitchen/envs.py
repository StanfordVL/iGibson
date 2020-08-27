import numpy as np
import os
from collections import OrderedDict

import pybullet as p
import pybullet_data
import gibson2

from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, Object
import gibson2.external.pybullet_tools.transformations as T

from gibson2.envs.kitchen.camera import Camera
from gibson2.envs.kitchen.env_utils import ObjectBank, set_friction, set_articulated_object_dynamics, pose_to_array, \
    change_object_rgba, action_to_delta_pose_axis_vector, action_to_delta_pose_euler, objects_center_in_container
import gibson2.external.pybullet_tools.utils as PBU
import gibson2.envs.kitchen.plan_utils as PU
import gibson2.envs.kitchen.skills as skills
from gibson2.envs.kitchen.objects import Faucet, Box
from gibson2.envs.kitchen.base_env import BaseEnv, EnvSkillWrapper


def env_factory(name, **kwargs):
    if name.endswith("Skill"):
        name = name[:-5]
        kwargs["use_skills"] = True
        kwargs["use_planner"] = True
        return EnvSkillWrapper(eval(name)(**kwargs))
    else:
        return eval(name)(**kwargs)


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
    def is_success_all_tasks(self):
        """Check if the task condition is reached."""
        can_position = self.objects["can"].get_position()
        drawer_aabb = PBU.get_aabb(self.objects["drawer"].body_id, 2)
        return {"task": PBU.aabb_contains_point(can_position, drawer_aabb)}

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["can"].body_id, self.objects["drawer"].body_id)
        self.objects["can"].set_position_orientation([0, 0, z], [0, 0, 0, 1])


class BasicKitchenLiftCan(BasicKitchenEnv):
    def is_success_all_tasks(self):
        """Check if the task condition is reached."""
        can_position = self.objects["can"].get_position()
        surface_z = PBU.stable_z(self.objects["can"].body_id, self.objects["drawer"].body_id)
        return {"task": can_position[2] - surface_z > 0.1}

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

    def is_success_all_tasks(self):
        num_contained = 0
        bowl_aabb = PBU.get_aabb(self.objects["bowl_red"].body_id, -1)
        for bid in self.red_beads_ids:
            if PBU.aabb_contains_point(p.getBasePositionAndOrientation(bid)[0], bowl_aabb):
                num_contained += 1
        return {"task": float(num_contained) / len(self.red_beads_ids) > 0.5}


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

    def is_success_all_tasks(self):
        on_top = PBU.is_placement(self.objects["can"].body_id, self.objects["target"].body_id, below_epsilon=1e-2)
        return {"task": on_top}


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

    def is_success_all_tasks(self):
        src_object_id = self.objects.body_ids[int(self.task_spec[0])]
        tgt_object_id = self.objects.body_ids[int(self.task_spec[1])]
        return {"task": PBU.is_center_stable(src_object_id, tgt_object_id, above_epsilon=0.04, below_epsilon=0.02)}

    def _create_skill_lib(self):
        lib_skills = (
            skills.GraspDistDiscreteOrn(lift_distance=0.1),
            skills.PlacePosDiscreteOrn(retract_distance=0.1)
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills)


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
        # o = Faucet(num_beads=10, dispense_freq=1, beads_color=(0.9, 0.9, 0, 1))
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

        # o = InteractiveObj(filename=os.path.join(gibson2.assets_path, "models/rbo/microwave/configuration/microwave.urdf"))
        # o.load()
        # self.objects.add_object("microwave", o)

    def _create_skill_lib(self):
        lib_skills = (
            skills.GraspDistDiscreteOrn(name="grasp_dist_discrete_orn", lift_height=0.1, lift_speed=0.01),
            skills.PlacePosDiscreteOrn(name="place_pos_discrete_orn", retract_distance=0.1, num_pause_steps=30),
            skills.PourPosAngle(name="pour_pos_angle", pour_angle_speed=np.pi / 32, num_pause_steps=30)
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills, verbose=True)

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

        # p.resetBasePositionAndOrientation(self.objects["microwave"].body_id, (0, 0, 0), skills.ORIENTATIONS["top"])
        # z = PBU.stable_z(self.objects["microwave"].body_id, self.fixtures["table"].body_id)
        # self.objects["microwave"].set_position_orientation(
        #     PU.sample_positions_in_box([-0.3, -0.2], [-0.3, -0.3], [z, z]),  skills.ORIENTATIONS["top"])

    def _get_feature_observation(self):
        num_beads = np.zeros((len(self.objects), 2))
        for i, o in enumerate(self.objects.object_list):
            num_beads[i, 0] = len(
                objects_center_in_container(self.objects["faucet_coffee"].beads, container_id=o.body_id)
            )
            num_beads[i, 1] = len(
                objects_center_in_container(self.objects["faucet_milk"].beads, container_id=o.body_id)
            )
        obs = dict(
            num_beads=num_beads
        )
        return obs

    def _sample_task(self):
        self._task_spec = np.random.randint(0, 2, size=1)
        self._target_faucet = "faucet_milk" if self._task_spec[0] == 1 else "faucet_coffee"

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

        ##################### continuous
        # full
        # place_delta[1] += 0.075 + (np.random.rand(1) - 0.5) * 0.3

        # centered
        # target_faucet = np.random.choice(["faucet_milk", "faucet_coffee"])
        # place_delta[1] += 0.075 + 0.075 * (-1 if target_faucet == 'faucet_milk' else 1) + (np.random.rand(1) - 0.5) * 0.04

        # correct goal
        if self._target_faucet == "faucet_milk":
            place_delta[1] += (np.random.rand(1) - 0.5) * 0.15
        else:
            place_delta[1] += 0.15 + (np.random.rand(1) - 0.5) * 0.15

        params = self.skill_lib.get_serialized_skill_params(
            "place_pos_discrete_orn", place_orn_name="front", place_pos=place_delta)
        skill_seq.append((params, self.objects["faucet_milk"].body_id))

        ################# discrete
        # target_faucet = np.random.choice(["faucet_milk", "faucet_coffee"])
        # discrete-unbiased
        # place_delta[1] += (np.random.rand(1) - 0.5) * 0.15
        # discrete-biased
        # place_delta[1] += (np.random.rand(1) - 0.2) * 0.15 * (-1 if target_faucet == 'faucet_milk' else 1)

        # params = self.skill_lib.get_serialized_skill_params(
        #     "place_pos_discrete_orn", place_orn_name="front", place_pos=place_delta)
        # skill_seq.append((params, self.objects[target_faucet].body_id))

        params = self.skill_lib.get_serialized_skill_params(
            "grasp_dist_discrete_orn", grasp_orn_name="back", grasp_distance=0.05)
        skill_seq.append((params, self.objects["mug"].body_id))

        pour_delta = np.array([0, 0, 0.3])
        pour_delta[:2] += (np.random.rand(2) * 0.1 + 0.03) * np.random.choice([-1, 1], size=2)
        pour_angle = float(np.random.rand(1) * np.pi * 0.75) + np.pi * 0.25

        params = self.skill_lib.get_serialized_skill_params(
            "pour_pos_angle", pour_pos=np.array(pour_delta), pour_angle=pour_angle)
        skill_seq.append((params, self.objects["bowl"].body_id))

        for skill_step, (skill_param, object_id) in enumerate(skill_seq):
            traj, exec_info = PU.execute_skill(
                self, self.skill_lib, skill_param,
                target_object_id=object_id,
                skill_step=skill_step,
                noise=noise
            )
            buffer.append(**traj)
        return buffer.aggregate(), None

    def get_demo_expert(self, noise=None):
        self.reset()
        buffer = PU.Buffer()
        skill_seq = []
        params = self.skill_lib.get_serialized_skill_params(
            "grasp_dist_discrete_orn", grasp_orn_name="back", grasp_distance=0.05)
        skill_seq.append((params, self.objects["mug"].body_id))

        place_delta = np.array((0, 0, 0.01))
        place_delta[:2] += (np.random.rand(2) - 0.5) * 0.1
        params = self.skill_lib.get_serialized_skill_params(
            "place_pos_discrete_orn", place_orn_name="front", place_pos=place_delta)
        skill_seq.append((params, self.objects[self._target_faucet].body_id))

        params = self.skill_lib.get_serialized_skill_params(
            "grasp_dist_discrete_orn", grasp_orn_name="back", grasp_distance=0.05)
        skill_seq.append((params, self.objects["mug"].body_id))

        pour_delta = np.array([0, 0, 0.3])
        pour_delta[:2] += (np.random.rand(2) - 0.5) * 0.1
        pour_angle = float(np.random.rand(1) * np.pi / 2) + np.pi / 2

        params = self.skill_lib.get_serialized_skill_params(
            "pour_pos_angle", pour_pos=np.array(pour_delta), pour_angle=pour_angle)
        skill_seq.append((params, self.objects["bowl"].body_id))

        for skill_step, (skill_param, object_id) in enumerate(skill_seq):
            traj, exec_info = PU.execute_skill(
                self, self.skill_lib, skill_param,
                target_object_id=object_id,
                skill_step=skill_step,
                noise=noise
            )
            buffer.append(**traj)
        return buffer.aggregate(), None

    def is_success_all_tasks(self):
        num_beads_in_mug_milk = len(objects_center_in_container(
            self.objects["faucet_milk"].beads, self.objects["mug"].body_id))
        num_beads_in_bowl_milk = len(objects_center_in_container(
            self.objects["faucet_milk"].beads, self.objects["bowl"].body_id))
        num_beads_in_mug_coffee = len(objects_center_in_container(
            self.objects["faucet_coffee"].beads, self.objects["mug"].body_id))
        num_beads_in_bowl_coffee = len(objects_center_in_container(
            self.objects["faucet_coffee"].beads, self.objects["bowl"].body_id))

        successes = {
            "fill_mug": num_beads_in_mug_milk >= 3 if self._target_faucet == "faucet_milk" else num_beads_in_mug_coffee >= 3,
            "fill_bowl": num_beads_in_bowl_milk >= 3 if self._target_faucet == "faucet_milk" else num_beads_in_bowl_coffee >= 3,
            "fill_mug_any": num_beads_in_mug_milk >= 3 or num_beads_in_mug_coffee >= 3,
            "fill_bowl_any": num_beads_in_bowl_milk >= 3 or num_beads_in_bowl_coffee >= 3,
        }
        successes["task"] = successes["fill_bowl"]
        return successes


class KitchenCoffeeAP(KitchenCoffee):
    def set_goal(self, task_specs):
        """Set env target with external specification"""
        self._task_spec = np.array(task_specs)

    def _sample_task(self):
        skill_name = np.random.choice(["fill_bowl_milk", "fill_bowl_coffee"])
        self._task_spec = np.array([self.skill_lib.name_to_skill_index(skill_name), self.objects.names.index("bowl")])
        self._target_faucet = "faucet_milk" if skill_name == "fill_bowl_milk" else "faucet_coffee"
        self._task_skill_name = skill_name
        self._task_object_name = "bowl"

    def _create_skill_lib(self):
        def fill_bowl(objects, pl):
            num_beads_in_bowl = len(objects_center_in_container(
                objects["faucet_" + pl].beads, objects["bowl"].body_id))
            return num_beads_in_bowl >= 3

        lib_skills = (
            skills.GraspDistDiscreteOrn(
                name="grasp", lift_height=0.1, lift_speed=0.01
            ),
            skills.GraspDistDiscreteOrn(
                name="grasp_fill_mug_any", lift_height=0.1, lift_speed=0.01,
                precondition_fn=lambda: self.is_success_all_tasks()["fill_mug_any"]
            ),
            skills.PlacePosDiscreteOrn(
                name="place", retract_distance=0.1, num_pause_steps=30,
            ),
            skills.PourPosAngle(
                name="pour", pour_angle_speed=np.pi / 32, num_pause_steps=30,
                params=OrderedDict(
                    pour_pos=skills.SkillParamsContinuous(low=(-1, -1, 0), high=(1, 1, 1)),
                    pour_angle=skills.SkillParamsContinuous(low=(0,), high=(np.pi,))
                )
            ),
            skills.ConditionSkill(
                name="fill_bowl_milk", precondition_fn=lambda objs=self.objects: fill_bowl(self.objects, "milk"),
            ),
            skills.ConditionSkill(
                name="fill_bowl_coffee", precondition_fn=lambda objs=self.objects: fill_bowl(self.objects, "coffee")
            )
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills, verbose=True)

    def get_demo_suboptimal(self, noise=None):
        self.reset()
        skill_seq = []
        params = self.skill_lib.sample_serialized_skill_params(
            "grasp",
            grasp_orn=dict(choices=[3]),
            grasp_distance=dict(low=[0.05], high=[0.05])
        )
        skill_seq.append((params, self.objects["mug"].body_id))

        params = self.skill_lib.sample_serialized_skill_params(
            "place",
            place_orn=dict(choices=[0]),
            place_pos=dict(low=(-0.025, -0.075, 0.01), high=(0.025, 0.075, 0.01))
        )
        target_faucet = np.random.choice(["faucet_milk", "faucet_coffee"])
        skill_seq.append((params, self.objects[target_faucet].body_id))

        params = self.skill_lib.sample_serialized_skill_params(
            "grasp_fill_mug_any",
            grasp_orn=dict(choices=[3]),
            grasp_distance=dict(low=[0.05], high=[0.05])
        )
        skill_seq.append((params, self.objects["mug"].body_id))

        def get_pour_pos():
            pour_delta = np.array([0, 0, 0.3])
            pour_delta[:2] += (np.random.rand(2) * 0.1 + 0.03) * np.random.choice([-1, 1], size=2)
            return pour_delta

        params = self.skill_lib.sample_serialized_skill_params(
            "pour",
            pour_pos=dict(sampler_fn=get_pour_pos),
            pour_angle=dict(low=[np.pi * 0.25], high=[np.pi])
        )
        skill_seq.append((params, self.objects["bowl"].body_id))

        # check goal
        final_skill_name = "fill_bowl_milk" if self.task_spec[0] == 1 else "fill_bowl_coffee"
        params = self.skill_lib.get_serialized_skill_params(final_skill_name)
        skill_seq.append((params, self.objects["bowl"].body_id))

        buffer = PU.Buffer()
        skill_step = 0
        exception = None
        for skill_param, object_id in skill_seq:
            traj, exec_info = PU.execute_skill(
                self, self.skill_lib, skill_param,
                target_object_id=object_id,
                skill_step=skill_step,
                noise=noise
            )
            buffer.append(**traj)
            if exec_info["exception"] is not None:
                exception = exec_info["exception"]
                break
        else:
            assert self.is_success()

        return buffer.aggregate(), exception


class SimpleCoffeeAP(KitchenCoffee):
    def set_goal(self, task_specs):
        """Set env target with external specification"""
        self._task_spec = np.array(task_specs)

    def _sample_task(self):
        skill_name = np.random.choice(["grasp_fill_mug_milk", "grasp_fill_mug_coffee"])
        self._task_spec = np.array([self.skill_lib.name_to_skill_index(skill_name), self.objects.names.index("mug")])
        self._target_faucet = "faucet_milk" if skill_name == "grasp_fill_mug_milk" else "faucet_coffee"
        self._task_skill_name = skill_name
        self._task_object_name = "mug"

    def is_success_all_tasks(self):
        num_beads_in_mug_milk = len(objects_center_in_container(
            self.objects["faucet_milk"].beads, self.objects["mug"].body_id))
        num_beads_in_mug_coffee = len(objects_center_in_container(
            self.objects["faucet_coffee"].beads, self.objects["mug"].body_id))

        successes = {
            "fill_mug": num_beads_in_mug_milk >= 3 if self._target_faucet == "faucet_milk" else num_beads_in_mug_coffee >= 3,
            "fill_mug_any": num_beads_in_mug_milk >= 3 or num_beads_in_mug_coffee >= 3,
        }
        successes["task"] = successes["fill_mug"]
        return successes

    def _create_skill_lib(self):
        def fill_mug(objects, pl):
            num_beads_in_bowl = len(objects_center_in_container(
                objects["faucet_" + pl].beads, objects["mug"].body_id))
            return num_beads_in_bowl >= 3

        lib_skills = (
            skills.GraspDistDiscreteOrn(
                name="grasp", lift_height=0.1, lift_speed=0.01
            ),
            skills.GraspDistDiscreteOrn(
                name="grasp_fill_mug_milk", lift_height=0.1, lift_speed=0.01,
                precondition_fn=lambda objs=self.objects: fill_mug(self.objects, "milk"),
            ),
            skills.GraspDistDiscreteOrn(
                name="grasp_fill_mug_coffee", lift_height=0.1, lift_speed=0.01,
                precondition_fn=lambda objs=self.objects: fill_mug(self.objects, "coffee"),
            ),
            skills.PlacePosDiscreteOrn(
                name="place", retract_distance=0.1, num_pause_steps=30,
            ),
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills, verbose=True)

    def get_demo_suboptimal(self, noise=None):
        self.reset()
        param_set = OrderedDict()
        param_set["grasp"] = self.skill_lib.sample_serialized_skill_params(
            "grasp",
            grasp_orn=dict(choices=[3]),
            grasp_distance=dict(low=[0.05], high=[0.05])
        )

        param_set["place"] = self.skill_lib.sample_serialized_skill_params(
            "place",
            place_orn=dict(choices=[0]),
            place_pos=dict(low=(-0.025, -0.075, 0.01), high=(0.025, 0.075, 0.01))
        )

        param_set["grasp_fill_mug_milk"] = self.skill_lib.sample_serialized_skill_params(
            "grasp_fill_mug_milk",
            grasp_orn=dict(choices=[3]),
            grasp_distance=dict(low=[0.05], high=[0.05])
        )

        param_set["grasp_fill_mug_coffee"] = self.skill_lib.sample_serialized_skill_params(
            "grasp_fill_mug_coffee",
            grasp_orn=dict(choices=[3]),
            grasp_distance=dict(low=[0.05], high=[0.05])
        )

        buffer = PU.Buffer()
        exception = None
        for skill_step in range(10):
            object_id = np.random.choice(self.objects.body_ids)
            skill_name = np.random.choice(list(param_set.keys()))
            skill_param = param_set[skill_name]
            print(skill_step, skill_name, object_id)

            traj, exec_info = PU.execute_skill(
                self, self.skill_lib, skill_param,
                target_object_id=object_id,
                skill_step=skill_step,
                noise=noise
            )
            buffer.append(**traj)
            if exec_info["exception"] is not None:
                exception = exec_info["exception"]
            if self.is_success():
                break

        return buffer.aggregate(), exception


class Kitchen(BaseEnv):
    def __init__(self, **kwargs):
        kwargs["robot_base_pose"] = ([0.5, 0.3, 1.2], [0, 0, 1, 0])
        super(Kitchen, self).__init__(**kwargs)

    def _create_sensors(self):
        PBU.set_camera(45, -45, 2.0, (0, 0, 0.7))
        self.camera = Camera(
            height=self._camera_width,
            width=self._camera_height,
            fov=60,
            near=0.01,
            far=10.,
            renderer=p.ER_TINY_RENDERER
        )
        self.camera.set_pose_ypr((0, 0, 0.7), distance=2.0, yaw=45, pitch=-45)

    def _create_fixtures(self):
        p.loadMJCF(os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml"))
        box = Box(color=(0.5, 0.5, 0.5, 1.0), size=(0.6, 1.0, 1.0))
        box.load()
        box.set_position((0.0, 1.0, 0.5))
        p.changeDynamics(box.body_id, -1, mass=1000.0)
        self.fixtures.add_object("platform1", box)

    def _create_objects(self):
        drawer = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf'))
        drawer.load()
        drawer.set_position([0, 0, 0.5])
        set_articulated_object_dynamics(drawer.body_id)
        self.objects.add_object("drawer", drawer)

        stove = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cooktop/textured.urdf'), scale=0.7)
        stove.load()
        p.changeDynamics(stove.body_id, -1, mass=1000.0)

        self.objects.add_object("stove", stove)

        can = YCBObject('005_tomato_soup_can')
        can.load()
        p.changeDynamics(can.body_id, -1, mass=1.0)
        set_friction(can.body_id)
        self.objects.add_object("can", can)
        # cabinet = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf'))
        # cabinet.load()
        # cabinet.set_position([0, 0, 2])
        # set_articulated_object_dynamics(cabinet.body_id)
        # self.objects.add_object("cabinet", cabinet)

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["stove"].body_id, self.fixtures["platform1"].body_id)
        self.objects["stove"].set_position_orientation(
            PU.sample_positions_in_box([0.0, 0.0], [1.0, 1.0], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["can"].body_id, self.objects["drawer"].body_id, surface_link=2) - 0.15
        self.objects["can"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [0.0, 0.0], [z, z]), PBU.unit_quat())

    def _create_skill_lib(self):
        lib_skills = (
            skills.GraspDistDiscreteOrn(lift_height=0.1, lift_speed=0.01),
            skills.PlacePosDiscreteOrn(retract_distance=0.1, num_pause_steps=30),
            skills.PourPosAngle(pour_angle_speed=np.pi / 32, num_pause_steps=30),
            skills.OperatePrismaticPosDistance()
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills, verbose=True)

    def get_demo_expert(self, noise=None):
        # for i in range(4):
        #     p.changeDynamics(self.robot.body_id, i, mass=0.01)
        self.reset()
        buffer = PU.Buffer()
        skill_seq = []
        params = self.skill_lib.get_serialized_skill_params(
            "operate_prismatic_pos_distance", grasp_pos=(0.37, 0.0, 0.19), prismatic_move_distance=-0.4)
        skill_seq.append((params, self.objects["drawer"].body_id))

        params = self.skill_lib.get_serialized_skill_params(
            "grasp_dist_discrete_orn", grasp_orn_name="top", grasp_distance=0.05)
        skill_seq.append((params, self.objects["can"].body_id))

        params = self.skill_lib.get_serialized_skill_params(
            "place_pos_discrete_orn", place_orn_name="front", place_pos=[0, 0, 0])
        skill_seq.append((params, self.objects["stove"].body_id))

        for skill_step, (skill_param, object_id) in enumerate(skill_seq):
            traj, exec_info = PU.execute_skill(
                self, self.skill_lib, skill_param,
                target_object_id=object_id,
                skill_step=skill_step,
                noise=noise
            )
            buffer.append(**traj)
        return buffer.aggregate(), None
