import numpy as np
import os
from collections import OrderedDict

import pybullet as p
import pybullet_data
import gibson2
from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, Object, VisualMarker
import gibson2.external.pybullet_tools.transformations as T

from gibson2.envs.kitchen.camera import Camera
import gibson2.envs.kitchen.env_utils as EU
import gibson2.external.pybullet_tools.utils as PBU
import gibson2.envs.kitchen.plan_utils as PU
import gibson2.envs.kitchen.skills as skills
from gibson2.envs.kitchen.objects import Faucet, Box, CoffeeMachine, CoffeeGrinder
from gibson2.envs.kitchen.base_env import BaseEnv, EnvSkillWrapper
from gibson2.envs.kitchen.envs import TableTop


class TablePour(TableTop):
    def _get_feature_observation(self):
        num_beads = np.zeros((len(self.objects), 2))
        for i, o in enumerate(self.objects.object_list):
            num_beads[i, 0] = len(
                EU.objects_center_in_container(self.objects["faucet_coffee"].beads, container_id=o.body_id)
            )
            num_beads[i, 1] = len(
                EU.objects_center_in_container(self.objects["faucet_milk"].beads, container_id=o.body_id)
            )
        obs = dict(
            num_beads=num_beads
        )
        return obs

    def _sample_task(self):
        pass

    def _create_objects(self):
        o = CoffeeGrinder()
        o.load()
        self.objects.add_object("grinder", o)
        self.interactive_objects.add_object("grinder", o)

        o = YCBObject('025_mug')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.0)
        EU.set_friction(o.body_id)
        self.objects.add_object("mug", o)

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["grinder"].body_id, self.fixtures["table"].body_id)
        self.objects["grinder"].set_position_orientation(
            PU.sample_positions_in_box([0.0, 0.0], [0.0, 0.0], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["mug"].body_id, self.objects["grinder"].body_id)
        self.objects["mug"].set_position_orientation(
            PU.sample_positions_in_box([0.0, 0.0], [0.0, 0.0], [z, z]), PBU.unit_quat())

    def _create_skill_lib(self):
        def fill_bowl(objects, pl, oid):
            return len(EU.objects_center_in_container(objects["faucet_" + pl].beads, oid)) >= 3

        lib_skills = (
            skills.GraspDistDiscreteOrn(
                name="grasp", lift_height=0.1, lift_speed=0.01,
                params=OrderedDict(
                    grasp_distance=skills.SkillParamsContinuous(low=[0.05], high=[0.05]),
                    grasp_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS))
                )
            ),
            skills.GraspDistDiscreteOrn(
                name="grasp_fill_mug_any", lift_height=0.1, lift_speed=0.01,
                precondition_fn=lambda oid: self.is_success_all_tasks()["fill_mug_any"],
                params=OrderedDict(
                    grasp_distance=skills.SkillParamsContinuous(low=[0.05], high=[0.05]),
                    grasp_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS)),
                )
            ),
            skills.PlacePosDiscreteOrn(
                name="place", retract_distance=0.1, num_pause_steps=30,
                params=OrderedDict(
                    place_pos=skills.SkillParamsContinuous(low=(-0.025, -0.075, 0.01), high=(0.025, 0.075, 0.01)),
                    place_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS)),
                )
            ),
            skills.PourPosAngle(
                name="pour", pour_angle_speed=np.pi / 32, num_pause_steps=30,
                params=OrderedDict(
                    pour_pos=skills.SkillParamsContinuous(low=(-0.1, -0.1, 0.3), high=(0.1, 0.1, 0.3)),
                    pour_angle=skills.SkillParamsContinuous(low=(np.pi * 0.25,), high=(np.pi,))
                )
            ),
            skills.ConditionSkill(
                name="fill_bowl_milk", precondition_fn=lambda oid: fill_bowl(self.objects, "milk", oid),
            ),
            skills.ConditionSkill(
                name="fill_bowl_coffee", precondition_fn=lambda oid: fill_bowl(self.objects, "coffee", oid)
            )
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
        EU.set_friction(o.body_id)
        self.objects.add_object("mug", o)

        o = Faucet(num_beads=10, dispense_freq=1,
                   beads_color=(111 / 255, 78 / 255, 55 / 255, 1),
                   base_color=(150 / 255, 100 / 255, 75 / 255, 1))
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

        # p.resetBasePositionAndOrientation(self.objects["microwave"].body_id, (0, 0, 0), skills.SKILL_ORIENTATIONS["top"])
        # z = PBU.stable_z(self.objects["microwave"].body_id, self.fixtures["table"].body_id)
        # self.objects["microwave"].set_position_orientation(
        #     PU.sample_positions_in_box([-0.3, -0.2], [-0.3, -0.3], [z, z]),  skills.SKILL_ORIENTATIONS["top"])

    def _get_feature_observation(self):
        num_beads = np.zeros((len(self.objects), 2))
        for i, o in enumerate(self.objects.object_list):
            num_beads[i, 0] = len(
                EU.objects_center_in_container(self.objects["faucet_coffee"].beads, container_id=o.body_id)
            )
            num_beads[i, 1] = len(
                EU.objects_center_in_container(self.objects["faucet_milk"].beads, container_id=o.body_id)
            )
        obs = dict(
            num_beads=num_beads
        )
        return obs


class KitchenCoffeeAP(KitchenCoffee):
    def is_success_all_tasks(self):
        num_beads_in_mug_milk = len(EU.objects_center_in_container(
            self.objects["faucet_milk"].beads, self.objects["mug"].body_id))
        num_beads_in_bowl_milk = len(EU.objects_center_in_container(
            self.objects["faucet_milk"].beads, self.objects["bowl"].body_id))
        num_beads_in_mug_coffee = len(EU.objects_center_in_container(
            self.objects["faucet_coffee"].beads, self.objects["mug"].body_id))
        num_beads_in_bowl_coffee = len(EU.objects_center_in_container(
            self.objects["faucet_coffee"].beads, self.objects["bowl"].body_id))

        successes = {
            "fill_mug": num_beads_in_mug_milk >= 3 if self._target_faucet == "faucet_milk" else num_beads_in_mug_coffee >= 3,
            "fill_bowl": num_beads_in_bowl_milk >= 3 if self._target_faucet == "faucet_milk" else num_beads_in_bowl_coffee >= 3,
            "fill_mug_any": num_beads_in_mug_milk >= 3 or num_beads_in_mug_coffee >= 3,
            "fill_bowl_any": num_beads_in_bowl_milk >= 3 or num_beads_in_bowl_coffee >= 3,
        }
        successes["task"] = successes["fill_bowl"]
        return successes

    def _sample_task(self):
        skill_name = np.random.choice(["fill_bowl_milk", "fill_bowl_coffee"])
        self._task_spec = np.array([self.skill_lib.name_to_skill_index(skill_name), self.objects.names.index("bowl")])
        self._target_faucet = "faucet_milk" if skill_name == "fill_bowl_milk" else "faucet_coffee"
        self._task_skill_name = skill_name
        self._task_object_name = "bowl"

    def _create_skill_lib(self):
        def fill_bowl(objects, pl, oid):
            return len(EU.objects_center_in_container(objects["faucet_" + pl].beads, oid)) >= 3

        lib_skills = (
            skills.GraspDistDiscreteOrn(
                name="grasp", lift_height=0.1, lift_speed=0.01,
                params=OrderedDict(
                    grasp_distance=skills.SkillParamsContinuous(low=[0.05], high=[0.05]),
                    grasp_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS))
                )
            ),
            skills.GraspDistDiscreteOrn(
                name="grasp_fill_mug_any", lift_height=0.1, lift_speed=0.01,
                precondition_fn=lambda oid: self.is_success_all_tasks()["fill_mug_any"],
                params=OrderedDict(
                    grasp_distance=skills.SkillParamsContinuous(low=[0.05], high=[0.05]),
                    grasp_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS)),
                )
            ),
            skills.PlacePosDiscreteOrn(
                name="place", retract_distance=0.1, num_pause_steps=30,
                params=OrderedDict(
                    place_pos=skills.SkillParamsContinuous(low=(-0.025, -0.075, 0.01), high=(0.025, 0.075, 0.01)),
                    place_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS)),
                )
            ),
            skills.PourPosAngle(
                name="pour", pour_angle_speed=np.pi / 32, num_pause_steps=30,
                params=OrderedDict(
                    pour_pos=skills.SkillParamsContinuous(low=(-0.1, -0.1, 0.3), high=(0.1, 0.1, 0.3)),
                    pour_angle=skills.SkillParamsContinuous(low=(np.pi * 0.25,), high=(np.pi,))
                )
            ),
            skills.ConditionSkill(
                name="fill_bowl_milk", precondition_fn=lambda oid: fill_bowl(self.objects, "milk", oid),
            ),
            skills.ConditionSkill(
                name="fill_bowl_coffee", precondition_fn=lambda oid: fill_bowl(self.objects, "coffee", oid)
            )
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills)

    def get_random_skeleton(self, horizon):
        param_set = OrderedDict()
        param_set["grasp"] = lambda: self.skill_lib.sample_serialized_skill_params(
            "grasp", grasp_orn=dict(choices=[3]),
        )
        param_set["place"] = lambda: self.skill_lib.sample_serialized_skill_params(
            "place", place_orn=dict(choices=[0]),
        )
        param_set["pour"] = lambda: self.skill_lib.sample_serialized_skill_params("pour")
        param_set["fill_bowl_milk"] = lambda: self.skill_lib.sample_serialized_skill_params("fill_bowl_milk")
        param_set["fill_bowl_coffee"] = lambda: self.skill_lib.sample_serialized_skill_params("fill_bowl_coffee")
        skeleton = []
        for _ in range(horizon):
            object_name = np.random.choice(self.objects.names)
            skill_name = np.random.choice(list(param_set.keys()))
            skeleton.append((param_set[skill_name], object_name))
        return skeleton

    def get_task_skeleton(self):
        param_seq = []
        param_seq.append((
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            "mug"
        ))
        param_seq.append((
            lambda: self.skill_lib.sample_serialized_skill_params("place", place_orn=dict(choices=[0])),
            self._target_faucet
        ))
        param_seq.append((
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            "mug"
        ))

        def get_pour_pos():
            pour_delta = np.array([0, 0, 0.3])
            pour_delta[:2] += (np.random.rand(2) * 0.07 + 0.03) * np.random.choice([-1, 1], size=2)
            return pour_delta

        param_seq.append((
            lambda: self.skill_lib.sample_serialized_skill_params("pour", pour_pos=dict(sampler_fn=get_pour_pos)),
            "bowl"
        ))
        param_seq.append((
            lambda: self.skill_lib.sample_serialized_skill_params(self._task_skill_name),
            self._task_object_name
        ))
        return param_seq


class SimpleCoffeeAP(KitchenCoffee):
    def _sample_task(self):
        skill_name = np.random.choice(["fill_mug_milk", "fill_mug_coffee"])
        self._task_spec = np.array([self.skill_lib.name_to_skill_index(skill_name), self.objects.names.index("mug")])
        self._target_faucet = "faucet_milk" if skill_name == "fill_mug_milk" else "faucet_coffee"
        self._task_skill_name = skill_name
        self._task_object_name = "mug"

    def is_success_all_tasks(self):
        num_beads_in_mug_milk = len(EU.objects_center_in_container(
            self.objects["faucet_milk"].beads, self.objects["mug"].body_id))
        num_beads_in_mug_coffee = len(EU.objects_center_in_container(
            self.objects["faucet_coffee"].beads, self.objects["mug"].body_id))

        successes = {
            "fill_mug": num_beads_in_mug_milk >= 3 if self._target_faucet == "faucet_milk" else num_beads_in_mug_coffee >= 3,
            "fill_mug_any": num_beads_in_mug_milk >= 3 or num_beads_in_mug_coffee >= 3,
        }
        successes["task"] = successes["fill_mug"]
        return successes

    def _create_skill_lib(self):
        def fill_mug(objects, pl, oid):
            return len(EU.objects_center_in_container(objects["faucet_" + pl].beads, oid)) >= 3

        lib_skills = (
            skills.GraspDistDiscreteOrn(
                name="grasp", lift_height=0.1, lift_speed=0.01,
                params=OrderedDict(
                    grasp_distance=skills.SkillParamsContinuous(low=[0.05], high=[0.05]),
                    grasp_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS))
                )
            ),
            skills.PlacePosDiscreteOrn(
                name="place", retract_distance=0.1, num_pause_steps=30,
                params=OrderedDict(
                    place_pos=skills.SkillParamsContinuous(low=(-0.025, -0.075, 0.01), high=(0.025, 0.075, 0.01)),
                    place_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS)),
                )
            ),
            skills.ConditionSkill(
                name="fill_mug_milk", precondition_fn=lambda oid: fill_mug(self.objects, "milk", oid),
            ),
            skills.ConditionSkill(
                name="fill_mug_coffee", precondition_fn=lambda oid: fill_mug(self.objects, "coffee", oid),
            )
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills)

    def get_random_skeleton(self, horizon):
        param_set = OrderedDict()
        param_set["grasp"] = lambda: self.skill_lib.sample_serialized_skill_params(
            "grasp", grasp_orn=dict(choices=[3]),
        )
        param_set["place"] = lambda: self.skill_lib.sample_serialized_skill_params(
            "place", place_orn=dict(choices=[0]),
        )
        param_set["fill_mug_milk"] = lambda: self.skill_lib.sample_serialized_skill_params("fill_mug_milk")
        param_set["fill_mug_coffee"] = lambda: self.skill_lib.sample_serialized_skill_params("fill_mug_coffee")
        skeleton = []
        for _ in range(horizon):
            object_name = np.random.choice(self.objects.names)
            skill_name = np.random.choice(list(param_set.keys()))
            skeleton.append((param_set[skill_name], object_name))
        return skeleton

    def get_task_skeleton(self):
        param_seq = [(
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            "mug"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("place", place_orn=dict(choices=[0])),
            self._target_faucet
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params(self._task_skill_name),
            self._task_object_name
        )]
        return param_seq


class Kitchen(BaseEnv):
    def __init__(self, **kwargs):
        kwargs["robot_base_pose"] = ([0.8, 0.3, 1.2], [0, 0, 1, 0])
        super(Kitchen, self).__init__(**kwargs)

    def _create_sensors(self):
        PBU.set_camera(45, -60, 1.5, (0, 0, 0.7))
        self.camera = Camera(
            height=self._camera_width,
            width=self._camera_height,
            fov=60,
            near=0.01,
            far=10.,
            renderer=p.ER_TINY_RENDERER
        )
        self.camera.set_pose_ypr((0, 0, 0.7), distance=1.5, yaw=45, pitch=-60)

    def _get_feature_observation(self):
        num_beads = np.zeros((len(self.objects), 2))
        for i, o in enumerate(self.objects.object_list):
            num_beads[i, 0] = len(
                EU.objects_center_in_container(self.objects["faucet_coffee"].beads, container_id=o.body_id)
            )
            num_beads[i, 1] = len(
                EU.objects_center_in_container(self.objects["coffee_machine"].beads, container_id=o.body_id)
            )
        obs = dict(
            num_beads=num_beads
        )
        return obs

    def _create_fixtures(self):
        pass
        # p.loadMJCF(os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml"))

    def _create_objects(self):
        drawer = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf'))
        drawer.load()
        drawer.set_position([0, 0, 0.5])
        p.setJointMotorControl2(drawer.body_id, 2, p.POSITION_CONTROL, force=0)
        EU.set_friction(drawer.body_id, friction=10.)
        self.objects.add_object("drawer", drawer)

        o = Box(color=(0.9, 0.9, 0.9, 1))
        o.load()
        self.objects.add_object("platform1", o, category="platform")

        # o = Box(color=(0.9, 0.9, 0.9, 1))
        # o.load()
        # self.objects.add_object("platform2", o, category="platform")

        o = YCBObject('025_mug')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.5)
        EU.set_friction(o.body_id)
        self.objects.add_object("mug1", o, category="mug")

        # o = YCBObject('025_mug')
        # o.load()
        # p.changeDynamics(o.body_id, -1, mass=1.5)
        # EU.set_friction(o.body_id)
        # self.objects.add_object("mug2", o, category="mug")

        o = CoffeeGrinder(mass=1000.0)
        o.load()
        self.objects.add_object("grinder", o)

        o = Faucet(num_beads=10, dispense_freq=1,
                   beads_color=(111 / 255, 78 / 255, 55 / 255, 1),
                   base_color=(0.5, 0.5, 0.5, 0.0),
                   base_size=(0.13, 0.13, 0.001), dispense_height=0.13, exclude_ids=(self.objects["grinder"].body_id,))
        o.load()
        self.objects.add_object("faucet_coffee", o)
        self.interactive_objects.add_object("faucet_coffee", o)

        o = CoffeeMachine(
            filename=os.path.join(gibson2.assets_path, "models/coffee_machine_slim/102901.urdf"),
            beans_set=self.objects["faucet_coffee"].beads,
            num_beans_trigger=5,
            dispense_position=np.array([0.05, 0, 0.02]),
            platform_position=np.array([0.07, 0, -0.13]),
            button_pose=((0.0, -0.13, 0.16), skills.ALL_ORIENTATIONS["backward"]),
            scale=0.25
        )
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1000.0)
        self.objects.add_object("coffee_machine", o)
        self.interactive_objects.add_object("coffee_machine", o)
        self.objects.add_object("coffee_machine_platform", o.platform)
        self.objects.add_object("coffee_machine_button", o.button)

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["coffee_machine"].body_id, self.objects["drawer"].body_id)
        self.objects["coffee_machine"].set_position_orientation(
            PU.sample_positions_in_box([-0.25, -0.15], [-0.25, -0.2], [z, z]), skills.ALL_ORIENTATIONS["left"])

        z = PBU.stable_z(self.objects["platform1"].body_id, self.objects["drawer"].body_id)
        self.objects["platform1"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [-0.1, -0.1], [z, z]), PBU.unit_quat())

        # self.objects["platform2"].set_position_orientation(
        #     PU.sample_positions_in_box([0.2, 0.2], [-0.3, -0.3], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["mug1"].body_id, self.objects["drawer"].body_id, surface_link=2)
        z -= 0.15
        self.objects["mug1"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [-0.05, 0.05], [z, z]), PBU.unit_quat())

        # z = PBU.stable_z(self.objects["mug2"].body_id, self.objects["platform2"].body_id)
        # self.objects["mug2"].set_position_orientation(
        #     PU.sample_positions_in_box([0.175, 0.225], [-0.325, -0.275], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["grinder"].body_id, self.objects["drawer"].body_id)
        pos = PU.sample_positions_in_box([-0.25, -0.15], [0.2, 0.2], [z, z])
        self.objects["grinder"].set_position_orientation(pos, skills.ALL_ORIENTATIONS["back"])

        z = PBU.stable_z(self.objects["faucet_coffee"].body_id, self.objects["grinder"].body_id, surface_link=-1)
        pos = PBU.get_pose(self.objects["grinder"].body_id)[0]
        self.objects["faucet_coffee"].set_position_orientation((pos[0], pos[1], z), PBU.unit_quat())

        EU.set_collision_between(self.robot.gripper.body_id, self.objects["coffee_machine"].body_id, 0)

    def _create_skill_lib(self):

        lib_skills = (
            skills.GraspDistDiscreteOrn(
                name="grasp", lift_height=0.02, lift_speed=0.01, reach_distance=0.02,
                params=OrderedDict(
                    grasp_distance=skills.SkillParamsContinuous(low=[0.03], high=[0.05]),
                    grasp_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS))
                ),
                # joint_resolutions=(0.05, 0.05, 0.05, np.pi / 32, np.pi / 32, np.pi / 32)
            ),
            skills.PlacePosYawOrn(
                name="place", retract_distance=0.1, num_pause_steps=30,
                params=OrderedDict(
                    place_pos=skills.SkillParamsContinuous(low=[-0.05, -0.05, 0.01], high=[0.05, 0.05, 0.01]),
                    place_orn=skills.SkillParamsContinuous(low=[-np.pi / 12], high=[np.pi / 12])
                ),
                joint_resolutions=(0.05, 0.05, 0.05, np.pi / 32, np.pi / 32, np.pi / 32)
            ),
            skills.PourPosAngle(
                name="pour", pour_angle_speed=np.pi / 32, num_pause_steps=30,
                params=OrderedDict(
                    pour_pos=skills.SkillParamsContinuous(low=(-0.1, -0.1, 0.5), high=(0.1, 0.1, 0.5)),
                    pour_angle=skills.SkillParamsContinuous(low=(np.pi * 0.25,), high=(np.pi,))
                ),
                joint_resolutions=(0.05, 0.05, 0.05, np.pi / 32, np.pi / 32, np.pi / 32)
            ),
            skills.OperatePrismaticPosDistance(
                name="open_prismatic",
                params=OrderedDict(
                    grasp_pos=skills.SkillParamsContinuous(low=[0.35, -0.05, 0.15], high=[0.45, 0.05, 0.25]),
                    prismatic_move_distance=skills.SkillParamsContinuous(low=[-0.3], high=[-0.1])
                ),
                # joint_resolutions=(0.05, 0.05, 0.05, np.pi / 32, np.pi / 32, np.pi / 32)
            ),
            skills.TouchPosition(
                name="touch", num_pause_steps=30,
                params=OrderedDict(
                    touch_pos=skills.SkillParamsContinuous(low=[-0.03, -0.03, 0.0], high=[0.03, 0.03, 0.05]),
                ),
                # joint_resolutions=(0.05, 0.05, 0.05, np.pi / 32, np.pi / 32, np.pi / 32)
            ),
            skills.ConditionSkill(
                name="filled_coffee", precondition_fn=lambda oid: len(EU.objects_center_in_container(
                    self.objects["coffee_machine"].beads, oid)) >= 3
            ),
            skills.ConditionSkill(
                name="on_platform1", precondition_fn=lambda oid: PBU.is_center_placed_on(
                    oid, self.objects["platform1"].body_id)
            ),
            skills.ConditionSkill(
                name="filled_on_platform1", precondition_fn=lambda oid:
                PBU.is_center_placed_on(oid, self.objects["platform1"].body_id) and \
                len(EU.objects_center_in_container(self.objects["coffee_machine"].beads, oid)) >= 3
            )
        )
        # PBU.draw_aabb(aabb=[[0.35, -0.15, 0.15 + 0.5], [0.45, 0.15, 0.25 + 0.5]])
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills)


class KitchenAP(Kitchen):
    def _get_pour_pos(self):
        pour_delta = np.array([0, 0, 0.5])
        pour_delta[:2] += (np.random.rand(2) * 0.07 + 0.03) * np.random.choice([-1, 1], size=2)
        return pour_delta

    def is_success_all_tasks(self):
        # num_beans_in_mug2 = len(EU.objects_center_in_container(
        #     self.objects["faucet_coffee"].beads, self.objects["mug2"].body_id))
        mug1_graspable = PBU.is_center_placed_on(self.objects["mug1"].body_id, self.objects["platform1"].body_id)
        num_coffee_in_mug1 = len(EU.objects_center_in_container(
            self.objects["coffee_machine"].beads, self.objects["mug1"].body_id))
        num_beans_in_mug1 = len(EU.objects_center_in_container(
            self.objects["faucet_coffee"].beads, self.objects["mug1"].body_id))
        mug1_on_platform = PBU.is_center_placed_on(self.objects["mug1"].body_id, self.objects["platform1"].body_id)

        successes = {
            "fill_mug1_coffee": num_coffee_in_mug1 >= 3,
            "fill_mug1_coffee_on_platform": num_coffee_in_mug1 >= 3 and mug1_on_platform,
            "fill_mug1_beans": num_beans_in_mug1 >= 3,
            "mug1_graspable": mug1_graspable
        }
        successes["task"] = successes["fill_mug1_coffee_on_platform"]
        return successes

    def _sample_task(self):
        self._task_spec = np.array([self.skill_lib.name_to_skill_index("filled_on_platform1"), self.objects.names.index("mug1")])

    def get_task_skeleton(self):
        skeleton = [(
            # lambda: self.skill_lib.sample_serialized_skill_params("open_prismatic", grasp_pos=dict(low=[0.4, 0, 0.2], high=[0.4, 0, 0.2]), prismatic_move_distance=dict(low=[-0.3], high=[-0.3])),
            lambda: self.skill_lib.sample_serialized_skill_params("open_prismatic"),
            "drawer"
        ), (
            # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[4])),
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
            "mug1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "platform1"
        ), (
            # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
            "mug1"
        ), (
            # lambda: self.skill_lib.sample_serialized_skill_params("place", place_pos=dict(low=[-0.02, -0.02, 0.01], high=[0.02, 0.02, 0.01])),
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "faucet_coffee"
        ), (
            # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
            "mug1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("pour", pour_pos=dict(sampler_fn=self._get_pour_pos)),
            "coffee_machine_platform"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("place", place_pos=dict(low=[-0.02, -0.02, 0.01], high=[0.02, 0.02, 0.01])),
            # lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "coffee_machine_platform"
        ), (
        #     lambda: self.skill_lib.sample_serialized_skill_params("touch"),
        #     "coffee_machine_button"
        # ), (
        #     lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
            "mug1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "platform1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("filled_on_platform1"),
            "mug1"
        )]
        return skeleton

    def get_constrained_skill_param_sampler(self, skill_name, object_name, num_samples=None):
        if skill_name == "grasp":
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "grasp", num_samples=num_samples, grasp_orn=dict(choices=[3, 4]))
        elif skill_name == "place" and object_name == "coffee_machine_platform":
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "place", num_samples=num_samples, place_pos=dict(low=[-0.02, -0.02, 0.01], high=[0.02, 0.02, 0.01]))
        elif skill_name == "pour":
            return lambda: self.skill_lib.sample_serialized_skill_params("pour", num_samples=num_samples, pour_pos=dict(sampler_fn=self._get_pour_pos))
        else:
            return lambda: self.skill_lib.sample_serialized_skill_params(skill_name, num_samples=num_samples)

    #
    # def get_task_skeleton(self):
    #     skeleton = [(
    #         # lambda: self.skill_lib.sample_serialized_skill_params("open_prismatic", grasp_pos=dict(low=[0.4, 0, 0.2], high=[0.4, 0, 0.2]), prismatic_move_distance=dict(low=[-0.3], high=[-0.3])),
    #         lambda: self.skill_lib.sample_serialized_skill_params("open_prismatic"),
    #         "drawer"
    #     ),  (
    #         # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[4])),
    #         lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
    #         "mug1"
    #     ), (
    #         lambda: self.skill_lib.sample_serialized_skill_params("place"),
    #         "platform1"
    #     ), (
    #         # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
    #         lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
    #         "mug1"
    #     ),  (
    #         lambda: self.skill_lib.sample_serialized_skill_params("place", place_pos=dict(low=[-0.02, -0.02, 0.01], high=[0.02, 0.02, 0.01])),
    #         "coffee_machine_platform"
    #     ), (
    #         # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
    #         lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
    #         "mug2"
    #     ),  (
    #         lambda: self.skill_lib.sample_serialized_skill_params("place"),
    #         "faucet_coffee"
    #     ),  (
    #         # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
    #         lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
    #         "mug2"
    #     ),  (
    #         lambda: self.skill_lib.sample_serialized_skill_params("pour"),
    #         "coffee_machine"
    #     ),   (
    #         lambda: self.skill_lib.sample_serialized_skill_params("place"),
    #         "platform2"
    #     ), (
    #         lambda: self.skill_lib.sample_serialized_skill_params("touch"),
    #         "coffee_machine_button"
    #     ), (
    #     #     lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
    #     #     "mug1"
    #     # ), (
    #     #     lambda: self.skill_lib.sample_serialized_skill_params("place"),
    #     #     "platform1"
    #     # ), (
    #         lambda: self.skill_lib.sample_serialized_skill_params("mug_coffee"),
    #         "mug1"
    #     )]
    #     return skeleton


class KitchenEasyAP(KitchenAP):
    def is_success_all_tasks(self):
        successes = super(KitchenEasyAP, self).is_success_all_tasks()
        successes["task"] = successes["fill_mug1_coffee"]
        return successes

    def _sample_task(self):
        self._task_spec = np.array([self.skill_lib.name_to_skill_index("filled_coffee"), self.objects.names.index("mug1")])

    def get_task_skeleton(self):
        skeleton = [(
            lambda: self.skill_lib.sample_serialized_skill_params("open_prismatic", grasp_pos=dict(low=[0.4, 0, 0.2], high=[0.4, 0, 0.2]), prismatic_move_distance=dict(low=[-0.3], high=[-0.3])),
            "drawer"
        ), (
            # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[4])),
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
            "mug1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "platform1"
        ), (
            # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
            "mug1"
        ), (
            # lambda: self.skill_lib.sample_serialized_skill_params("place", place_pos=dict(low=[-0.02, -0.02, 0.01], high=[0.02, 0.02, 0.01])),
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "faucet_coffee"
        ), (
            # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
            "mug1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("pour"),
            "coffee_machine_platform"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("place", place_pos=dict(low=[0, 0, 0.01], high=[0, 0, 0.01])),
            "coffee_machine_platform"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("filled_coffee"),
            "mug1"
        )]
        return skeleton

    def get_constrained_skill_param_sampler(self, skill_name, object_name, num_samples=None):
        if skill_name == "grasp":
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "grasp", num_samples=num_samples, grasp_orn=dict(choices=[3, 4]))
        elif skill_name == "place" and object_name == "coffee_machine_platform":
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "place", num_samples=num_samples, place_pos=dict(low=[0, 0, 0.01], high=[0, 0, 0.01]))
        elif skill_name == "pour":
            return lambda: self.skill_lib.sample_serialized_skill_params("pour",  num_samples=num_samples, pour_pos=dict(sampler_fn=self._get_pour_pos))
        elif skill_name == "open_prismatic":
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "open_prismatic", num_samples=num_samples, grasp_pos=dict(low=[0.4, 0, 0.2], high=[0.4, 0, 0.2]), prismatic_move_distance=dict(low=[-0.3], high=[-0.3]))
        else:
            return lambda: self.skill_lib.sample_serialized_skill_params(skill_name, num_samples=num_samples)


class KitchenDrawerAP(Kitchen):
    def _sample_task(self):
        self._task_spec = np.array([self.skill_lib.name_to_skill_index("on_platform1"), self.objects.names.index("mug1")])

    def is_success_all_tasks(self):
        successes = {
            "mug1_on_platform1": PBU.is_center_placed_on(self.objects["mug1"].body_id, self.objects["platform1"].body_id)
        }
        successes["task"] = successes["mug1_on_platform1"]
        return successes

    def _reset_objects(self):
        super(KitchenDrawerAP, self)._reset_objects()
        z = PBU.stable_z(self.objects["mug1"].body_id, self.objects["drawer"].body_id, surface_link=2)
        z -= 0.15
        self.objects["mug1"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [-0.05, 0.05], [z, z]), PBU.unit_quat())

    def _create_skill_lib(self):
        super(KitchenDrawerAP, self)._create_skill_lib()
        self.skill_lib = self.skill_lib.sub_library(names=("grasp", "place", "open_prismatic", "on_platform1"))

    def get_task_skeleton(self):
        skeleton = [(
            # lambda: self.skill_lib.sample_serialized_skill_params("open_prismatic", grasp_pos=dict(low=[0.4, 0, 0.2], high=[0.4, 0, 0.2]), prismatic_move_distance=dict(low=[-0.3], high=[-0.3])),
            lambda: self.skill_lib.sample_serialized_skill_params("open_prismatic"),
            "drawer"
        ),  (
            # lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[4])),
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3, 4])),
            "mug1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "platform1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("on_platform1"),
            "mug1"
        )]
        return skeleton

    def get_random_skeleton(self, horizon):
        param_set = OrderedDict()
        param_set["grasp"] = lambda: self.skill_lib.sample_serialized_skill_params(
            "grasp", grasp_orn=dict(choices=[4]),
        )
        param_set["place"] = lambda: self.skill_lib.sample_serialized_skill_params("place")
        param_set["on_platform1"] = lambda: self.skill_lib.sample_serialized_skill_params("on_platform1")
        param_set["open_prismatic"] = lambda: self.skill_lib.sample_serialized_skill_params("open_prismatic")
        skeleton = []
        for _ in range(horizon):
            object_name = np.random.choice(self.objects.names)
            skill_name = np.random.choice(list(param_set.keys()))
            skeleton.append((param_set[skill_name], object_name))
        return skeleton

    # def _create_robot(self):
    #     from gibson2.envs.kitchen.robots import Gripper, JointActuatedRobot, Arm
    #     gripper = Gripper(
    #         joint_names=("left_gripper_joint", "right_gripper_joint"),
    #         finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip")
    #     )
    #     gripper.load(os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper_plannable.urdf'))
    #     arm = Arm(joint_names=("txj", "tyj", "tzj", "rxj", "ryj", "rzj"))
    #     arm.load(body_id=gripper.body_id)
    #     robot = JointActuatedRobot(
    #         eef_link_name="eef_link", init_base_pose=self._robot_base_pose, gripper=gripper, arm=arm)
    #
    #     self.robot = robot
    #
    # def step(self, action, sleep_per_sim_step=0.0, return_obs=True):
    #     assert len(action) == self.action_dimension
    #     import time
    #     action = action.copy()
    #     gri = action[-1]
    #     self.robot.close_loop_joint_control(action[:6])
    #     if gri > 0:
    #         self.robot.gripper.grasp()
    #     else:
    #         self.robot.gripper.ungrasp()
    #
    #     for o in self.interactive_objects.object_list:
    #         o.step(self.objects.object_list)
    #
    #     for _ in range(self._num_sim_per_step):
    #         p.stepSimulation()
    #         time.sleep(sleep_per_sim_step)
    #
    #     if not return_obs:
    #         return self.get_reward(), self.is_done(), {}
    #
    #     return self.get_observation(), self.get_reward(), self.is_done(), {}