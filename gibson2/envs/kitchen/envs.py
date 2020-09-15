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
from gibson2.envs.kitchen.objects import Faucet, Box, CoffeeMachine
from gibson2.envs.kitchen.base_env import BaseEnv, EnvSkillWrapper


def env_factory(name, **kwargs):
    if name.endswith("Skill"):
        name = name[:-5]
        kwargs["use_skills"] = True
        kwargs["use_planner"] = True
        return EnvSkillWrapper(eval(name)(**kwargs))
    else:
        return eval(name)(**kwargs)


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
                objects_center_in_container(self.objects["faucet_coffee"].beads, container_id=o.body_id)
            )
            num_beads[i, 1] = len(
                objects_center_in_container(self.objects["faucet_milk"].beads, container_id=o.body_id)
            )
        obs = dict(
            num_beads=num_beads
        )
        return obs


class KitchenCoffeeAP(KitchenCoffee):
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
                name="grasp", lift_height=0.1, lift_speed=0.01,
                params=OrderedDict(
                    grasp_distance=skills.SkillParamsContinuous(low=[0.05], high=[0.05]),
                    grasp_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS))
                )
            ),
            skills.GraspDistDiscreteOrn(
                name="grasp_fill_mug_any", lift_height=0.1, lift_speed=0.01,
                precondition_fn=lambda: self.is_success_all_tasks()["fill_mug_any"],
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
                name="fill_bowl_milk", precondition_fn=lambda objs=self.objects: fill_bowl(self.objects, "milk"),
            ),
            skills.ConditionSkill(
                name="fill_bowl_coffee", precondition_fn=lambda objs=self.objects: fill_bowl(self.objects, "coffee")
            )
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills, verbose=True)

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


class KitchenCoffeeTwoBowlAP(KitchenCoffeeAP):
    def _create_objects(self):
        super(KitchenCoffeeTwoBowlAP, self)._create_objects()
        o = YCBObject('024_bowl')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=10.0)
        self.objects.add_object("bowl/1", o, category="bowl")

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


class SimpleCoffeeAP(KitchenCoffee):
    def _sample_task(self):
        skill_name = np.random.choice(["fill_mug_milk", "fill_mug_coffee"])
        self._task_spec = np.array([self.skill_lib.name_to_skill_index(skill_name), self.objects.names.index("mug")])
        self._target_faucet = "faucet_milk" if skill_name == "fill_mug_milk" else "faucet_coffee"
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
                name="fill_mug_milk", precondition_fn=lambda objs=self.objects: fill_mug(self.objects, "milk"),
            ),
            skills.ConditionSkill(
                name="fill_mug_coffee", precondition_fn=lambda objs=self.objects: fill_mug(self.objects, "coffee"),
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

    def _get_feature_observation(self):
        num_beads = np.zeros((len(self.objects), 2))
        for i, o in enumerate(self.objects.object_list):
            num_beads[i, 0] = len(
                objects_center_in_container(self.objects["faucet_coffee"].beads, container_id=o.body_id)
            )
            num_beads[i, 1] = len(
                objects_center_in_container(self.objects["coffee_machine"].beads, container_id=o.body_id)
            )
        obs = dict(
            num_beads=num_beads
        )
        return obs

    def _create_fixtures(self):
        p.loadMJCF(os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml"))
        # box = Box(color=(0.5, 0.5, 0.5, 1.0), size=(0.6, 1.0, 1.0))
        # box.load()
        # box.set_position((0.0, 1.0, 0.5))
        # p.changeDynamics(box.body_id, -1, mass=1000.0)
        # self.fixtures.add_object("platform1", box)

    def _create_objects(self):
        drawer = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf'))
        drawer.load()
        drawer.set_position([0, 0, 0.5])
        set_articulated_object_dynamics(drawer.body_id)
        self.objects.add_object("drawer", drawer)

        o = Box(color=(0.9, 0.9, 0.9, 1))
        o.load()
        self.objects.add_object("platform1", o, category="platform")

        o = Box(color=(0.9, 0.9, 0.9, 1))
        o.load()
        self.objects.add_object("platform2", o, category="platform")

        # stove = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cooktop/textured.urdf'), scale=0.7)
        # stove.load()
        # p.changeDynamics(stove.body_id, -1, mass=1000.0)
        # self.objects.add_object("stove", stove)

        o = YCBObject('025_mug')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.0)
        set_friction(o.body_id)
        self.objects.add_object("mug1", o, category="mug")

        o = YCBObject('025_mug')
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.0)
        set_friction(o.body_id)
        self.objects.add_object("mug2", o, category="mug")

        o = Faucet(num_beads=10, dispense_freq=1,
                   beads_color=(111 / 255, 78 / 255, 55 / 255, 1),
                   base_color=(150 / 255, 100 / 255, 75 / 255, 1))
        o.load()
        self.objects.add_object("faucet_coffee", o)
        self.interactive_objects.add_object("faucet_coffee", o)

        o = CoffeeMachine(
            filename=os.path.join(gibson2.assets_path, "models/coffee_machine/102901.urdf"),
            beans_set=self.objects["faucet_coffee"].beads,
            num_beans_trigger=5,
            dispense_position=np.array([0.05, 0, 0.02]),
            platform_position=np.array([0.07, 0, -0.13]),
            button_pose=((0.0, -0.13, 0.16), skills.ALL_ORIENTATIONS["backward"]),
            scale=0.25
        )
        o.load()
        p.changeDynamics(o.body_id, -1, mass=100.0)
        self.objects.add_object("coffee_machine", o)
        self.interactive_objects.add_object("coffee_machine", o)
        self.objects.add_object("coffee_machine_platform", o.platform)
        self.objects.add_object("coffee_machine_button", o.button)

    def _reset_objects(self):
        # z = PBU.stable_z(self.objects["stove"].body_id, self.fixtures["platform1"].body_id)
        # self.objects["stove"].set_position_orientation(
        #     PU.sample_positions_in_box([0.0, 0.0], [1.0, 1.0], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["coffee_machine"].body_id, self.objects["drawer"].body_id)
        self.objects["coffee_machine"].set_position_orientation(
            PU.sample_positions_in_box([-0.2, -0.2], [0.0, 0.0], [z, z]), skills.SKILL_ORIENTATIONS["left"])

        z = PBU.stable_z(self.objects["platform1"].body_id, self.objects["drawer"].body_id)
        self.objects["platform1"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [0.0, 0.0], [z, z]), skills.SKILL_ORIENTATIONS["left"])

        self.objects["platform2"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [-0.3, -0.3], [z, z]), skills.SKILL_ORIENTATIONS["left"])

        z = PBU.stable_z(self.objects["mug1"].body_id, self.objects["platform1"].body_id)
        self.objects["mug1"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [0.0, 0.0], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["mug2"].body_id, self.objects["platform2"].body_id)
        self.objects["mug2"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [-0.3, -0.3], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["faucet_coffee"].body_id, self.objects["drawer"].body_id)
        pos = PU.sample_positions_in_box([0.2, 0.2], [0.2, 0.2], [z, z])
        coffee_pos = pos + np.array([0, 0.075, 0])
        self.objects["faucet_coffee"].set_position_orientation(coffee_pos, PBU.unit_quat())


class KitchenAP(Kitchen):
    def is_success_all_tasks(self):
        num_beans_in_mug2 = len(objects_center_in_container(
            self.objects["faucet_coffee"].beads, self.objects["mug2"].body_id))
        num_coffee_in_mug1 = len(objects_center_in_container(
            self.objects["coffee_machine"].beads, self.objects["mug1"].body_id))

        successes = {
            "fill_mug2_beans": num_beans_in_mug2 >= 3,
            "fill_mug1_coffee": num_coffee_in_mug1 >= 3
        }
        successes["task"] = successes["fill_mug1_coffee"]
        return successes

    def _sample_task(self):
        self._task_spec = np.array([self.skill_lib.name_to_skill_index("mug_coffee"), self.objects.names.index("mug1")])

    def _create_skill_lib(self):
        lib_skills = (
            skills.GraspDistDiscreteOrn(
                name="grasp", lift_height=0.05, lift_speed=0.01,
                params=OrderedDict(
                    grasp_distance=skills.SkillParamsContinuous(low=[0.05], high=[0.05]),
                    grasp_orn=skills.SkillParamsDiscrete(size=len(skills.SKILL_ORIENTATIONS))
                ),
                # joint_resolutions=(0.05, 0.05, 0.05, np.pi / 32, np.pi / 32, np.pi / 32)
            ),
            skills.PlacePosYawOrn(
                name="place", retract_distance=0.1, num_pause_steps=30,
                params=OrderedDict(
                    place_pos=skills.SkillParamsContinuous(low=[-0.05, -0.05, 0.01], high=[0.05, 0.05, 0.01]),
                    place_orn=skills.SkillParamsContinuous(low=[-np.pi / 6], high=[np.pi / 6])
                ),
                joint_resolutions=(0.025, 0.025, 0.025, np.pi / 32, np.pi / 32, np.pi / 32)
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
                    grasp_pos=skills.SkillParamsContinuous(low=[0.35, -0.03, 0.15], high=[0.4, 0.03, 0.25]),
                    prismatic_move_distance=skills.SkillParamsContinuous(low=[-0.3], high=[-0.2])
                ),
                joint_resolutions=(0.05, 0.05, 0.05, np.pi / 32, np.pi / 32, np.pi / 32)
            ),
            skills.TouchPosition(
                name="touch", num_pause_steps=30,
                params=OrderedDict(
                    touch_pos=skills.SkillParamsContinuous(low=[-0.03, -0.03, 0.0], high=[0.03, 0.03, 0.05]),
                ),
                # joint_resolutions=(0.05, 0.05, 0.05, np.pi / 32, np.pi / 32, np.pi / 32)
            ),
            skills.ConditionSkill(
                name="mug_coffee", precondition_fn=lambda oid: len(objects_center_in_container(
                    self.objects["coffee_machine"].beads, oid)) >= 3
            ),
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills)

    # def get_task_skeleton(self):
    #     skeleton = [(
    #         lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
    #         "mug1"
    #     ),  (
    #         lambda: self.skill_lib.sample_serialized_skill_params("place"),
    #         "faucet_coffee"
    #     ), (
    #         lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
    #         "mug1"
    #     ), (
    #         lambda: self.skill_lib.sample_serialized_skill_params("pour"),
    #         "coffee_machine"
    #     ),   (
    #         lambda: self.skill_lib.sample_serialized_skill_params("place", place_pos=dict(low=[-0.02, -0.02, 0.01], high=[0.02, 0.02, 0.01])),
    #         "coffee_machine_platform"
    #     ), (
    #         lambda: self.skill_lib.sample_serialized_skill_params("touch"),
    #         "coffee_machine_button"
    #     ), (
    #         lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
    #         "mug1"
    #     ), (
    #         lambda: self.skill_lib.sample_serialized_skill_params("place"),
    #         "platform1"
    #     ), ]
    #     return skeleton

    def get_task_skeleton(self):
        skeleton = [(
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            "mug1"
        ),  (
            lambda: self.skill_lib.sample_serialized_skill_params("place", place_pos=dict(low=[-0.02, -0.02, 0.01], high=[0.02, 0.02, 0.01])),
            "coffee_machine_platform"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            "mug2"
        ),  (
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "faucet_coffee"
        ),  (
            lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
            "mug2"
        ),  (
            lambda: self.skill_lib.sample_serialized_skill_params("pour"),
            "coffee_machine"
        ),   (
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "platform2"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("touch"),
            "coffee_machine_button"
        ), (
        #     lambda: self.skill_lib.sample_serialized_skill_params("grasp", grasp_orn=dict(choices=[3])),
        #     "mug1"
        # ), (
        #     lambda: self.skill_lib.sample_serialized_skill_params("place"),
        #     "platform1"
        # ), (
            lambda: self.skill_lib.sample_serialized_skill_params("mug_coffee"),
            "mug1"
        )]
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