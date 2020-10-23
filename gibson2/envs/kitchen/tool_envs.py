import numpy as np
from collections import OrderedDict

import pybullet as p
from gibson2.core.physics.interactive_objects import VisualMarker

from gibson2.envs.kitchen.camera import Camera
import gibson2.envs.kitchen.env_utils as EU
import gibson2.external.pybullet_tools.utils as PBU
import gibson2.envs.kitchen.plan_utils as PU
import gibson2.envs.kitchen.skills as skills
from gibson2.envs.kitchen.objects import Box, Hook, Tube
from gibson2.envs.kitchen.envs import TableTop


class SimpleTool(TableTop):
    def __init__(self, **kwargs):
        kwargs["robot_base_pose"] = ([0.3, 0.3, 1.2], [0, 0, 1, 0])
        # kwargs["robot_base_pose"] = ([0.3, 0.3, 1.1], skills.ALL_ORIENTATIONS["top"])
        kwargs["gripper_joint_max"] = (0.6, 0.6)
        self.eef_x_limit = -0.4
        kwargs["eef_position_limits"] = (np.array([self.eef_x_limit, -10, -10]), np.array([10, 10, 10]))
        super(TableTop, self).__init__(**kwargs)

    def _create_fixtures(self):
        super(SimpleTool, self)._create_fixtures()

        o = Tube(color=(0.5, 0.5, 0.5, 0.5), size=(0.3, 0.08, 0.1), width=0.01, mass=1.)
        o.load()
        p.changeDynamics(o.body_id, -1, mass=10000.)
        self.fixtures.add_object("tube", o)

    def _create_objects(self):
        o = Hook(width=0.025, length1=0.5, length2=0.2, color=(0.7, 0.7, 0.7, 1))
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.)
        self.objects.add_object("tool", o)

        o = Box(color=(0.7, 0.1, 0.1, 1), size=(0.05, 0.05, 0.05))
        o.load()
        p.changeDynamics(o.body_id, -1, mass=0.5)
        EU.set_friction(o.body_id, friction=0.2)
        self.objects.add_object("cube1", o)

        o = Box(color=(0.1, 0.1, 0.7, 1), size=(0.06, 0.06, 0.04))
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1.0)
        EU.set_friction(o.body_id, friction=0.5)
        self.objects.add_object("cube2", o)

        o = Box(color=(0.1, 0.8, 0.1, 0.7), size=(0.1, 0.1, 0.02))
        o.load()
        p.changeDynamics(o.body_id, -1, mass=1000.)
        self.objects.add_object("target", o)

        # setup virtual divider
        # vm = Box(color=(0.5, 0.5, 0.5, 0.3), size=(1.0, 10, 10), mass=0)
        # vm.load()
        # vm.set_position((-0.9, 0, 0.3))
        # for o in self.objects.body_ids + self.fixtures.body_ids:
        #     EU.set_collision_between(vm.body_id, o, collision=0)
        vm = VisualMarker(visual_shape=p.GEOM_BOX,
                          rgba_color=(0.5, 0.5, 0.5, 0.3),
                          half_extents=(0.01, 0.6, 0.1),
                          # half_extents=(0.01, 10.0, 10.0),
                          initial_offset=(self.eef_x_limit, 0, 0.6)
                          )
        vm.load()

    def _reset_objects(self):
        z = PBU.stable_z(self.objects["tool"].body_id, self.fixtures["table"].body_id)
        self.objects["tool"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [0.3, 0.3], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["cube1"].body_id, self.fixtures["table"].body_id)
        self.objects["cube1"].set_position_orientation(
            PU.sample_positions_in_box([-0.6, -0.4], [0.1, 0.3], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.fixtures["tube"].body_id, self.fixtures["table"].body_id)
        self.fixtures["tube"].set_position_orientation(
            PU.sample_positions_in_box([0.2, 0.2], [-0.3, -0.3], [z, z]), PBU.unit_quat())

        z = PBU.stable_z(self.objects["cube2"].body_id, self.fixtures["tube"].body_id, surface_link=-1)
        self.objects["cube2"].set_position_orientation(
            PU.sample_positions_in_box([0.1, 0.15], [-0.3, -0.3], [z, z]), skills.ALL_ORIENTATIONS["back"])

        z = PBU.stable_z(self.objects["target"].body_id, self.fixtures["table"].body_id)
        self.objects["target"].set_position_orientation(
            PU.sample_positions_in_box([0.4, 0.4], [0.0, 0.0], [z, z]), PBU.unit_quat())


class SimpleToolAP(SimpleTool):
    def _create_skill_lib(self):
        lib_skills = (
            skills.GraspTopPos(
                name="grasp", lift_height=0.1, lift_speed=0.01, reach_distance=0.03, grasp_speed=0.1,
                params=OrderedDict(
                    grasp_pos=skills.SkillParamsContinuous(low=[-0.2, -0.03, 0.03], high=[0.3, 0.03, 0.05]),
                )
            ),
            skills.PlaceFixed(name="place", retract_distance=0.1),
            skills.MoveWithPosDiscreteOrn(
                name="hook", num_pause_steps=30, move_speed=0.02,
                orientations=OrderedDict([(k, skills.ALL_ORIENTATIONS[k]) for k in ["front"]]),
                params=OrderedDict(
                    start_pos=skills.SkillParamsContinuous(low=[0.0, 0.05, 0.02], high=[0.1, 0.15, 0.02]),
                    move_pos=skills.SkillParamsContinuous(low=[0.4, 0, 0], high=[0.4, 0, 0]),
                    start_orn=skills.SkillParamsDiscrete(size=1)
                )
            ),
            skills.MoveWithPosDiscreteOrn(
                name="poke", num_pause_steps=30, move_speed=0.02,
                orientations=OrderedDict([(k, skills.ALL_ORIENTATIONS[k]) for k in ["back"]]),
                params=OrderedDict(
                    start_pos=skills.SkillParamsContinuous(low=[0.6, 0.0, -0.01], high=[0.6, 0.0, 0.01]),
                    move_pos=skills.SkillParamsContinuous(low=[-0.45, 0, 0], high=[-0.35, 0, 0]),
                    start_orn=skills.SkillParamsDiscrete(size=1)
                )
            ),
            skills.ConditionSkill(
                name="on_target",
                precondition_fn=lambda oid: PBU.is_center_placed_on(oid, self.objects["target"].body_id),
            ),
            skills.ConditionSkill(
                name="on_cube1",
                precondition_fn=lambda oid: PBU.is_center_placed_on(oid, self.objects["cube1"].body_id),
            ),
            skills.ConditionSkill(
                name="on_cube1_on_target",
                precondition_fn=lambda oid: PBU.is_center_placed_on(oid, self.objects["cube1"].body_id) and
                                            PBU.is_center_placed_on(self.objects["cube1"].body_id,
                                                                    self.objects["target"].body_id),
            )
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills)

    def set_goal(self, task_specs):
        self._task_spec = task_specs
        self.target_object = self.objects.names[task_specs[1]]

    def _sample_task(self):
        self.target_object = np.random.choice(["cube1", "cube2"])
        # self.target_object = "cube1"
        self._task_spec = np.array([self.skill_lib.name_to_skill_index("on_target"),
                                    self.objects.names.index(self.target_object)])

    def is_success_all_tasks(self):
        conds = dict(
            cube1=PBU.is_center_placed_on(self.objects["cube1"].body_id, self.objects["target"].body_id),
            cube2=PBU.is_center_placed_on(self.objects["cube2"].body_id, self.objects["target"].body_id),
            cube1_graspable=self.objects["cube1"].get_position()[0] > self.eef_x_limit,
            cube2_graspable=not PBU.is_center_placed_on(self.objects["cube2"].body_id, self.fixtures["tube"].body_id, -1),
        )
        success = {k: conds[k] for k in conds if k.startswith(self.target_object)}
        success["task"] = success[self.target_object]
        return success

    def get_constrained_skill_param_sampler(self, skill_name, object_name, num_samples=None):
        if skill_name == "grasp" and object_name in ["cube1", "cube2"]:
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "grasp", num_samples=num_samples, grasp_pos=dict(low=[-0.03, -0.03, 0.03], high=[0.03, 0.03, 0.05]))
        else:
            return lambda: self.skill_lib.sample_serialized_skill_params(skill_name, num_samples=num_samples)

    def get_task_skeleton(self):
        if self.target_object == "cube1":
            skeleton = [(
                lambda: self.skill_lib.sample_serialized_skill_params("grasp"),
                "tool"
            ),  (
                lambda: self.skill_lib.sample_serialized_skill_params("hook"),
                "cube1"
            ), (
                lambda: self.skill_lib.sample_serialized_skill_params(
                    "grasp", grasp_pos=dict(low=[-0.03, -0.03, 0.03], high=[0.03, 0.03, 0.05])),
                "cube1"
            ), (
                lambda: self.skill_lib.sample_serialized_skill_params("place"),
                "target"
            ), (
                lambda: self.skill_lib.sample_serialized_skill_params("on_target"),
                "cube1"
            )]
        else:
            skeleton = [(
                lambda: self.skill_lib.sample_serialized_skill_params("grasp"),
                "tool"
            ), (
                lambda: self.skill_lib.sample_serialized_skill_params("poke"),
                "cube2"
            ), (
                lambda: self.skill_lib.sample_serialized_skill_params(
                    "grasp", grasp_pos=dict(low=[-0.03, -0.03, 0.03], high=[0.03, 0.03, 0.05])),
                "cube2"
            ), (
                lambda: self.skill_lib.sample_serialized_skill_params("place"),
                "target"
            ), (
                lambda: self.skill_lib.sample_serialized_skill_params("on_target"),
                "cube2"
            )]
        return skeleton


class SimpleToolHardAP(SimpleToolAP):
    def _create_skill_lib(self):
        lib_skills = (
            skills.GraspTopPos(
                name="grasp", lift_height=0.1, lift_speed=0.01, reach_distance=0.03, grasp_speed=0.1,
                params=OrderedDict(
                    grasp_pos=skills.SkillParamsContinuous(low=[-0.2, -0.05, 0.03], high=[0.3, 0.05, 0.05]),
                )
            ),
            skills.PlaceFixed(name="place", retract_distance=0.1),
            skills.MoveWithPosDiscreteOrn(
                name="hook", num_pause_steps=30, move_speed=0.02,
                orientations=OrderedDict([(k, skills.ALL_ORIENTATIONS[k]) for k in ["front"]]),
                params=OrderedDict(
                    start_pos=skills.SkillParamsContinuous(low=[0.0, 0.05, 0.02], high=[0.1, 0.15, 0.02]),
                    move_pos=skills.SkillParamsContinuous(low=[0.2, -0.1, 0], high=[0.4, 0.1, 0]),
                    start_orn=skills.SkillParamsDiscrete(size=1)
                )
            ),
            skills.MoveWithPosDiscreteOrn(
                name="poke", num_pause_steps=30, move_speed=0.02,
                orientations=OrderedDict([(k, skills.ALL_ORIENTATIONS[k]) for k in ["back"]]),
                params=OrderedDict(
                    start_pos=skills.SkillParamsContinuous(low=[0.55, -0.05, -0.01], high=[0.65, 0.05, 0.01]),
                    move_pos=skills.SkillParamsContinuous(low=[-0.45, 0, 0], high=[-0.35, 0, 0]),
                    start_orn=skills.SkillParamsDiscrete(size=1)
                )
            ),
            skills.ConditionSkill(
                name="on_target",
                precondition_fn=lambda oid: PBU.is_center_placed_on(oid, self.objects["target"].body_id),
            ),
            skills.ConditionSkill(
                name="on_cube1",
                precondition_fn=lambda oid: PBU.is_center_placed_on(oid, self.objects["cube1"].body_id),
            ),
            skills.ConditionSkill(
                name="on_cube1_on_target",
                precondition_fn=lambda oid: PBU.is_center_placed_on(oid, self.objects["cube1"].body_id) and
                                            PBU.is_center_placed_on(self.objects["cube1"].body_id,
                                                                    self.objects["target"].body_id),
            )
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills)


class ToolAP(SimpleToolAP):
    def _create_skill_lib(self):
        lib_skills = (
            skills.GraspTopPos(
                name="grasp", lift_height=0.1, lift_speed=0.01, reach_distance=0.03, grasp_speed=0.1,
                params=OrderedDict(
                    grasp_pos=skills.SkillParamsContinuous(low=[-0.3, -0.2, 0.03], high=[0.3, 0.1, 0.05]),
                    # grasp_orn=skills.SkillParamsDiscrete(size=2)
                )
            ),
            skills.PlaceFixed(name="place", retract_distance=0.1),
            skills.MoveWithPosDiscreteOrn(
                name="hook", num_pause_steps=30, move_speed=0.02,
                orientations=OrderedDict([(k, skills.ALL_ORIENTATIONS[k]) for k in ["front"]]),
                params=OrderedDict(
                    start_pos=skills.SkillParamsContinuous(low=[0.0, 0.05, 0.02], high=[0.1, 0.15, 0.02]),
                    move_pos=skills.SkillParamsContinuous(low=[0.2, -0.1, 0], high=[0.4, 0.1, 0]),
                    start_orn=skills.SkillParamsDiscrete(size=1)
                )
            ),
            skills.MoveWithPosDiscreteOrn(
                name="poke", num_pause_steps=30, move_speed=0.02,
                orientations=OrderedDict([(k, skills.ALL_ORIENTATIONS[k]) for k in ["back"]]),
                params=OrderedDict(
                    start_pos=skills.SkillParamsContinuous(low=[0.55, -0.05, -0.01], high=[0.65, 0.05, 0.01]),
                    move_pos=skills.SkillParamsContinuous(low=[-0.45, 0, 0], high=[-0.35, 0, 0]),
                    start_orn=skills.SkillParamsDiscrete(size=1)
                ),
                joint_resolutions=(0.05, 0.05, 0.05, np.pi / 32, np.pi / 32, np.pi / 32)
            ),
            skills.ConditionSkill(
                name="on_target",
                precondition_fn=lambda oid: PBU.is_center_placed_on(oid, self.objects["target"].body_id),
            ),
            skills.ConditionSkill(
                name="on_cube1",
                precondition_fn=lambda oid: PBU.is_center_placed_on(oid, self.objects["cube1"].body_id),
            ),
            skills.ConditionSkill(
                name="on_cube1_on_target",
                precondition_fn=lambda oid: PBU.is_center_placed_on(oid, self.objects["cube1"].body_id) and
                                            PBU.is_center_placed_on(self.objects["cube1"].body_id,
                                                                    self.objects["target"].body_id),
            )
        )
        self.skill_lib = skills.SkillLibrary(self, self.planner, obstacles=self.obstacles, skills=lib_skills)
        # PBU.draw_aabb(aabb=[[-0.1, 0.1, 0.7], [0.5, 0.4, 0.7]])


class VizToolAP(ToolAP):
    # def __init__(self, **kwargs):
    #     kwargs["robot_base_pose"] = ([-0.3, 0.3, 1.2], [0, 0, 1, 0])
    #     # kwargs["robot_base_pose"] = ([0.3, 0.3, 1.1], skills.ALL_ORIENTATIONS["top"])
    #     kwargs["gripper_joint_max"] = (0.6, 0.6)
    #     self.eef_x_limit = -0.4
    #     kwargs["eef_position_limits"] = (np.array([self.eef_x_limit, -10, -10]), np.array([10, 10, 10]))
    #     super(TableTop, self).__init__(**kwargs)
    #
    # def _create_sensors(self):
    #     PBU.set_camera(0, -90, 0.8, (0.2, 0.3, 1.0))
    #     self.camera = Camera(
    #         height=self._camera_width,
    #         width=self._camera_height,
    #         fov=60,
    #         near=0.01,
    #         far=10.,
    #         renderer=p.ER_TINY_RENDERER
    #     )
    #     self.camera.set_pose_ypr((0.0, -0.3, 1.0), distance=0.8, yaw=0, pitch=-45)

    def get_constrained_skill_param_sampler(self, skill_name, object_name, num_samples=None):
        if skill_name == "grasp" and object_name == "tool":
            def sample_grasp_pos(ns):
                sx = np.linspace(-0.3, 0.3, num=60)
                sy = np.linspace(-0.2, 0.1, num=30)
                sz = np.array([0.035])
                samples = np.concatenate(np.meshgrid(sx, sy, sz), -1).reshape((-1, 3))
                assert ns <= samples.shape[0]
                return samples[:ns]
            return lambda: self.skill_lib.sample_serialized_skill_params(
                "grasp", num_samples=num_samples, grasp_pos=dict(sampler_fn=sample_grasp_pos)
            )
        elif skill_name == "grasp" and object_name in ["cube1", "cube2"]:
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "grasp", num_samples=num_samples, grasp_pos=dict(low=[-0.0, -0.0, 0.03], high=[0.0, 0.0, 0.05]))
        else:
            return lambda: self.skill_lib.sample_serialized_skill_params(skill_name, num_samples=num_samples)


class SimpleToolStackAP(SimpleToolAP):
    @property
    def black_listed_skills(self):
        return np.array([
            (self.skill_lib.name_to_skill_index("on_target"), self.objects.names.index("cube1")),
            (self.skill_lib.name_to_skill_index("on_target"), self.objects.names.index("cube2"))
        ])

    def _sample_task(self):
        self.target_object = "cube2"
        self._task_spec = np.array([self.skill_lib.name_to_skill_index("on_cube1_on_target"),
                                    self.objects.names.index(self.target_object)])

    def is_success_all_tasks(self):
        conds = dict(
            cube2_on_cube1=PBU.is_center_placed_on(self.objects["cube2"].body_id, self.objects["cube1"].body_id),
            cube1_on_target=PBU.is_center_placed_on(self.objects["cube1"].body_id, self.objects["target"].body_id),
            cube1_graspable=self.objects["cube1"].get_position()[0] > self.eef_x_limit,
            cube2_graspable=not PBU.is_center_placed_on(self.objects["cube2"].body_id, self.fixtures["tube"].body_id, -1),
        )
        success = conds
        success["task"] = success["cube2_on_cube1"] and success["cube1_on_target"]
        return success

    def get_task_skeleton(self):
        skeleton = [(
            lambda: self.skill_lib.sample_serialized_skill_params("grasp"),
            "tool"
        ), (
           lambda: self.skill_lib.sample_serialized_skill_params("hook"),
           "cube1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params(
                "grasp", grasp_pos=dict(low=[-0.03, -0.03, 0.03], high=[0.03, 0.03, 0.05])),
            "cube1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "target"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("grasp"),
            "tool"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("poke"),
            "cube2"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params(
                "grasp", grasp_pos=dict(low=[-0.03, -0.03, 0.03], high=[0.03, 0.03, 0.05])),
            "cube2"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("place"),
            "cube1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("on_cube1_on_target"),
            "cube2"
        )]
        return skeleton


class ToolStackAP(ToolAP):
    @property
    def black_listed_skills(self):
        return np.array([
            (self.skill_lib.name_to_skill_index("on_target"), self.objects.names.index("cube1")),
            (self.skill_lib.name_to_skill_index("on_target"), self.objects.names.index("cube2"))
        ])

    def _sample_task(self):
        self.target_object = "cube2"
        self._task_spec = np.array([self.skill_lib.name_to_skill_index("on_cube1_on_target"),
                                    self.objects.names.index(self.target_object)])

    def is_success_all_tasks(self):
        conds = dict(
            cube2_on_cube1=PBU.is_center_placed_on(self.objects["cube2"].body_id, self.objects["cube1"].body_id),
            cube1_on_target=PBU.is_center_placed_on(self.objects["cube1"].body_id, self.objects["target"].body_id),
            cube1_graspable=self.objects["cube1"].get_position()[0] > self.eef_x_limit,
            cube2_graspable=not PBU.is_center_placed_on(self.objects["cube2"].body_id, self.fixtures["tube"].body_id, -1),
        )
        success = conds
        success["task"] = success["cube2_on_cube1"] and success["cube1_on_target"]
        return success

    def get_task_skeleton(self):
        skeleton = [(
            self.get_constrained_skill_param_sampler("grasp", "tool"),
            "tool"
        ), (
            self.get_constrained_skill_param_sampler("hook", "cube1"),
            "cube1"
        ), (
            self.get_constrained_skill_param_sampler("grasp", "cube1"),
            "cube1"
        ), (
            self.get_constrained_skill_param_sampler("place", "target"),
            "target"
        ), (
            self.get_constrained_skill_param_sampler("grasp", "tool"),
            "tool"
        ), (
            self.get_constrained_skill_param_sampler("poke", "cube2"),
            "cube2"
        ), (
            self.get_constrained_skill_param_sampler("grasp", "cube2"),
            "cube2"
        ), (
            self.get_constrained_skill_param_sampler("place", "cube1"),
            "cube1"
        ), (
            lambda: self.skill_lib.sample_serialized_skill_params("on_cube1_on_target"),
            "cube2"
        )]
        return skeleton


class ToolStackEasyAP(ToolStackAP):
    def get_constrained_skill_param_sampler(self, skill_name, object_name, num_samples=None):
        if skill_name == "grasp" and object_name == "tool":
            # return lambda: self.skill_lib.sample_serialized_skill_params("grasp", num_samples=num_samples)
            return lambda: self.skill_lib.sample_serialized_skill_params(
                "grasp", num_samples=num_samples, grasp_pos=dict(low=[-0.2, -0.1, 0.03], high=[0.3, 0.1, 0.05])
            )
        elif skill_name == "poke":
            return lambda: self.skill_lib.sample_serialized_skill_params(
                "poke", num_samples=num_samples,
                start_pos=dict(low=[0.55, -0.01, -0.01], high=[0.65, 0.01, 0.01])
            )
        elif skill_name == "grasp" and object_name in ["cube1", "cube2"]:
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "grasp", num_samples=num_samples, grasp_pos=dict(low=[0, 0, 0.04], high=[0, 0, 0.05]))
        else:
            return lambda: self.skill_lib.sample_serialized_skill_params(skill_name, num_samples=num_samples)


class ToolStackMediumAP(ToolStackAP):
    def get_constrained_skill_param_sampler(self, skill_name, object_name, num_samples=None):
        if skill_name == "poke":
            return lambda: self.skill_lib.sample_serialized_skill_params(
                "poke", num_samples=num_samples,
                start_pos=dict(low=[0.55, -0.01, -0.01], high=[0.65, 0.01, 0.01])
            )
        elif skill_name == "grasp" and object_name in ["cube1", "cube2"]:
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "grasp", num_samples=num_samples, grasp_pos=dict(low=[0, 0, 0.04], high=[0, 0, 0.05]))
        else:
            return lambda: self.skill_lib.sample_serialized_skill_params(skill_name, num_samples=num_samples)


class ToolStackMediumHardAP(ToolStackAP):
    def get_constrained_skill_param_sampler(self, skill_name, object_name, num_samples=None):
        if skill_name == "grasp" and object_name in ["cube1", "cube2"]:
            return lambda: self.skill_lib.sample_serialized_skill_params(
                    "grasp", num_samples=num_samples, grasp_pos=dict(low=[0, 0, 0.04], high=[0, 0, 0.05]))
        else:
            return lambda: self.skill_lib.sample_serialized_skill_params(skill_name, num_samples=num_samples)
