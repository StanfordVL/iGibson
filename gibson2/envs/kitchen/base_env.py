import numpy as np
import os
import time
from copy import deepcopy

import pybullet as p
import gibson2

from gibson2.envs.kitchen.transform_utils import quat2col

from gibson2.envs.kitchen.camera import Camera, crop_pad_resize, get_bbox2d_from_segmentation, get_masks_from_segmentation
from gibson2.envs.kitchen.robots import Arm, ConstraintActuatedRobot, PlannerRobot, Robot, Gripper
import gibson2.envs.kitchen.env_utils as EU
import gibson2.external.pybullet_tools.utils as PBU
import gibson2.envs.kitchen.plan_utils as PU


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
            obs_crop=False,
            obs_crop_size=24,
            use_skills=False,
            gripper_use_magic_grasp=True,
            gripper_joint_max=(1.0, 1.0),
            plan_joint_limits=None,
            eef_position_limits=None
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
        self._obs_crop = obs_crop
        self.obs_crop_size = obs_crop_size
        self._gripper_use_magic_grasp = gripper_use_magic_grasp
        self._gripper_joint_max = gripper_joint_max
        self._plan_joint_limits = plan_joint_limits
        self._eef_position_limits = eef_position_limits

        self.objects = EU.ObjectBank()
        self.interactive_objects = EU.ObjectBank()
        self.fixtures = EU.ObjectBank()
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
        raise NotImplementedError

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
            finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip"),
            use_magic_grasp=self._gripper_use_magic_grasp,
            joint_max=self._gripper_joint_max
        )
        gripper.env = self
        gripper.load(os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper.urdf'))
        robot = ConstraintActuatedRobot(
            eef_link_name="eef_link", init_base_pose=self._robot_base_pose, gripper=gripper)

        for l in PBU.get_all_links(body=robot.body_id):
            p.changeDynamics(robot.body_id, l, mass=0.1)
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
            joint_limits=self._plan_joint_limits,
            eef_position_limit=self._eef_position_limits
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
        raise NotImplementedError

    def _create_env_extras(self):
        pass

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
        p.stepSimulation()
        self.robot.reset_base_position_orientation(*self._robot_base_pose)
        self.robot.reset()
        self._reset_objects()
        self._sample_task()
        if self.skill_lib is not None:
            self.skill_lib.reset()
        for o in self.interactive_objects.object_list:
            o.reset()
        # p.stepSimulation()

    def reset_to(self, serialized_world_state, return_obs=True):
        exclude = []
        if self.planner is not None:
            exclude.append(self.planner.body_id)
        state = PBU.WorldSaver(exclude_body_ids=exclude)
        state.deserialize(serialized_world_state)
        state.restore()
        p.stepSimulation()
        if return_obs:
            return self.get_observation()

    @property
    def serialized_world_state(self):
        exclude = []
        if self.planner is not None:
            exclude.append(self.planner.body_id)
        return PBU.WorldSaver(exclude_body_ids=exclude).serialize()

    def step(self, action, sleep_per_sim_step=0.0, return_obs=False):
        assert len(action) == self.action_dimension
        action = action.copy()
        gri = action[-1]
        pos, orn = EU.action_to_delta_pose_euler(action[:6], max_dpos=self.MAX_DPOS, max_drot=self.MAX_DROT)
        # pos, orn = action_to_delta_pose_axis_vector(action[:6], max_dpos=self.MAX_DPOS, max_drot=self.MAX_DROT)
        # print(np.linalg.norm(pos), np.linalg.norm(T.euler_from_quaternion(orn)))
        self.robot.set_relative_eef_position_orientation(pos, orn)
        if gri > 0:
            self.robot.gripper.grasp()
        else:
            self.robot.gripper.ungrasp()

        for o in self.interactive_objects.object_list:
            o.step(self.objects.object_list, self.robot.gripper)

        for _ in range(self._num_sim_per_step):
            p.stepSimulation()
            time.sleep(sleep_per_sim_step)

        if return_obs:
            return self.get_observation(), self.get_reward(), self.is_done(), {}

    def get_reward(self):
        return float(self.is_success())

    def render(self, mode, width=None, height=None):
        """Render"""
        rgba, _, _ = self.camera.capture_raw(width=width, height=height)
        return rgba[:, :, :3]

    def _get_pixel_observation(self, camera):
        obs = {}
        rgb, depth, seg_obj, seg_link = camera.capture_frame()
        if self._obs_image:
            obs["image_" + camera.name] = rgb
            if self._obs_crop:
                bbox = get_bbox2d_from_segmentation(seg_obj, self.objects.body_ids)
                obs["image_crops_" + camera.name] = crop_pad_resize(
                    rgb, bbox=bbox[:, 1:], target_size=self.obs_crop_size, expand_ratio=1.1)
                obs["image_crops_flat_" + camera.name] = obs["image_crops"].reshape((-1, self.obs_crop_size, 3))
        if self._obs_depth:
            obs["depth_" + camera.name] = depth[:, :, None]
        if self._obs_segmentation:
            obs["segmentation_objects_" + camera.name] = \
                get_masks_from_segmentation(seg_obj, self.objects.body_ids)[:, :, :, None]
            # obs["segmentation_links_" + camera.name] = seg_link[:, :, :, None]
        return obs

    def _get_state_observation(self):
        # get object info
        gpose = self.robot.get_eef_position_orientation()
        object_states = self.objects.serialize()
        rel_link_poses = np.zeros_like(object_states["link_poses"])
        for i, lpose in enumerate(object_states["link_poses"]):
            rel_link_poses[i] = EU.pose_to_array(PBU.multiply((lpose[:3], lpose[3:]), PBU.invert(gpose)))
        return {
            "link_poses": object_states["link_poses"],
            "link_relative_poses": rel_link_poses,
            "link_positions": object_states["link_poses"][:, :3],
            "link_relative_positions": rel_link_poses[:, :3],
            "categories": object_states["categories"]
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

    def _get_feature_observation(self):
        return {}

    def get_observation(self):
        obs = {}
        obs.update(self._get_proprio_observation())
        obs.update(self._get_state_observation())
        obs.update(self._get_feature_observation())
        obs["task_specs"] = self.task_spec
        if self._obs_image or self._obs_depth or self._obs_segmentation:
            obs.update(self._get_pixel_observation(self.camera))
        return obs

    @property
    def obstacles(self):
        return self.objects.body_ids + self.fixtures.body_ids

    def set_goal(self, task_specs):
        """Set env target with external specification"""
        self._task_spec = np.array(task_specs)

    def is_done(self):
        """Check if the agent is done (not necessarily successful)."""
        return self.is_success()

    def is_success(self):
        return self.is_success_all_tasks()["task"]

    def is_success_all_tasks(self):
        return {"task": False}

    @property
    def name(self):
        """Environment name"""
        return self.__class__.__name__

    def execute_skeleton(self, skeleton, stop_on_exception=False, noise=None):
        buffer = PU.Buffer()
        exception = None
        for skill_step, (param_func, object_name) in enumerate(skeleton):
            skill_param = param_func()
            print(self.skill_lib.skill_params_to_string(skill_param, self.objects.name_to_body_id(object_name)))
            traj, exec_info = PU.execute_skill(
                self, self.skill_lib, skill_param,
                target_object_id=self.objects.name_to_body_id(object_name),
                skill_step=skill_step,
                noise=noise
            )
            buffer.append(**traj)
            if exec_info["exception"] is not None:
                exception = exec_info["exception"]
                print(exception)
                if stop_on_exception:
                    return buffer.aggregate(), exception

        if exception is None:  # goal skill is run
            assert self.is_success()
        return buffer.aggregate(), exception


class EnvSkillWrapper(object):
    def __init__(self, env):
        self.env = env
        self.skill_lib = env.skill_lib
        self.debug_viz_handles = []

    @property
    def action_dimension(self):
        return self.skill_lib.action_dimension + len(self.env.objects)

    def parse_action_array_to_skill_info(self, action):
        assert action.shape[0] == self.action_dimension
        skill_dict = dict()
        skill_params = action[:self.skill_lib.action_dimension]
        skill_object_index = int(np.argmax(action[self.skill_lib.action_dimension:]))
        skill_object_index_arr = np.zeros(len(self.env.objects))
        skill_object_index_arr[skill_object_index] = 1
        skill_dict["skill_params"] = skill_params
        skill_dict["skill_param_masks"] = self.skill_lib.get_serialized_skill_param_mask(skill_params)
        skill_dict["skill_object_index"] = skill_object_index_arr
        skill_param_dict = self.skill_lib.deserialize_skill_params(skill_params)
        skill_param_dict.update(self.skill_lib.get_skill_param_dict_metadata(skill_param_dict))
        skill_dict["skill_param_dict"] = skill_param_dict
        return skill_dict

    def _clear_viz(self):
        PBU.remove_handles(self.debug_viz_handles)
        self.debug_viz_handles = []

    def reset(self):
        self._clear_viz()
        self.env.reset()

    def random_action_generator(self, horizon=None):
        """generate random valid actions"""
        step_i = 0
        while horizon is None or step_i < horizon:
            step_i += 1
            skill_index, object_index = self.sample_skill()
            skill_index_param = self.sample_skill_param(skill_index_arr=skill_index)
            yield np.concatenate([skill_index_param, object_index])

    def random_action_dict_generator(self, horizon=None):
        step_i = 0
        while horizon is None or step_i < horizon:
            step_i += 1
            skill_index, object_index = self.sample_skill()
            skill_index_param = self.sample_skill_param(skill_index_arr=skill_index)
            yield dict(skill_index=skill_index_param[:len(self.skill_lib)],
                       skill_param=skill_index_param[len(self.skill_lib):],
                       skill_object_index=object_index)

    def skeleton_action_generator(self):
        """generate actions with ground truth skill skeleton and sampled parameters."""
        for param_func, object_name in self.env.get_task_skeleton():
            skill_param = param_func()
            object_index = self.env.objects.names.index(object_name)
            object_index = self.env.objects.object_index_to_array(object_index)
            yield np.concatenate([skill_param, object_index])

    def skeleton_action_dict_generator(self):
        for param_func, object_name in self.env.get_task_skeleton():
            skill_index_param = param_func()
            object_index = self.env.objects.names.index(object_name)
            object_index = self.env.objects.object_index_to_array(object_index)
            yield dict(skill_index=skill_index_param[:len(self.skill_lib)],
                       skill_param=skill_index_param[len(self.skill_lib):],
                       skill_object_index=object_index)

    def sample_skill(self):
        """Sample skill_index, object_index pairs"""
        skill_index_arr = self.skill_lib.skill_index_to_array(np.random.randint(0, len(self.skill_lib)))
        object_index_arr = self.env.objects.object_index_to_array(np.random.randint(0, len(self.env.objects)))
        return skill_index_arr, object_index_arr

    def get_goal_skill(self):
        skill_index_arr = self.skill_lib.skill_index_to_array(self.env.task_spec[0])
        object_index_arr = self.env.objects.object_index_to_array(self.env.task_spec[1])
        return skill_index_arr, object_index_arr

    def sample_skill_param(self, skill_index_arr):
        assert len(skill_index_arr) == len(self.skill_lib)
        skill_name = self.skill_lib.skill_names[int(np.argmax(skill_index_arr))]
        return self.skill_lib.sample_serialized_skill_params(skill_name)

    def action_to_string(self, actions):
        skill_params = actions[:self.skill_lib.action_dimension]
        object_index = int(np.argmax(actions[self.skill_lib.action_dimension:]))
        object_id = self.env.objects.body_ids[object_index]
        return self.skill_lib.skill_params_to_string(skill_params, object_id)

    def visualize_action(self, actions, color=(1, 0, 0, 1), length=0.1):
        skill_params = actions[:self.skill_lib.action_dimension]
        object_index = int(np.argmax(actions[self.skill_lib.action_dimension:]))
        object_id = self.env.objects.body_ids[object_index]
        eef_pose = self.skill_lib.skill_params_to_pose(skill_params, object_id)
        if eef_pose is not None:
            self.debug_viz_handles.extend(PBU.draw_pose_axis(eef_pose, axis=(-1, 0, 0), length=length, color=color))

    def step(self, actions, sleep_per_sim_step=0.0, return_obs=True, step_callback=None):
        assert actions.shape[0] == self.action_dimension
        if isinstance(actions, np.ndarray):
            skill_params = actions[:self.skill_lib.action_dimension]
            object_index = int(np.argmax(actions[self.skill_lib.action_dimension:]))
        elif isinstance(actions, dict):
            actions = deepcopy(actions)
            object_index = int(np.argmax(actions.pop("skill_object_index")))
            skill_params = actions
        else:
            raise TypeError("Wrong actions data type: expecting np.ndarray or dict, got {}".format(type(actions)))
        object_id = self.env.objects.body_ids[object_index]
        path = self.skill_lib.plan(params=skill_params, target_object_id=object_id)

        PU.execute_planned_path(
            self.env,
            path,
            sleep_per_sim_step=sleep_per_sim_step,
            store_full_trajectory=False,
            step_callback=step_callback
        )
        self._clear_viz()
        if return_obs:
            return self.get_observation(), self.get_reward(), self.is_done(), {}
        else:
            return None, self.get_reward(), self.is_done(), {}

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
