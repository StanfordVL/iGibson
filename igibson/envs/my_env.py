import argparse
import logging
import os
import time
from collections import OrderedDict

import gym
# https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py
from gym import spaces
import copy
import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat

import bddl

import igibson
from igibson import object_states
from igibson.envs.env_base import BaseEnv
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.tasks.behavior_task import BehaviorTask
from igibson.tasks.dummy_task import DummyTask
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.utils import quatToXYZW
from igibson.utils.constants import ViewerMode
from igibson.object_states.pose import Pose
from igibson.utils.transform_utils import quat2mat, quat2axisangle, mat2euler


from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper

log = logging.getLogger(__name__)

skill_object_offset_params = {
    0:  # skill id
        {'printer.n.03_1': [-0.7, 0, 0, 0],  # dx, dy, dz, target_yaw
         'table.n.02_1': [0, -0.6, 0, 0.5 * np.pi], }
    ,
    1:
        {'printer.n.03_1': [-0.2, 0.0, 0.2],  # dx, dy, dz
         'table.n.02_1': [0, 0, 0.5], }
    ,
    2:
        {'printer.n.03_1': [-0.3, -0.25, 0.23],  # dx, dy, dz
         }
    ,
}

action_list = [
    [0, 'printer.n.03_1'],  # skill id, target_obj
    [1, 'printer.n.03_1'],
    [0, 'table.n.02_1'],
    [1, 'table.n.02_1'],
    # [2, 'printer.n.03_1'],
]

class SkillEnv(gym.Env):
    """
    Skill RL Environment (OpenAI Gym interface).
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        rendering_settings=None,
        vr_settings=None,
        device_idx=0,
        automatic_reset=False,
        use_pb_gui=False,
        num_discrete_action=5,
        print_log=True,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, headless_tensor, gui_interactive, gui_non_interactive, vr
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        :param vr_settings: vr_settings to override the default one
        :param device_idx: which GPU to run the simulation and rendering on
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        self.print_log = print_log
        self.automatic_reset = automatic_reset
        full_observability_2d_planning = True
        collision_with_pb_2d_planning = False
        self.env = iGibsonEnv(
            config_file=config_file,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
        )
        # 'occupancy_grid' modality is required as input
        self.planner = MotionPlanningWrapper(
            self.env,
            optimize_iter=10,
            full_observability_2d_planning=full_observability_2d_planning,
            collision_with_pb_2d_planning=collision_with_pb_2d_planning,
            visualize_2d_planning=not (mode == 'headless'),
            visualize_2d_result=not (mode == 'headless'),
            fine_motion_plan=False,
            print_log=print_log,
        )
        self.current_step = self.env.current_step

        self.env.task.initial_state = self.env.task.save_scene(self.env)
        if mode == 'gui_interactive':
            # Set viewer camera and directly set mode to planning mode
            self.env.simulator.viewer.initial_pos = [1.5, -1.0, 1.9]
            self.env.simulator.viewer.initial_view_direction = [-0.5, 0.0, -1.0]
            self.env.simulator.viewer.reset_viewer()
            self.env.simulator.viewer.mode = ViewerMode.PLANNING

        self.task_obj_list = self.env.task.object_scope

        self.observation_space = self.env.observation_space
        self.sensors = self.env.sensors
        self.action_space = spaces.Discrete(num_discrete_action)

    def step(self, action_index):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: 0: move_to, 1: pick 2: place
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        action = action_list[action_index]
        params = skill_object_offset_params[action[0]][action[1]]
        self.env.current_step += 1
        self.current_step = self.env.current_step

        hit_normal = (0.0, 0.0, 1.0)  # default hit normal

        if action[0] == 0: # base move to
            obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

            # process the offset from object frame to world frame
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])

            # acquire the base direction
            euler = mat2euler(mat)
            target_yaw = euler[-1] + params[3]

            plan = self.planner.plan_base_motion([obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw])
            if plan is not None and len(plan) > 0:
                self.planner.dry_run_base_plan(plan)

            if self.print_log:
                print('move')

        elif action[0] == 1: # arm pick and place
            obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

            # process the offset from object frame to world frame
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])

            pick_place_pos = copy.deepcopy(obj_pos)
            pick_place_pos[0] += vector[0]
            pick_place_pos[1] += vector[1]
            pick_place_pos[2] += vector[2]

            plan = self.planner.plan_arm_pick_place(pick_place_pos)
            self.planner.execute_arm_pick_place(plan, pick_place_pos, -np.array(hit_normal))

            if self.print_log:
                print('pick and place')

        elif action[0] == 2:  #arm toggle
            obj_pos = self.task_obj_list[action[1]].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[action[1]].states[Pose].get_value()[1]

            # process the offset from object frame to world frame
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])

            toggle_pos = copy.deepcopy(obj_pos)
            toggle_pos[0] += vector[0]
            toggle_pos[1] += vector[1]
            toggle_pos[2] += vector[2]

            plan = self.planner.plan_arm_toggle(toggle_pos, -np.array(hit_normal))
            self.planner.execute_arm_toggle(plan, toggle_pos, -np.array(hit_normal))

            if self.print_log:
                print('toggle')

        # elif action[0] == 2:
        #     plan = self.planner.plan_arm_push(hit_pos, -np.array(hit_normal))
        #     self.planner.execute_arm_push(plan, hit_pos, -np.array(hit_normal))
        # elif action[0] == 3:
        #     plan = self.planner.plan_arm_pull(hit_pos, np.array(hit_normal))
        #     self.planner.execute_arm_pull(plan, hit_pos, np.array(hit_normal))
        # elif action[0] == 4:
        #     plan = self.planner.plan_arm_toggle(hit_pos, -np.array(hit_normal))
        #     self.planner.execute_arm_toggle(plan, hit_pos, -np.array(hit_normal))
        ## ------------------- Prev
        # if action is not None:
        #     self.env.robots[0].apply_action(action)
        # collision_links = self.env.run_simulation()

        self.env.simulator.sync()

        state = self.env.get_state()
        info = {}
        reward, info = self.env.task.get_reward(self.env)
        done, info = self.env.task.get_termination(self.env)

        print(reward)

        if done and self.env.automatic_reset:
            info["last_observation"] = state
            state = self.env.reset()

        return state, reward, done, info

    def reset(self):
        """
        Reset episode.
        """
        state = self.env.reset()
        return state
    
    def close(self):
        """
        Reset episode.
        """
        self.env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=os.path.join(igibson.configs_path, "fetch_rl.yaml"), help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default='headless', # "headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = SkillEnv(config_file=args.config, mode=args.mode, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)

    step_time_list = []
    for episode in range(1):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        for i in range(len(action_list)):  # 10 seconds
            state, reward, done, _ = env.step(i)
            print("reward: {}, done: {}".format(reward, done))
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
