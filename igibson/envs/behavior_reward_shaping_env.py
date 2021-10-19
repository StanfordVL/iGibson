import argparse
import time
from collections import OrderedDict, deque

import gym
import numpy
import numpy as np
import pybullet as p
from IPython import embed

from igibson.envs.behavior_env import BehaviorEnv
from igibson.utils.utils import l2_distance


class BehaviorRewardShapingEnv(BehaviorEnv):
    """
    BehaviorRewardShapingEnv (OpenAI Gym interface)
    """

    def load_observation_space(self):

        super(BehaviorRewardShapingEnv, self).load_observation_space()
        self.frames = deque([], maxlen=3)
        self.task_obs_dim = 27
        self.observation_space.spaces["task_obs"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.task_obs_dim,), dtype=np.float32
        )
        self.observation_space.spaces["action"] = self.action_space

        self.observation_space.spaces["discount"] = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.observation_space.spaces["rgb"] = gym.spaces.Box(
            shape=(9, self.image_height, self.image_width), low=0.0, high=1.0, dtype=np.float32
        )

    def get_state(self, collision_links=[], reset=False):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = super(BehaviorRewardShapingEnv, self).get_state(collision_links)
        task_obs = np.zeros(self.task_obs_dim)
        state_dict = OrderedDict()
        state_dict["robot_pos"] = np.array(self.robots[0].get_position())
        state_dict["robot_orn"] = np.array(self.robots[0].get_rpy())
        for hand in ["left_hand", "right_hand"]:
            state_dict["robot_{}_pos".format(hand)] = np.array(self.robots[0].parts[hand].get_position())
            state_dict["robot_{}_orn".format(hand)] = np.array(
                p.getEulerFromQuaternion(self.robots[0].parts[hand].get_orientation())
            )

        state_dict["obj_valid"] = 0.0
        state_dict["obj_pos"] = np.zeros(3)
        state_dict["obj_orn"] = np.zeros(3)
        state_dict["obj_in_gripper_0"] = 0.0
        state_dict["obj_in_gripper_1"] = 0.0
        if self.goal_target_id != -1:
            state_dict["obj_valid"] = 1.0
            v = self.goal_relevant_objs[self.goal_target_id]
            state_dict["obj_pos"] = np.array(v.get_position())
            state_dict["obj_orn"] = np.array(p.getEulerFromQuaternion(v.get_orientation()))
            grasping_objects = self.robots[0].is_grasping(v.get_body_id())
            for grasp_idx, grasping in enumerate(grasping_objects):
                state_dict["obj_in_gripper_{}".format(grasp_idx)] = float(grasping)

        state_list = []
        for k, v in state_dict.items():
            if isinstance(v, list):
                state_list.extend(v)
            elif isinstance(v, tuple):
                state_list.extend(list(v))
            elif isinstance(v, np.ndarray):
                state_list.extend(list(v))
            elif isinstance(v, (float, int)):
                state_list.append(v)
            else:
                raise ValueError("cannot serialize task obs")

        assert len(state_list) == len(task_obs)
        task_obs = np.array(state_list)
        state["task_obs"] = task_obs

        state["rgb"] *= 255
        state["rgb"] = state["rgb"].astype(np.uint8)
        state["rgb"] = state["rgb"].transpose(2, 0, 1).copy()
        if reset:
            for i in range(3):
                self.frames.append(state['rgb'])
        else:
            self.frames.append(state["rgb"])
        state["rgb"] = np.concatenate(list(self.frames), axis=0)

        state["discount"] = np.array([1], dtype=np.float32)

        return state

    def load_behavior_task_setup(self):
        """
        Load task setup
        """
        super(BehaviorRewardShapingEnv, self).load_behavior_task_setup()
        self.ground_goal_conditions = self.task.ground_goal_state_options[0]
        self.goal_relevant_objs = []
        self.goal_positions = []
        self.goal_has_picked = []
        self.goal_target_id = -1
        for goal_condition in self.ground_goal_conditions:
            assert goal_condition.children[0].STATE_NAME in ["ontop", "inside", "onfloor", "under"]
            goal_condition.children[0].kwargs["use_ray_casting_method"] = True
            sampling_success = goal_condition.children[0].sample(True)
            assert sampling_success
            goal_relevant_obj = self.task.object_scope[goal_condition.children[0].input1]
            self.goal_relevant_objs.append(goal_relevant_obj)
            self.goal_positions.append(goal_relevant_obj.get_position())
            self.goal_has_picked.append(False)
        self.reward_picked_weight = 5.0 / len(self.goal_has_picked)
        self.reward_predicate_weight = 10.0
        self.reward_distance_weight = 1.0

    def get_shaped_reward(self, satisfied_predicates):
        shaped_reward = 0.0
        if len(satisfied_predicates["unsatisfied"]) == 0:
            return shaped_reward
        target_id = min(satisfied_predicates["unsatisfied"])
        has_picked = []
        for goal_relevant_obj in self.goal_relevant_objs:
            has_picked.append(
                goal_relevant_obj.get_body_id() == self.robots[0].parts["left_hand"].object_in_hand
                or goal_relevant_obj.get_body_id() == self.robots[0].parts["right_hand"].object_in_hand
            )

        if not has_picked[target_id]:
            distance_potential = min(
                l2_distance(
                    self.goal_relevant_objs[target_id].get_position(), self.robots[0].parts["left_hand"].get_position()
                ),
                l2_distance(
                    self.goal_relevant_objs[target_id].get_position(), self.robots[0].parts["right_hand"].get_position()
                ),
            )
        else:
            distance_potential = l2_distance(
                self.goal_relevant_objs[target_id].get_position(), self.goal_positions[target_id]
            )

        if target_id == self.goal_target_id:
            if not self.goal_has_picked[target_id] and has_picked[target_id]:
                # picked target object at this timestep
                shaped_reward += self.reward_picked_weight
                # print("picked")
            elif self.goal_has_picked[target_id] and not has_picked[target_id]:
                # dropped target object at this timestep
                shaped_reward -= self.reward_picked_weight
                # print("placed")
            else:
                if self.goal_distance_potential is not None:
                    shaped_reward += (self.goal_distance_potential - distance_potential) * self.reward_distance_weight
                    # print("distance")

        self.goal_target_id = target_id
        self.goal_has_picked = has_picked
        self.goal_distance_potential = distance_potential

        return shaped_reward

    def get_reward(self, satisfied_predicates):
        reward, info = super(BehaviorRewardShapingEnv, self).get_reward(satisfied_predicates)
        # if reward > 0:
        #     print("predicate success")
        shaped_reward = self.get_shaped_reward(satisfied_predicates)
        reward = reward * self.reward_predicate_weight + shaped_reward
        # print("reward", reward)
        return reward, info

    def reset_variables(self):
        super(BehaviorRewardShapingEnv, self).reset_variables()
        self.goal_has_picked = [False] * len(self.goal_has_picked)
        self.goal_target_id = -1
        self.goal_distance_potential = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="igibson/examples/configs/behavior_onboard_sensing_fetch.yaml",
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "gui", "iggui", "pbgui"],
        default="iggui",
        help="which mode for simulation (default: headless)",
    )
    parser.add_argument(
        "--action_filter",
        "-af",
        choices=["navigation", "tabletop_manipulation", "mobile_manipulation", "all"],
        default="mobile_manipulation",
        help="which action filter",
    )
    args = parser.parse_args()

    env = BehaviorRewardShapingEnv(
        config_file=args.config,
        mode=args.mode,
        action_timestep=1.0 / 30.0,
        physics_timestep=1.0 / 300.0,
        action_filter=args.action_filter,
        episode_save_dir=None,
    )
    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        embed()
        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
