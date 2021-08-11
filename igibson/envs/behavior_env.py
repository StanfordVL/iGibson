import argparse
import datetime
import os
import time
from collections import OrderedDict

import bddl
import gym.spaces
import numpy as np
from bddl.condition_evaluation import evaluate_state

from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.checkpoint_utils import load_checkpoint
from igibson.utils.ig_logging import IGLogWriter


class BehaviorEnv(iGibsonEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 30.0,
        physics_timestep=1 / 300.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        seed=0,
        action_filter="all",
        instance_id=0,
        episode_save_dir=None,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        self.action_filter = action_filter
        self.instance_id = instance_id
        super(BehaviorEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
        )
        self.rng = np.random.default_rng(seed=seed)
        self.automatic_reset = automatic_reset
        self.reward_potential = None
        self.episode_save_dir = episode_save_dir
        if self.episode_save_dir is not None:
            os.makedirs(self.episode_save_dir, exist_ok=True)

        self.log_writer = None

        # Make sure different parallel environments will have different random seeds
        np.random.seed(os.getpid())

    def load_action_space(self):
        """
        Load action space
        """
        if self.action_filter == "navigation":
            self.action_space = gym.spaces.Box(shape=(3,), low=-1.0, high=1.0, dtype=np.float32)
        elif self.action_filter == "mobile_manipulation":
            self.action_space = gym.spaces.Box(shape=(17,), low=-1.0, high=1.0, dtype=np.float32)
        elif self.action_filter == "tabletop_manipulation":
            self.action_space = gym.spaces.Box(shape=(7,), low=-1.0, high=1.0, dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(shape=(26,), low=-1.0, high=1.0, dtype=np.float32)

    def load_behavior_task_setup(self):
        """
        Load task setup
        """
        # task
        task = self.config["task"]
        task_id = self.config["task_id"]
        scene_id = self.config["scene_id"]
        clutter = self.config["clutter"]
        online_sampling = self.config["online_sampling"]
        if online_sampling:
            scene_kwargs = {}
        else:
            scene_kwargs = {
                "urdf_file": "{}_task_{}_{}_{}_fixed_furniture".format(scene_id, task, task_id, self.instance_id),
            }
        bddl.set_backend("iGibson")
        self.task = iGBEHAVIORActivityInstance(task, task_id)
        self.task.initialize_simulator(
            simulator=self.simulator,
            scene_id=scene_id,
            load_clutter=clutter,
            scene_kwargs=scene_kwargs,
            online_sampling=online_sampling,
        )

        for _, obj in self.task.object_scope.items():
            if obj.category in ["agent", "room_floor"]:
                continue
            obj.highlight()

        self.scene = self.task.scene
        self.robots = [self.task.agent]

        self.reset_checkpoint_idx = self.config.get("reset_checkpoint_idx", -1)
        self.reset_checkpoint_dir = self.config.get("reset_checkpoint_dir", None)

    def load_task_setup(self):
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)

        self.load_behavior_task_setup()

        # Activate the robot constraints so that we don't need to feed in
        # trigger press action in the first couple frames
        self.robots[0].activate()

    def load(self):
        """
        Load environment
        """
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def load_observation_space(self):
        super(BehaviorEnv, self).load_observation_space()
        if "proprioception" in self.output:
            proprioception_dim = self.robots[0].get_proprioception_dim()
            self.observation_space.spaces["proprioception"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(proprioception_dim,)
            )
            self.observation_space = gym.spaces.Dict(self.observation_space.spaces)

    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1

        if self.action_filter == "navigation":
            new_action = np.zeros((28,))
            new_action[:2] = action[:2]
            new_action[5] = action[2]
        elif self.action_filter == "mobile_manipulation":
            new_action = np.zeros((28,))
            # body x,y,yaw
            new_action[:2] = action[:2]
            new_action[5] = action[2]
            # left hand 7d
            new_action[12:19] = action[3:10]
            # right hand 7d
            new_action[20:27] = action[10:17]
        elif self.action_filter == "tabletop_manipulation":
            # only using right hand
            new_action = np.zeros((28,))
            new_action[20:27] = action[:7]
        else:
            # all action dims except hand reset
            new_action = np.zeros((28,))
            new_action[:19] = action[:19]
            new_action[20:27] = action[19:]

        # The original action space for BehaviorRobot is too wide for random exploration
        new_action *= 0.05

        self.robots[0].update(new_action)
        if self.log_writer is not None:
            self.log_writer.process_frame()
        self.simulator.step()

        state = self.get_state()
        info = {}
        done, satisfied_predicates = self.task.check_success()
        # Compute the initial reward potential here instead of during reset
        # because if an intermediate checkpoint is loaded, we need step the
        # simulator before calling task.check_success
        if self.current_step == 1:
            self.reward_potential = self.get_potential(satisfied_predicates)

        if self.current_step >= self.config["max_step"]:
            done = True
        reward, info = self.get_reward(satisfied_predicates)

        self.populate_info(info)

        if done and self.automatic_reset:
            info["last_observation"] = state
            state = self.reset()

        return state, reward, done, info

    def get_potential(self, satisfied_predicates):
        potential = 0.0

        # Evaluate the first ground goal state option as the potential
        _, satisfied_predicates = evaluate_state(self.task.ground_goal_state_options[0])
        success_score = len(satisfied_predicates["satisfied"]) / (
            len(satisfied_predicates["satisfied"]) + len(satisfied_predicates["unsatisfied"])
        )
        predicate_potential = success_score
        potential += predicate_potential

        return potential

    def get_reward(self, satisfied_predicates):
        new_potential = self.get_potential(satisfied_predicates)
        reward = new_potential - self.reward_potential
        self.reward_potential = new_potential
        return reward, {"satisfied_predicates": satisfied_predicates}

    def get_state(self, collision_links=[]):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if "task_obs" in self.output:
            state["task_obs"] = self.task.get_task_obs(self)
        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
        if "scan_occ" in self.sensors:
            scan_obs = self.sensors["scan_occ"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "bump" in self.sensors:
            state["bump"] = self.sensors["bump"].get_obs(self)

        if "proprioception" in self.output:
            state["proprioception"] = np.array(self.robots[0].get_proprioception())

        return state

    def reset_scene_and_agent(self):
        if self.reset_checkpoint_dir is not None and self.reset_checkpoint_idx != -1:
            load_checkpoint(self.simulator, self.reset_checkpoint_dir, self.reset_checkpoint_idx)
        else:
            self.task.reset_scene(snapshot_id=self.task.initial_state)
        # set the constraints to the current poses
        self.robots[0].update(np.zeros(28))

    def reset(self, resample_objects=False):
        """
        Reset episode
        """
        # if self.log_writer is not None, save previous episode
        if self.log_writer is not None:
            self.log_writer.end_log_session()
            del self.log_writer
            self.log_writer = None

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task = self.config["task"]
        task_id = self.config["task_id"]
        scene = self.config["scene_id"]
        if self.episode_save_dir:
            vr_log_path = os.path.join(
                self.episode_save_dir,
                "{}_{}_{}_{}_{}.hdf5".format(task, task_id, scene, timestamp, self.current_episode),
            )
            self.log_writer = IGLogWriter(
                self.simulator,
                frames_before_write=200,
                log_filepath=vr_log_path,
                task=self.task,
                store_vr=False,
                vr_robot=self.robots[0],
                filter_objects=True,
            )
            self.log_writer.set_up_data_storage()

        # if self.episode_save_dir not set, self.log_writer will be None

        self.robots[0].robot_specific_reset()

        self.reset_scene_and_agent()

        self.simulator.sync(force_sync=True)
        state = self.get_state()
        self.reset_variables()

        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="igibson/examples/configs/behavior.yaml",
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "gui", "iggui", "pbgui"],
        default="gui",
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

    env = BehaviorEnv(
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
        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
