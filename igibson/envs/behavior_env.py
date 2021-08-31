import argparse
import datetime
import os
import time
from collections import OrderedDict

import bddl
import gym.spaces
import numpy as np
import pybullet as p
from bddl.condition_evaluation import evaluate_state
from IPython import embed

from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.envs.igibson_env import iGibsonEnv
from igibson.object_states import HeatSourceOrSink, Stained, WaterSource
from igibson.object_states.factory import get_state_from_name
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.robots.fetch_gripper_robot import FetchGripper
from igibson.utils.checkpoint_utils import load_checkpoint
from igibson.utils.ig_logging import IGLogWriter
from igibson.utils.utils import l2_distance


class BehaviorEnv(iGibsonEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
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
        self.distance_potential = None
        self.reward_shaping_relevant_objs = self.config.get("reward_shaping_relevant_objs", None)
        self.progress_reward_objs = self.config.get("progress_reward_objs", None)
        self.predicate_reward_weight = self.config.get("predicate_reward_weight", 20.0)
        self.distance_reward_weight = self.config.get("distance_reward_weight", 20.0)
        self.pickup_reward_weight = self.config.get("pickup_reward_weight", 5.0)
        self.pickup_correct_obj = False
        self.pickup_correct_obj_prev_step = False
        self.magic_grasping_cid = None

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
        if isinstance(self.robots[0], BehaviorRobot):
            if self.action_filter == "navigation":
                self.action_space = gym.spaces.Box(shape=(3,), low=-1.0, high=1.0, dtype=np.float32)
            elif self.action_filter == "mobile_manipulation":
                self.action_space = gym.spaces.Box(shape=(17,), low=-1.0, high=1.0, dtype=np.float32)
            elif self.action_filter == "tabletop_manipulation":
                self.action_space = gym.spaces.Box(shape=(7,), low=-1.0, high=1.0, dtype=np.float32)
            elif self.action_filter == "bimanual_manipulation":
                self.action_space = gym.spaces.Box(shape=(14,), low=-1.0, high=1.0, dtype=np.float32)
            else:
                self.action_space = gym.spaces.Box(shape=(26,), low=-1.0, high=1.0, dtype=np.float32)
        elif isinstance(self.robots[0], FetchGripper):
            if self.action_filter == "navigation":
                self.action_space = gym.spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32)
            elif self.action_filter == "tabletop_manipulation":
                self.action_space = gym.spaces.Box(shape=(7,), low=-1.0, high=1.0, dtype=np.float32)
            elif self.action_filter == "magic_grasping":
                self.action_space = gym.spaces.Box(shape=(6,), low=-1.0, high=1.0, dtype=np.float32)
            else:
                self.action_space = gym.spaces.Box(shape=(11,), low=-1.0, high=1.0, dtype=np.float32)
        else:
            Exception("Only BehaviorRobot and FetchGripper are supported for behavior_env")

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
        robot_class = self.config.get("robot")
        if robot_class == "BehaviorRobot":
            robot_type = BehaviorRobot
        elif robot_class == "FetchGripper":
            robot_type = FetchGripper
        else:
            Exception("Only BehaviorRobot and FetchGripper are supported for behavior_env")

        self.task = iGBEHAVIORActivityInstance(task, task_id, robot_type=robot_type, robot_config=self.config)
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
        self.robots = self.simulator.robots

        if isinstance(self.robots[0], FetchGripper) and self.action_filter in [
            "tabletop_manipulation",
            "magic_grasping",
        ]:
            self.robot_constraint = p.createConstraint(
                0,
                -1,
                self.robots[0].get_body_id(),
                -1,
                p.JOINT_FIXED,
                [0, 0, 1],
                self.robots[0].get_position(),
                [0, 0, 0],
                self.robots[0].get_orientation(),
            )
            p.changeConstraint(self.robot_constraint, maxForce=10000)

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
        if isinstance(self.robots[0], BehaviorRobot):
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
            if self.action_filter == "magic_grasping":
                proprioception_dim += 1
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

        if isinstance(self.robots[0], BehaviorRobot):
            new_action = np.zeros((28,))
            if self.action_filter == "navigation":
                new_action[:2] = action[:2]
                new_action[5] = action[2]
            elif self.action_filter == "mobile_manipulation":
                # body x,y,yaw
                new_action[:2] = action[:2]
                new_action[5] = action[2]
                # left hand 7d
                new_action[12:19] = action[3:10]
                # right hand 7d
                new_action[20:27] = action[10:17]
            elif self.action_filter == "tabletop_manipulation":
                # only using right hand
                new_action[20:27] = action[:7]
            elif self.action_filter == "bimanual_manipulation":
                new_action[12:19] = action[:7]
                new_action[20:27] = action[7:]
            else:
                # all action dims except hand reset
                new_action[:19] = action[:19]
                new_action[20:27] = action[19:]
            # The original action space for BehaviorRobot is too wide for random exploration
            new_action *= 0.05
        elif isinstance(self.robots[0], FetchGripper):
            new_action = np.zeros((11,))
            if self.action_filter == "navigation":
                new_action[:2] = action[:2]
            elif self.action_filter == "tabletop_manipulation":
                new_action[4:] = action[:7]
            elif self.action_filter == "magic_grasping":
                new_action[4:10] = action[:6]
            else:
                new_action = action
        else:
            Exception("Only BehaviorRobot and FetchGripper are supported for behavior_env")

        self.robots[0].apply_action(new_action)
        if self.log_writer is not None:
            self.log_writer.process_frame()
        self.simulator.step()

        if self.action_filter == "magic_grasping":
            self.check_magic_grasping()

        state = self.get_state()
        info = {}
        done, satisfied_predicates = self.task.check_success()
        if len(satisfied_predicates["satisfied"]) == 1:
            print("success")
        # Compute the initial reward potential here instead of during reset
        # because if an intermediate checkpoint is loaded, we need step the
        # simulator before calling task.check_success
        if self.current_step == 1:
            self.reward_potential, self.distance_potential = self.get_potential(satisfied_predicates)

        if self.current_step >= self.config["max_step"]:
            done = True
        reward, info = self.get_reward(satisfied_predicates)

        self.populate_info(info)

        if done and self.automatic_reset:
            info["last_observation"] = state
            state = self.reset()

        return state, reward, done, info

    def get_child_frame_pose(self, ag_bid, ag_link):
        # Different pos/orn calculations for base/links
        if ag_link == -1:
            body_pos, body_orn = p.getBasePositionAndOrientation(ag_bid)
        else:
            body_pos, body_orn = p.getLinkState(ag_bid, ag_link)[:2]

        # Get inverse world transform of body frame
        inv_body_pos, inv_body_orn = p.invertTransform(body_pos, body_orn)
        link_state = p.getLinkState(self.robots[0].get_body_id(), self.robots[0].end_effector_part_index())
        link_pos = link_state[0]
        link_orn = link_state[1]
        # B * T = P -> T = (B-1)P, where B is body transform, T is target transform and P is palm transform
        child_frame_pos, child_frame_orn = p.multiplyTransforms(inv_body_pos, inv_body_orn, link_pos, link_orn)

        return child_frame_pos, child_frame_orn

    def check_magic_grasping(self):
        if self.reward_shaping_relevant_objs is None:
            return
        if self.magic_grasping_cid is not None:
            return
        target_obj = self.task.object_scope[self.reward_shaping_relevant_objs[0]]

        should_grasp = False
        for link in self.robots[0].gripper_joint_ids:
            if (
                len(
                    p.getContactPoints(
                        bodyA=self.robots[0].get_body_id(),
                        linkIndexA=link,
                        bodyB=target_obj.get_body_id(),
                    )
                )
                > 0
            ):
                should_grasp = True
                break

        if should_grasp:
            child_frame_pos, child_frame_orn = self.get_child_frame_pose(target_obj.get_body_id(), -1)
            self.magic_grasping_cid = p.createConstraint(
                parentBodyUniqueId=self.robots[0].get_body_id(),
                parentLinkIndex=self.robots[0].end_effector_part_index(),
                childBodyUniqueId=target_obj.get_body_id(),
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=(0, 0, 0),
                childFramePosition=child_frame_pos,
                childFrameOrientation=child_frame_orn,
            )
            p.changeConstraint(self.magic_grasping_cid, maxForce=10000)

    def get_potential(self, satisfied_predicates):
        potential = 0.0

        # Evaluate the first ground goal state option as the potential
        _, satisfied_predicates = evaluate_state(self.task.ground_goal_state_options[0])
        success_score = len(satisfied_predicates["satisfied"]) / (
            len(satisfied_predicates["satisfied"]) + len(satisfied_predicates["unsatisfied"])
        )
        predicate_potential = success_score * self.predicate_reward_weight
        potential += predicate_potential

        if self.reward_shaping_relevant_objs is not None:
            if isinstance(self.robots[0], FetchGripper):
                reward_shaping_relevant_objs = [self.robots[0].parts["gripper_link"]]
            else:
                if self.action_filter == "bimanual_manipulation":
                    reward_shaping_relevant_objs = [
                        self.robots[0].parts["left_hand"],
                        self.robots[0].parts["right_hand"],
                    ]
                else:
                    reward_shaping_relevant_objs = [self.robots[0].parts["right_hand"]]
            for obj_name in self.reward_shaping_relevant_objs:
                reward_shaping_relevant_objs.append(self.task.object_scope[obj_name])

            if isinstance(self.robots[0], FetchGripper):
                pickup_potential = 0.0
                correct_obj = reward_shaping_relevant_objs[1]
                if self.action_filter == "tabletop_manipulation":
                    self.pickup_correct_obj = (
                        self.robots[0].ag_data is not None and self.robots[0].ag_data[0] == correct_obj.get_body_id()
                    )
                else:
                    self.pickup_correct_obj = self.magic_grasping_cid is not None

                if self.pickup_correct_obj:
                    pickup_potential += self.pickup_reward_weight
                potential += pickup_potential

                self.same_stage = self.pickup_correct_obj == self.pickup_correct_obj_prev_step
                self.pickup_correct_obj_prev_step = self.pickup_correct_obj
            else:
                self.same_stage = True

            distance = None
            if self.action_filter == "bimanual_manipulation":
                # two hands
                pos1 = reward_shaping_relevant_objs[0].get_position()
                pos2 = reward_shaping_relevant_objs[1].get_position()

                # cauldron
                pos3 = reward_shaping_relevant_objs[2].get_position()

                # table
                pos4 = reward_shaping_relevant_objs[3].get_position()
                pos4 = pos4 + np.array([0, 0, 0.25])

                distance = l2_distance(pos1, pos3) + l2_distance(pos2, pos3) + l2_distance(pos3, pos4)
            elif not self.pickup_correct_obj:
                pos1 = reward_shaping_relevant_objs[0].get_position()
                pos2 = reward_shaping_relevant_objs[1].get_position()
                distance = l2_distance(pos1, pos2)
            elif len(reward_shaping_relevant_objs) == 3:
                try:
                    pos1 = reward_shaping_relevant_objs[1].get_position()
                    obj2 = reward_shaping_relevant_objs[2]
                    if obj2.category == "sink":
                        # approach 0.1m below the water source link
                        pos2 = obj2.states[WaterSource].get_link_position()
                        pos2 = pos2 + np.array([0, 0, -0.1])
                    elif obj2.category == "stove":
                        pos2 = obj2.states[HeatSourceOrSink].get_link_position()
                    elif obj2.category == "shelf":
                        # distance to the closest dust particle
                        particles = obj2.states[Stained].dirt.get_active_particles()
                        particle_pos = np.array([particle.get_position() for particle in particles])
                        distance_to_particles = np.linalg.norm(particle_pos - pos1, axis=1)
                        closest_particle = np.argmin(distance_to_particles)
                        pos2 = particles[closest_particle].get_position()
                    else:
                        pos2 = obj2.get_position()
                    distance = l2_distance(pos1, pos2)
                except Exception:
                    # One of the objects has been sliced, skip distance
                    pass

            # Use the current distance potential if the previous and current steps are within the same stage
            # If not, use the previous distance potential
            if distance is None:
                distance_potential = self.distance_potential
            else:
                distance_potential = -distance * self.distance_reward_weight

            # distance = 0.0
            # for i in range(len(reward_shaping_relevant_objs) - 1):
            #     try:
            #         pos1 = reward_shaping_relevant_objs[i].get_position()
            #         obj2 = reward_shaping_relevant_objs[i + 1]
            #         if obj2.category == "sink":
            #             # approach 0.1m below the water source link
            #             pos2 = obj2.states[WaterSource].get_link_position()
            #             pos2 = pos2 + np.array([0, 0, -0.1])
            #         elif obj2.category == "stove":
            #             pos2 = obj2.states[HeatSourceOrSink].get_link_position()
            #         elif obj2.category == "shelf":
            #             # distance to the closest dust particle
            #             particles = obj2.states[Stained].dirt.get_active_particles()
            #             particle_pos = np.array([particle.get_position() for particle in particles])
            #             distance_to_particles = np.linalg.norm(particle_pos - pos1, axis=1)
            #             closest_particle = np.argmin(distance_to_particles)
            #             pos2 = particles[closest_particle].get_position()
            #         else:
            #             pos2 = obj2.get_position()
            #         distance += l2_distance(pos1, pos2)
            #     except Exception:
            #         # One of the objects has been sliced, skip distance
            #         continue
            # distance_potential = -distance * self.distance_reward_weight
            # potential += distance_potential

        if self.progress_reward_objs is not None:
            for obj_name, state_name, potential_weight in self.progress_reward_objs:
                obj = self.task.object_scope[obj_name]
                state = obj.states[get_state_from_name(state_name)]
                if isinstance(state, Stained):
                    potential += state.dirt.get_num_active() * potential_weight
                else:
                    assert False, "unknown progress reward"

        return potential, distance_potential

    def get_reward(self, satisfied_predicates):
        reward = 0.0
        new_potential, new_distance_potential = self.get_potential(satisfied_predicates)
        reward += new_potential - self.reward_potential
        if self.same_stage:
            reward += new_distance_potential - self.distance_potential
        self.reward_potential = new_potential
        self.distance_potential = new_distance_potential
        return reward, {"satisfied_predicates": satisfied_predicates}

    # def get_reward(self, satisfied_predicates):
    #     new_potential = self.get_potential(satisfied_predicates)
    #     reward = new_potential - self.reward_potential
    #     self.reward_potential = new_potential
    #     return reward, {"satisfied_predicates": satisfied_predicates}

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
            if self.action_filter == "magic_grasping":
                # add another dimension: whether the magic grasping is active
                state["proprioception"] = np.append(state["proprioception"], float(self.magic_grasping_cid is not None))
        return state

    def reset_scene_and_agent(self):
        if self.reset_checkpoint_dir is not None and self.reset_checkpoint_idx != -1:
            load_checkpoint(self.simulator, self.reset_checkpoint_dir, self.reset_checkpoint_idx)
        else:
            self.task.reset_scene(snapshot_id=self.task.initial_state)
        # set the constraints to the current poses
        self.robots[0].apply_action(np.zeros(self.robots[0].action_dim))

    def reset(self):
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

        self.reset_scene_and_agent()
        if self.magic_grasping_cid is not None:
            p.removeConstraint(self.magic_grasping_cid)
            self.magic_grasping_cid = None

        self.simulator.sync(force_sync=True)
        state = self.get_state()
        self.reset_variables()

        return state

    def reset_variables(self):
        super(BehaviorEnv, self).reset_variables()
        self.pickup_correct_obj = False
        self.pickup_correct_obj_prev_step = False
        self.distance_potential = None


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
        default="gui",
        help="which mode for simulation (default: headless)",
    )
    parser.add_argument(
        "--action_filter",
        "-af",
        choices=[
            "navigation",
            "tabletop_manipulation",
            "mobile_manipulation",
            "bimanual_manipulation",
            "magic_grasping",
            "all",
        ],
        default="mobile_manipulation",
        help="which action filter",
    )
    args = parser.parse_args()

    env = BehaviorEnv(
        config_file=args.config,
        mode=args.mode,
        action_timestep=1.0 / 10.0,
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
            action[:] = 0.0
            embed()
            state, reward, done, info = env.step(action)
            print("reward:", reward, "done:", done, "info:", info)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
        embed()
    env.close()
