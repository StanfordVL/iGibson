import argparse
import logging
import os
import time
from collections import OrderedDict

#TODO: GOKUL HACK
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
import sys
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from PIL import Image
import skimage
import gym
import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat
from igibson.utils.mesh_util import ortho

from igibson import object_states
from igibson.envs.env_base import BaseEnv
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
from igibson.tasks.audiogoal_nav_task import AudioGoalNavTask, AudioPointGoalNavTask, AudioGoalVRNavTask
from igibson.tasks.savi_task import SAViTask
from igibson.tasks.savi_rt_task import SAViRTTask
from igibson.tasks.audio_nav_in_savi_task import avNavSAViTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.utils import quatToXYZW
from igibson.agents.savi.utils.dataset import CATEGORIES, CATEGORY_MAP, MAP_SIZE
from igibson.utils.utils import rotate_vector_3d
from igibson.agents.savi.utils.logs import logger
log = logging.getLogger(__name__)


class iGibsonEnv(BaseEnv):
    """
    iGibson Environment (OpenAI Gym interface).
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
        use_pb_gui=False,
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
        super(iGibsonEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            rendering_settings=rendering_settings,
            vr_settings=vr_settings,
            device_idx=device_idx,
            use_pb_gui=use_pb_gui,
        )
        self.scene_id = scene_id
        self.num_episode = 0
        self.automatic_reset = self.config.get("automatic_reset", True)

    def load_task_setup(self):
        """
        Load task setup.
        """
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

        # task
        if "task" not in self.config:
            self.task = DummyTask(self)
        elif self.config["task"] == "point_nav_fixed":
            self.task = PointNavFixedTask(self)
        elif self.config["task"] == "point_nav_random":
            self.task = PointNavRandomTask(self)
        elif self.config["task"] == "interactive_nav_random":
            self.task = InteractiveNavRandomTask(self)
        elif self.config["task"] == "dynamic_nav_random":
            self.task = DynamicNavRandomTask(self)
        elif self.config["task"] == "reaching_random":
            self.task = ReachingRandomTask(self)
        elif self.config["task"] == "room_rearrangement":
            self.task = RoomRearrangementTask(self)
        elif self.config['task'] == 'audiogoal_nav':
            self.task = AudioGoalNavTask(self)
        elif self.config['task'] == 'audiogoal_nav_vr':
            self.task = AudioGoalVRNavTask(self)
        elif self.config['task'] == 'audiopointgoal_nav':
            self.task = AudioPointGoalNavTask(self)
        elif self.config['task'] == 'SAVi':
            self.task = SAViTask(self)
        elif self.config['task'] == 'SAViRT':
            self.task = SAViRTTask(self)
        elif self.config['task'] == "avNavSAViTask":
            self.task = avNavSAViTask(self)
        else:
            try:
                import bddl

                with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
                    all_activities = [line.strip() for line in f.readlines()]

                if self.config["task"] in all_activities:
                    self.task = BehaviorTask(self)
                else:
                    raise Exception("Invalid task: {}".format(self.config["task"]))
            except ImportError:
                raise Exception("bddl is not available.")

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces.

        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def load_observation_space(self):
        """
        Load observation space.
        """
        self.output = self.config["output"]
        self.image_width = self.config.get("image_width", 128)
        self.image_height = self.config.get("image_height", 128)
        self.image_width_video = self.config.get("image_width_video", 960)
        self.image_height_video = self.config.get("image_height_video", 960)
        self.rt_map_height_video = self.config.get("rt_map_height_video", 960)
        self.rt_map_width_video = self.config.get("rt_map_width_video", 960)
        
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if "task_obs" in self.output:
            observation_space["task_obs"] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf
            )
        if "rgb" in self.output:
            if len(self.config["VIDEO_OPTION"])!=0:
                observation_space["rgb_video"] = self.build_obs_space(
                shape=(self.image_height_video, self.image_width_video, 3), low=0.0, high=1.0
                )
                
            observation_space["rgb"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb")
            
        if "depth" in self.output:
            if len(self.config["VIDEO_OPTION"])!=0:
                observation_space["depth_video"] = self.build_obs_space(
                shape=(self.image_height_video, self.image_width_video, 1), low=0.0, high=1.0
                )     
            observation_space["depth"] = self.build_obs_space(
            shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            
            vision_modalities.append("depth")
        if "pc" in self.output:
            observation_space["pc"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("pc")
        if "optical_flow" in self.output:
            observation_space["optical_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2), low=-np.inf, high=np.inf
            )
            vision_modalities.append("optical_flow")
        if "scene_flow" in self.output:
            observation_space["scene_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("scene_flow")
        if "normal" in self.output:
            observation_space["normal"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("normal")
        if "seg" in self.output:
            observation_space["seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_CLASS_COUNT
            )
            vision_modalities.append("seg")
        if "ins_seg" in self.output:
            observation_space["ins_seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_INSTANCE_COUNT
            )
            vision_modalities.append("ins_seg")
        if "rgb_filled" in self.output:  # use filler
            observation_space["rgb_filled"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb_filled")
        if "highlight" in self.output:
            observation_space["highlight"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("highlight")
        if "scan" in self.output:
            self.n_horizontal_rays = self.config.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan")
        if "occupancy_grid" in self.output:
            self.grid_resolution = self.config.get("grid_resolution", 128)
            self.occupancy_grid_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.grid_resolution, self.grid_resolution, 1)
            )
            observation_space["occupancy_grid"] = self.occupancy_grid_space
            scan_modalities.append("occupancy_grid")
        if "bump" in self.output:
            observation_space["bump"] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
            sensors["bump"] = BumpSensor(self)
        if "proprioception" in self.output:
            observation_space["proprioception"] = self.build_obs_space(
                shape=(self.robots[0].proprioception_dim,), low=-np.inf, high=np.inf
            )
        if 'audio' in self.output:
            if self.audio_system is not None:
                spectrogram = self.audio_system.get_spectrogram()
                observation_space['audio'] = self.build_obs_space(
                    shape=spectrogram.shape, low=-np.inf, high=np.inf)
            else:
                # GOKUL HACK
                observation_space['audio'] = self.build_obs_space(
                    shape=(257, 83, 2), low=-np.inf, high=np.inf)
        if 'top_down' in self.output:
            if len(self.config["VIDEO_OPTION"])!=0:
                observation_space["top_down_video"] = self.build_obs_space(
                shape=(self.image_height_video, self.image_width_video, 3), low=0.0, high=1.0
                )
            observation_space['top_down'] = self.build_obs_space(
                shape=(self.image_height_video, self.image_width_video, 3), low=-np.inf, high=np.inf)
        if 'category_belief' in self.output:
            observation_space['category_belief'] = self.build_obs_space(
                shape=(len(CATEGORIES),), low=0.0, high=1.0)
        if 'location_belief' in self.output:
            observation_space['location_belief'] = self.build_obs_space(
                shape=(2,), low=0.0, high=1.0)
        if 'pose_sensor' in self.output:
            observation_space['pose_sensor'] = self.build_obs_space(
                shape=(4,), low=-np.inf, high=np.inf)
        if 'category' in self.output:
            observation_space['category'] = self.build_obs_space(
                shape=(len(CATEGORIES),), low=0.0, high=1.0)
        if 'floorplan_map' in self.output:
            observation_space['floorplan_map'] = self.build_obs_space(
                shape=(self.image_height, self.image_height), low=0, high=23)
        if 'rt_map_features' in self.output:
            if len(self.config["VIDEO_OPTION"])!=0:
                observation_space["rt_map_video"] = self.build_obs_space(
                shape=(self.rt_map_height_video, self.rt_map_width_video, 3), low=0.0, high=1.0
                )
                
                observation_space["rt_map_gt_video"] = self.build_obs_space(
                shape=(self.rt_map_height_video, self.rt_map_width_video, 3), low=0.0, high=1.0
                )
            observation_space['rt_map_features'] = self.build_obs_space(
                shape=(8192,), low=-np.inf, high=np.inf)
            observation_space['rt_map'] = self.build_obs_space(
                shape=(23,32,32), low=-np.inf, high=np.inf)
            observation_space['rt_map_gt'] = self.build_obs_space(
                shape=(32,32), low=-np.inf, high=np.inf)
            observation_space['visual_features'] = self.build_obs_space(
                shape=(128,), low=-np.inf, high=np.inf)
            observation_space['audio_features'] = self.build_obs_space(
                shape=(128,), low=-np.inf, high=np.inf)
            observation_space['map_resolution'] = self.build_obs_space(
                shape=(1,), low=0.0, high=np.inf)

        if len(vision_modalities) > 0:
            sensors["vision"] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors["scan_occ"] = ScanSensor(self, scan_modalities)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.sensors = sensors

    def load_action_space(self):
        """
        Load action space.
        """
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping.
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []

    def load(self):
        """
        Load environment.
        """
        super(iGibsonEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def get_state(self):
        """
        Get the current observation.

        :return: observation as a dictionary
        """
        state = OrderedDict()
        if "task_obs" in self.output:
            state["task_obs"] = self.task.get_task_obs(self)
        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
            if len(self.config["VIDEO_OPTION"])!=0:
                state['depth_video'] = state['depth']
                state['rgb_video'] = state['rgb']
                state["depth"] = skimage.measure.block_reduce(state["depth"], (6,6,3), np.mean)
#                 if not self.config["extra_rgb"]:
                state["rgb"] = skimage.measure.block_reduce(state["rgb"], (6,6,1), np.mean)
            # (img_height, img_height, 1)
            # because the rendered robot camera should have the same image size for rgb and depth
        if "scan_occ" in self.sensors:
            scan_obs = self.sensors["scan_occ"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "bump" in self.sensors:
            state["bump"] = self.sensors["bump"].get_obs(self)
        if "proprioception" in self.output:
            state["proprioception"] = np.array(self.robots[0].get_proprioception())

        if 'audio' in self.output:
            state['audio'] = self.audio_system.get_spectrogram()
            
        if 'top_down' in self.output:
            camera_pose = np.array([0, 0, 4.0])
            view_direction = np.array([0, 0, -1])
            self.simulator.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
            p_range = MAP_SIZE[self.scene_id] / 200.0
            prevP = self.simulator.renderer.P.copy()
            self.simulator.renderer.P = ortho(-p_range, p_range, -p_range, p_range, -10, 20.0)
            frame, three_d = self.simulator.renderer.render(modes=("rgb", "3d"))
            depth = -three_d[:, :, 2]
            frame[depth == 0] = 1.0
#             frame = cv2.flip(frame, 0)
            bg = (frame[:, :, 0:3][:, :, ::-1] * 255).astype(np.uint8)
            state['top_down'] = bg
            state['top_down_video'] = state['top_down']
            self.simulator.renderer.P = prevP
            
        if 'pose_sensor' in self.output:
            # TODO: pose sensor in the episode frame
            # (x, y, heading, time)
            pos = np.array(self.robots[0].get_position()) #[x,y,z]
            rpy = np.array(self.robots[0].get_rpy()) #(3,)
            pos[2] = 0
            self.task.initial_pos[2] = 0 # remove z axis offset
            #initial rpy[2]: -pi to pi
            #initial orn[2]: 0 to 2pi
            pos_eframe = rotate_vector_3d(pos - self.task.initial_pos, 0, 0, self.task.initial_rpy[2])
            # the new x axis is the forward direction of the initial pose
            # the new y axis is the leftward direction of the initial pose
            if rpy[2] - self.task.initial_rpy[2] > np.pi:
                orn_eframe = rpy[2] - self.task.initial_rpy[2] - 2*np.pi
            elif rpy[2] - self.task.initial_rpy[2] < -np.pi:
                orn_eframe = rpy[2] - self.task.initial_rpy[2] + 2*np.pi
            else:
                orn_eframe = rpy[2] - self.task.initial_rpy[2]
            state['pose_sensor'] = np.array(
                [*pos_eframe[:2], orn_eframe, self.task._episode_time],
                dtype=np.float32)
            self.task._episode_time += 1.0 #* action_time
        
        if 'category' in self.output:
            index = CATEGORY_MAP[self.task.cat]
            onehot = np.zeros(len(CATEGORIES))
            onehot[index] = 1
            state['category'] = onehot
        
        # categoty_belief and location_belief are updated in _collect_rollout_step
        if "category_belief" in self.output:
            state["category_belief"] = np.zeros(len(CATEGORIES))
        if "location_belief" in self.output:
            state["location_belief"] = np.zeros(2)
        if 'rt_map_features' in self.output:
            state['rt_map_features'] = np.zeros(8192)
            state['rt_map'] = np.zeros((23,32,32))
            state['rt_map_gt'] = self.task.get_room_type_map()
            state['visual_features'] = np.zeros(128)
            state['audio_features'] = np.zeros(128)
            state['map_resolution'] = self.scene.trav_map_resolution
            if len(self.config["VIDEO_OPTION"])!=0:
                state['rt_map_video'] = state['rt_map']
                state['rt_map_gt_video'] = state['rt_map_gt']
            
        if "floorplan_map" in self.output:
            mapdir = '/viscam/u/wangzz/avGibson/data/ig_dataset/scenes/resized_sem/' + self.scene_id + ".png"
            state["floorplan_map"] = np.array(Image.open(mapdir))

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class).

        :return: a list of collisions from the last physics timestep
        """
        self.simulator_step()
        collision_links = [
            collision for bid in self.robots[0].get_body_ids() for collision in p.getContactPoints(bodyA=bid)
        ]
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored.

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        # TODO: Improve this to accept multi-body robots.
        new_collision_links = []
        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].base_link.body_id and item[4] in self.collision_ignore_link_a_ids:
                continue
            new_collision_links.append(item)
        return new_collision_links

    def populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        :param info: the info dictionary to populate
        """
        info["episode_length"] = self.current_step
        info["collision_step"] = self.collision_step

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1
        if action is not None:
            self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        state = self.get_state()
        info = {}
        reward, info = self.task.get_reward(self, collision_links, action, info)
        done, info = self.task.get_termination(self, collision_links, action, info)
        self.task.step(self)
        self.populate_info(info)
        
        if done and len(self.config['VIDEO_OPTION'])>0: # generate video
            self.audio_system.save_audio()

        if done and self.automatic_reset:
            info["last_observation"] = state
            episodes_per_scene = self.config.get("TRAIN_EPISODE_PER_SCENE", None)
            if episodes_per_scene is not None and self.num_episode % episodes_per_scene == 0:# and self.num_episode!=0:
                next_scene_id = np.random.choice(self.config['scene_splits'])
                logger.info("reloading scene {}".format(next_scene_id))
                self.config["robot"] = self.robot_config_
                self.reload_model(next_scene_id)
            self.num_episode += 1            
            state = self.reset()
            

        return state, reward, done, info

    def check_collision(self, body_id):
        """
        Check whether the given body_id has collision after one simulator step

        :param body_id: pybullet body id
        :return: whether the given body_id has collision
        """
        self.simulator_step()
        collisions = list(p.getContactPoints(bodyA=body_id))

        if log.isEnabledFor(logging.INFO):  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                log.debug("bodyA:{}, bodyB:{}, linkA:{}, linkB:{}".format(item[1], item[2], item[3], item[4]))

        return len(collisions) > 0

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), "wxyz"))
        # get the AABB in this orientation
        lower, _ = obj.states[object_states.AABB].get_value()
        # Get the stable Z
        stable_z = pos[2] + (pos[2] - lower[2])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def test_valid_position(self, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: whether the position is valid
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        has_collision = any(self.check_collision(body_id) for body_id in obj.get_body_ids())
        return not has_collision

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if any(len(p.getContactPoints(bodyA=body_id)) > 0 for body_id in obj.get_body_ids()):
                land_success = True
                break

        if not land_success:
            log.warning("Object failed to land.")

        if is_robot:
            obj.reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links = []

    def randomize_domain(self):
        """
        Randomize domain.
        Object randomization loads new object models with the same poses.
        Texture randomization loads new materials and textures for the same object models.
        """
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

    def reset(self):
        """
        Reset episode.
        """
        if self.audio_system is not None:
            self.audio_system.reset()
        self.randomize_domain()
        # Move robot away from the scene.
        self.robots[0].set_position([100.0, 100.0, 100.0])
        self.task.reset(self)
        self.simulator.sync(force_sync=True)
        state = self.get_state()
        self.reset_variables()

        return state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default="headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = iGibsonEnv(config_file=args.config, mode=args.mode, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)

    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        for _ in range(100):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print("reward", reward)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
