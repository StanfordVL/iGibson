import logging
import random
import numpy as np
import pybullet as p
import os
import time
import glob
import csv
import json
import xml.etree.ElementTree as ET
import kornia.geometry.transform as korntransforms
import torch

import igibson
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.point_goal import PointGoal
from igibson.termination_conditions.timeout import Timeout
from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition
from igibson.objects import cube
from igibson.objects.visual_marker import VisualMarker
from igibson.utils.utils import l2_distance, restoreState
from igibson.agents.savi.utils import dataset
from igibson.agents.savi.utils.dataset import CATEGORIES, CATEGORY_MAP, MP3D_CAT_MAP
from igibson.utils.utils import rotate_vector_3d
from igibson.utils.assets_utils import get_scene_path
from PIL import Image
import cv2
from scipy import ndimage
import math


log = logging.getLogger(__name__)

class TimeReward(BaseRewardFunction):
    """
    Time reward
    A negative reward per time step
    """

    def __init__(self, config):
        super().__init__(config)
        self.time_reward_weight = self.config.get(
            'time_reward_weight', -0.01)

    def get_reward(self, task, env):
        """
        Reward is proportional to the number of steps
        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        return self.time_reward_weight

    
class ViewPoints(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super().__init__(config)
        self.dist_tol = self.config.get("dist_tol", 0.5)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)
        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = False
        if any([l2_distance(env.robots[0].get_position()[:2], view_point) < self.dist_tol for view_point in task.view_points]):
            done = True
        success = done
        return done, success
    
def buildMatterportCategories():
    matterportSemanticClassToCategory = {}
    with open("../../audio/category_mapping.tsv") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            if i > 0:
                matterportSemanticClassToCategory[int(row[0])] = MP3D_CAT_MAP[row[2]] if row[2] in MP3D_CAT_MAP else row[2]
    return matterportSemanticClassToCategory


def mp3d_load_object_poses(scene_id, cat, floor_height, next_floor_height):
    mapping = buildMatterportCategories()
    json_file_path = os.path.join(get_scene_path(scene_id), "house_segmentations/"+scene_id+".semseg.json")
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    contents = contents["segGroups"]
    objects_poses = []
    for c in contents:
        if c["label_index"] == -1:
            continue
        if mapping[c["label_index"]] == cat and c["obb"]["centroid"][2] >= floor_height and (next_floor_height is None or c["obb"]["centroid"][2] <= next_floor_height):
            
            objects_poses.append(np.asarray(c["obb"]["centroid"]))
            
    return objects_poses


class SAViRTTask(PointNavRandomTask):
    # reward function
    def __init__(self, env):
        super().__init__(env)
        self.reward_functions = [
            PotentialReward(self.config), # geodesic distance, potential_reward_weight
            TimeReward(self.config), # time_reward_weight 
            PointGoalReward(self.config), # success_reward
            CollisionReward(self.config),
             
        ]
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
            ViewPoints(self.config),
        ]
        self.cat = None # audio category
        self._episode_time = 0.0
        self.target_obj = None
        self.target_pos = None
        self.view_points = []
        self.num_view_points = 10
        self.num_vp_per_joint = 2
        self.viewpoints_marker = [None for _ in range(self.num_view_points)]
        self.load_target(env)
        self.load_viewpoints(env)
        
        
    def load_target(self, env):
        """
        Load target marker, hidden by default
        :param env: environment instance
        """

        cyl_length = 0.2

        self.target_marker = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius=self.dist_tol, # * 2 for clearer view
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        )

        env.simulator.import_object(self.target_marker)

        # The visual object indicating the target location may be visible
        for instance in self.target_marker.renderer_instances:
            instance.hidden = not self.visible_target

            
    def load_viewpoints(self, env):
        cyl_length = 0.2
        
        for i in range(self.num_view_points):
            self.viewpoints_marker[i] = VisualMarker(
                visual_shape=p.GEOM_CYLINDER,
                rgba_color=[255/255, 99/255, 71/255, 0.5/255],
                radius=self.dist_tol,
                length=cyl_length,
                initial_offset=[0, 0, cyl_length / 2.0],
            )

            env.simulator.import_object(self.viewpoints_marker[i])

            # The visual object indicating the target location may be visible
            for instance in self.viewpoints_marker[i].renderer_instances:
                instance.hidden = not self.visible_target
                
            
    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        task_obs = self.global_to_local(env, self.target_pos)[:2]
        if self.goal_format == "polar":
            task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        task_obs = np.append(task_obs, [linear_velocity, angular_velocity])
        return task_obs
    
    
    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        max_trials = 200
        dist = 0.0
        set_target = False
        for i in range(max_trials):
            self.cat = random.choice(CATEGORIES)
            if self.config['scene'] == "igibson":
                objects = env.scene.objects_by_category[self.cat]
                if len(objects) != 0:
                    self.target_obj = random.choice(objects)             
                    target_pos = np.array(self.target_obj.get_position())
                    set_target = True
                    break
            elif self.config['scene'] == "gibson" or self.config['scene'] == "mp3d": # gibson or mp3d
                target_poses = mp3d_load_object_poses(env.scene.scene_id, self.cat, 
                                                      env.scene.floor_heights[self.floor_num],
                                                      None if len(env.scene.floor_heights)<=self.floor_num+1 else \
                                                      env.scene.floor_heights[self.floor_num+1])
                if len(target_poses) != 0:
                    self.target_obj = None 
                    target_pos = random.choice(target_poses)   
                    set_target = True
                    break
        if not set_target:
            logging.warning("WARNING: Failed to set target")
        
        for _ in range(max_trials):
            _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
            if env.scene.build_graph:
                _, dist = env.scene.get_shortest_path(
                    self.floor_num,
                    initial_pos[:2],
                    target_pos[:2], entire_path=False)
            else:
                dist = l2_distance(initial_pos, target_pos)
            if self.target_dist_min < dist < self.target_dist_max: 
                break
                
        if not (self.target_dist_min < dist < self.target_dist_max):
            logging.warning("WARNING: Failed to sample initial and target positions")
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn, target_pos


    def sample_pose_near_object(self, env, pos=None, check_pose_per_joint=False):
        obj = self.target_obj
        pos_on_obj = self.target_pos if pos is None else pos
        if self.config["scene"] == "igibson":
            obj_rooms = obj.in_rooms if obj.in_rooms else [env.scene.get_room_instance_by_point(pos_on_obj[:2])]
        max_trials = 100
        count = 0
        for _ in range(max_trials):
            if len(self.view_points) >= self.num_view_points:
                break
            if check_pose_per_joint and count >= self.num_vp_per_joint:
                break
            distance = np.random.uniform(self.dist_tol*2, 1.0)
            yaw = np.random.uniform(-np.pi, np.pi)
            pose_2d = np.array(
                [pos_on_obj[0] + distance * np.cos(yaw), pos_on_obj[1] + distance * np.sin(yaw), np.pi+yaw]
            )

            # Check room
            if self.config["scene"] == "igibson" and env.scene.get_room_instance_by_point(pose_2d[:2]) not in obj_rooms:
                continue
            if not env.test_valid_position(env.robots[0], pose_2d):
                continue
            
            self.view_points.append(pose_2d[:2])
            count += 1
        
        
    def reset_agent(self, env, train=True):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.
        :param env: environment instance
        """
        reset_success = False
        max_trials = 100
        self.view_points = []

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, target_pos = \
                self.sample_initial_pose_and_target_pos(env)
            reset_success = env.test_valid_position(
                env.robots[0], initial_pos, initial_orn) #and \
#                 env.test_valid_position(
#                     env.robots[0], target_pos)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")
            
        self.target_pos = target_pos
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        
        if self.config["scene"] == "igibson":
            poses = self.target_obj.get_poses()
            for i in range(1, len(poses)):
                pose_i = np.array([*poses[i][0][:2], env.scene.floor_heights[self.floor_num]])
                self.sample_pose_near_object(env, pose_i, check_pose_per_joint=True)
             
        self.sample_pose_near_object(env)
        if len(self.view_points) == 0:
            logging.warning("WARNING: Failed to sample poses near the object" + str(self.cat))
        for i, view_point in enumerate(self.view_points):
            self.viewpoints_marker[i].set_position(np.array([*view_point, env.scene.floor_heights[self.floor_num]]))
              
        self.target_marker.set_position(np.array([*self.target_pos[:2], env.scene.floor_heights[self.floor_num]]))
            
        p.removeState(state_id)
        
        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        
        self.initial_rpy = np.array(env.robots[0].get_rpy())
        self.load_gt_rt_map(env)
        # for savi
        if self.target_obj is not None:
            self.audio_obj_id = self.target_obj.get_body_ids()[0]
        else:
            self.audio_obj_id = self.target_marker.get_body_ids()[0]
            
        if train:
            env.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'] \
                                            +"/train/"+self.cat+".wav", enabled=True)
        else:
            env.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'] \
                                            +"/val/"+self.cat+".wav", enabled=True)    
        env.audio_system.setSourceRepeat(self.audio_obj_id, repeat = True)


    def load_gt_rt_map(self, env):
        if self.config["scene"] == "igibson":
            maps_path = "/viscam/u/wangzz/iGibson/gibson2/data/ig_dataset/scenes/"  \
                        + env.config['scene_id'] + "/layout/"
            floor = 0
            gt_rt = np.array(Image.open(os.path.join(maps_path, "floor_semseg_{}.png".format(floor))))
            gt_rt = cv2.resize(gt_rt, (env.scene.trav_map_size, env.scene.trav_map_size))

            ######sanity check
            # pos = env.scene.world_to_map(np.array(env.robots[0].get_position())[:2])
            # print("robo pos", pos)
            # gt_rt[pos[0], pos[1]] = 255
            # cv2.imwrite(env.config['scene_id'] + "_gt_rt.png", gt_rt /23 * 255)
            ######

            gt_rt = np.flip(gt_rt, axis=0)
            padded_shape = 350
            padded_gt_rt = np.zeros((padded_shape, padded_shape))
            padded_gt_rt[int(padded_gt_rt.shape[0]/2 - gt_rt.shape[0]/2):int(padded_gt_rt.shape[0]/2 + gt_rt.shape[0]/2),
                int(padded_gt_rt.shape[1]/2 - gt_rt.shape[1]/2):int(padded_gt_rt.shape[1]/2 + gt_rt.shape[1]/2)] = gt_rt

            # cv2.imwrite(env.config['scene_id'] + "_gt_rt_after_pad.png", padded_gt_rt /23 * 255)

            delta_pos = self.initial_pos[:2]/env.scene.trav_map_resolution
            padded_gt_rt = ndimage.shift(padded_gt_rt, [delta_pos[1], -delta_pos[0]]) #move gt_map [left, down] by delta_pos        
            # cv2.imwrite(env.config['scene_id'] + "gt_rt_shifted.png", padded_gt_rt/23 * 255)
            padded_gt_rt = ndimage.rotate(padded_gt_rt, -90+self.initial_rpy[2]*180.0/math.pi, reshape=False)# rotate gt_map by theta-90 ccw

            ######sanity check
    #         cv2.imwrite("gt_rt_rotated.png", gt_rt)
            ######
#         elif self.config["scene"] == "gibson" or self.config["scene"] == "mp3d":

        self.gt_rt = cv2.resize(padded_gt_rt, (50, 50))
        # cv2.imwrite(env.config['scene_id'] + "gt_rt_final_resize.png", self.gt_rt/23 * 255)
    
    def get_room_type_map(self):
        return self.gt_rt
    
    
if __name__ == "__main__":
    from igibson.envs.igibson_env import iGibsonEnv

    # config = "/viscam/u/li2053/iGibson-dev/igibson/agents/savi_rt/config/savi_rt_audiogoal.yaml"
    # env = iGibsonEnv(config_file=config, 
    #                  mode="headless", 
    #                  scene_id = "Pomaria_1_int",
    #                  action_timestep=1.0 / 10.0, 
    #                  physics_timestep=1.0 / 40.0)
    # env.reset()   
    # env.task.load_gt_rt_map(env)
    # env.close()
    scenes = "Pomaria_1_int", "Benevolence_2_int", "Beechwood_1_int", "Ihlen_0_int", "Benevolence_1_int","Pomaria_2_int", "Merom_1_int", "Ihlen_1_int", "Wainscott_0_int",  # "Benevolence_0_int"], 
    # scenes = "Merom_1_int", "Beechwood_0_int", "Wainscott_1_int", "Merom_0_int", "Rs_int", "Pomaria_0_int"
    for s in scenes:
        print(s)
        maps_path = "/viscam/u/wangzz/iGibson/gibson2/data/ig_dataset/scenes/"  \
                            + s + "/layout/"
        floor = 0
        gt_rt = np.array(Image.open(os.path.join(maps_path, "floor_semseg_{}.png".format(floor))))
        if s == "Beechwood_1_int":
            gt_rt = gt_rt[ 500:2220,250:1480]
        print(gt_rt.shape)
        # gt_rt = cv2.resize(gt_rt, (100, 100))
        # padded_gt_rt = np.zeros((150, 150))
        # padded_gt_rt[25:125, 25:125] = gt_rt

        cv2.imwrite(s + "_gt_rt.png", gt_rt /23 * 255)