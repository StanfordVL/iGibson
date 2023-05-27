import logging

import numpy as np
import pybullet as p

from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.objects.visual_marker import VisualMarker
from igibson.utils.utils import cartesian_to_polar, restoreState, l2_distance, rotate_vector_3d
import cv2
from PIL import Image
from scipy import ndimage
import os
import math

class AudioGoalNavTask(PointNavRandomTask):
    """
    Redefine the task (reward functions)
    """
    def __init__(self, env):
        super(AudioGoalNavTask, self).__init__(env)
        self.target_obj  = None
        self.load_target(env)

    def reset_agent(self, env):
        super().reset_agent(env)
        self.target_pos = [0., -1.0, 0.73]
        self.target_obj.set_position([.0, -1., 0.73])#self.target_pos)
        audio_obj_id = self.target_obj.get_body_ids()[0]
        env.audio_system.registerSource(audio_obj_id, self.config['audio_dir'], enabled=True)
        env.audio_system.setSourceRepeat(audio_obj_id)

    def load_target(self, env):
        """
        Load target marker, hidden by default
        :param env: environment instance
        """

        cyl_length = 0.2

        self.target_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius= 0.1, #self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        )

        env.simulator.import_object(self.target_obj)

        # The visual object indicating the target location may be visible
        # for instance in self.target_obj.renderer_instances:
        #     instance.hidden = not self.visible_target
        


    def get_task_obs(self, env):
        """
        Get current velocities

        :param env: environment instance
        :return: task-specific observation
        """
        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        task_obs = np.append(linear_velocity, angular_velocity)

        return task_obs

class AudioGoalNavTaskSim2Real(PointNavRandomTask):
    """
    Redefine the task (reward functions)
    """
    def __init__(self, env):
        super(AudioGoalNavTaskSim2Real, self).__init__(env)
        self.target_obj  = None
        self.load_target(env)
        self.audio_category = None # audio category
        self.audio_setting = self.config['audio_setting']
        self.need_pose_obs = self.config["need_pose"]
        self._episode_time = 0.0

    def reset_agent(self, env):
        super().reset_agent(env)

        # based on audio_setting, randomize the audio type
        if self.audio_setting == "unheard":
            self.audio_category = random.choice(CATEGORIES)
        else:
            self.audio_category = "telephone"

        audio_dir = self.config['audio_dir']  #+ "/train/" + self.audio_category +".wav"

        #set the source height to be the same in the real world
        self.target_pos = [0.0, -1.0, 0.725]
        self.target_obj.set_position(self.target_pos)#self.target_pos)
        # pos = np.zeros((3,))
        # pos[:2] = self.target_pos[:2]
        # pos[2] = 0.725
        # self.target_obj.set_position(pos)
        audio_obj_id = self.target_obj.get_body_ids()[0]

        # randomize the source gain
        src_gain = np.random.uniform(self.config['src_gain_min'], self.config['src_gain_max'])
        print("current source gain is", src_gain)
        env.audio_system.registerSource(audio_obj_id, audio_dir, enabled=True, source_gain=src_gain, near_field_gain=self.config["near_field_gain"], reverb_gain=self.config["reverb_gain"])
        env.audio_system.setSourceRepeat(audio_obj_id)
        self.load_gt_rt_map(env)


    def load_target(self, env):
        """
        Load target marker, hidden by default
        :param env: environment instance
        """

        cyl_length = 0.2

        self.target_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius= self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        )

        env.simulator.import_object(self.target_obj)

        # The visual object indicating the target location may be visible
        # for instance in self.target_obj.renderer_instances:
        #     instance.hidden = not self.visible_target

    def get_task_obs(self, env):
        """
        Get current velocities

        :param env: environment instance
        :return: task-specific observation
        """
        local_tgt_pose = self.global_to_local(env, self.target_pos)[:2]
        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        if self.need_pose_obs:
            task_obs = np.append(local_tgt_pose, [linear_velocity, angular_velocity])
        else:
            task_obs = np.append(linear_velocity, angular_velocity)

        return task_obs
    
    def load_gt_rt_map(self, env):
        if self.config["scene"] == "igibson":
            maps_path = "C:/Users/capri28/Documents/iGibson-dev/igibson/data/ig_dataset/scenes/"  \
                        + env.config['scene_id'] + "/layout/"
            floor = 0
            gt_rt = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_door_{}.png".format(floor))))
            gt_rt = cv2.resize(gt_rt, (env.scene.trav_map_size, env.scene.trav_map_size))

            ######sanity check
            ori_pos = np.array(env.robots[0].get_position())[:2]
            ori = np.array(env.robots[0].get_rpy())
            # cv2.imwrite("cur_gt_rt.png", gt_rt)
            ######

            gt_rt = np.flip(gt_rt, axis=0)
            delta_pos = ori_pos/env.scene.trav_map_resolution
            gt_rt = ndimage.shift(gt_rt, [delta_pos[1], -delta_pos[0]]) #move gt_map [left, down] by delta_pos        
            # cv2.imwrite("gt_rt_shifted.png", gt_rt)
            gt_rt = ndimage.rotate(gt_rt, 90.-ori[2]*180.0/math.pi, reshape=False)# rotate gt_map by theta-90 ccw

            ######sanity check
            # cv2.imwrite("gt_rt_rotated.png", gt_rt)
            ######
            H, W = gt_rt.shape
            local_map = gt_rt[int(H/2-100):int(H/2),int(W/2 - 50):int(W/2 + 50)] / 255
            # cv2.imwrite("local_map.png", local_map * 255)

        elif self.config["scene"] == "gibson" or self.config["scene"] == "mp3d":
            maps_path = "/cvgl/group/Gibson/matterport3d-downsized/v1/"  \
            + env.config['scene_id'] + "/"
            gt_rt = np.array(Image.open(os.path.join(maps_path, "floor_trav_{}.png".format(self.floor_num))))
            gt_rt = cv2.resize(gt_rt, (env.scene.trav_map_size, env.scene.trav_map_size))

            ######sanity check
            ori_pos = np.array(env.robots[0].get_position())[:2]
            ori = np.array(env.robots[0].get_rpy())
            # pos = env.scene.world_to_map(np.array(env.robots[0].get_position())[:2])
            # print("robo pos", pos)
            # gt_rt[pos[0], pos[1]] = 255
            # cv2.imwrite("cur_gt_rt.png", gt_rt)
            ######

            gt_rt = np.flip(gt_rt, axis=0)
            delta_pos = ori_pos/env.scene.trav_map_resolution
            gt_rt = ndimage.shift(gt_rt, [delta_pos[1], -delta_pos[0]]) #move gt_map [left, down] by delta_pos        
            # cv2.imwrite("gt_rt_shifted.png", gt_rt)
            gt_rt = ndimage.rotate(gt_rt, 90.-ori[2]*180.0/math.pi, reshape=False)# rotate gt_map by theta-90 ccw

            ######sanity check
            # cv2.imwrite("gt_rt_rotated.png", gt_rt)
            ######
            H, W = gt_rt.shape
            local_map = gt_rt[int(H/2-100):int(H/2),int(W/2 - 50):int(W/2 + 50)] / 255
            # cv2.imwrite("local_map.png", local_map * 255)
        self.gt_rt = local_map
    def get_room_type_map(self, env):
        self.load_gt_rt_map(env)
        return self.gt_rt

class AudioGoalVRNavTask(PointNavRandomTask):
    """
    Redefine the task (reward functions)
    """
    def __init__(self, env):
        super(AudioGoalVRNavTask, self).__init__(env)

    def reset_agent(self, env):
        super().reset_agent(env)

    def get_task_obs(self, env):
        """
        Get current velocities

        :param env: environment instance
        :return: task-specific observation
        """
        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        task_obs = np.append(linear_velocity, angular_velocity)

        return task_obs

class AudioPointGoalNavTask(AudioGoalNavTask):
    """ AudioGoal, but with target position information in observation space """
    def __init__(self, env):
        super().__init__(env)

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