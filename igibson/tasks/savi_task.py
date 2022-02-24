import logging

import numpy as np
import pybullet as p

from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.objects import cube
from igibson.utils.utils import l2_distance, restoreState

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

class SAViTask(PointNavRandomTask):
    # reward function
    def __init__(self, env):
        super().__init__(env)
        self.reward_funcions = [
            PotentialReward(self.config), # geodesic distance, potential_reward_weight
            PointGoalReward(self.config), # success_reward
            CollisionReward(self.config),
            TimeReward(self.config), # time_reward_weight
        ]
        
    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        max_trials = 100
        dist = 0.0
        for _ in range(max_trials):
            cat = random.choice(CATEGORIES)
            scene_files = glob.glob(os.path.join(igibson.ig_dataset_path, "scenes", env.scene_id, f"urdf/{env.scene_id}.urdf"), 
                                   recursive=True)
            sf = scene_files[0]
            tree = ET.parse(sf)
            links = tree.findall(".//link[@category='%s']" % cat)
            if len(links) == 0:
                continue
            link = random.choice(links)
            joint_name = "j_"+link.attrib["name"]
            joint = tree.findall(".//joint[@name='%s']" % joint_name)
            target_pos = [float(i) for i in joint[0].find('origin').attrib["xyz"].split()]
            target_pos = np.array(target_pos)
            
            if env.scene.build_graph:
                _, dist = env.scene.get_shortest_path(
                    self.floor_num,
                    initial_pos[:2],
                    target_pos[:2], entire_path=False)
            else:
                dist = l2_distance(initial_pos, target_pos)
            if self.target_dist_min < dist < self.target_dist_max:
                env.cat = cat
                break
        if not (self.target_dist_min < dist < self.target_dist_max):
            print("WARNING: Failed to sample initial and target positions")
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        env.initial_pos = initial_pos
        env.initial_orn = initial_orn
        return initial_pos, initial_orn, target_pos

    def reset_scene(self, env):
        super().reset_scene(env)
        source_location = self.target_pos
        self.audio_obj = cube.Cube(pos=source_location, dim=[0.05, 0.05, 0.05], 
                                    visual_only=False, 
                                    mass=0.5, color=[255, 0, 0, 1]) # pos initialized with default
        env.simulator.import_object(self.audio_obj)
        self.audio_obj_id = self.audio_obj.get_body_id()[0]
        # for savi
        if train:
            env.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'] \
                                            +"/train/"+self.cat+".wav", enabled=True)
        else:
            env.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'] \
                                            +"/val/"+self.cat+".wav", enabled=True)    
#             self.audio_system.setSourceRepeat(self.audio_obj_id)
        env.simulator.attachAudioSystem(env.audio_system)

        env.audio_system.step()