import numpy as np
import pybullet as p
from igibson.envs.igibson_env import iGibsonEnv
from igibson.external.pybullet_tools.utils import transform_obj_file


COUNT_CURR_EPISODE = 0


class AVNavRLEnv(iGibsonEnv):
    """
    Redefine the environment (robot, task, dataset)
    """
    def __init__(self, config_file, mode, scene_splits, device_idx, trail=0):
        scene_id = np.random.choice(scene_splits)
        super().__init__(config_file, scene_id, mode, device_idx=device_idx,trail=trail)
        self.config["scene_splits"] = scene_splits
    
    def load(self):
        """
        Load scene, robot, and environment
        """
        super().load()
        
        if self.config['scene'] == 'igibson':
            carpets = []
            if "carpet" in self.scene.objects_by_category.keys():
                carpets = self.scene.objects_by_category["carpet"]
            for carpet in carpets:
                for robot_link_id in range(p.getNumJoints(self.robots[0].get_body_ids()[0])):
                    for i in range(len(carpet.get_body_ids())):
                        p.setCollisionFilterPair(carpet.get_body_ids()[i], 
                                                 self.robots[0].get_body_ids()[0], -1, robot_link_id, 0)
