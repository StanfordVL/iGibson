import gibson2
from gibson2.envs.locomotor_env import NavigationEnv
from time import time
import os
from gibson2.utils.assets_utils import download_assets, download_demo_data
import numpy as np
from gibson2.external.pybullet_tools.utils import control_joints
from gibson2.external.pybullet_tools.utils import get_joint_positions
from gibson2.external.pybullet_tools.utils import get_joint_velocities
from gibson2.external.pybullet_tools.utils import get_max_limits
from gibson2.external.pybullet_tools.utils import get_min_limits
from gibson2.external.pybullet_tools.utils import plan_joint_motion
from gibson2.external.pybullet_tools.utils import link_from_name
from gibson2.external.pybullet_tools.utils import set_joint_positions
from gibson2.external.pybullet_tools.utils import get_sample_fn
from gibson2.external.pybullet_tools.utils import set_base_values_with_z
from gibson2.external.pybullet_tools.utils import get_base_values
from gibson2.external.pybullet_tools.utils import plan_base_motion_2d
from gibson2.utils.utils import rotate_vector_2d, rotate_vector_3d


class MotionPlanningWrapper(object):
    def __init__(self,
                 env: NavigationEnv = None,
                 base_mp_algo: str = 'birrt',
                 optimize_iter: int = 0):
        """
        Get planning related parameters.
        """
        self.env = env
        assert 'occupancy_grid' in self.env.output
        # get planning related parameters from env
        self.robot_id = self.env.robots[0].robot_ids[0]
        # self.mesh_id = self.scene.mesh_body_id
        # mesh id should not be used
        self.map_size = self.env.scene.trav_map_original_size * \
                        self.env.scene.trav_map_default_resolution

        self.grid_resolution = self.env.grid_resolution
        self.occupancy_range = self.env.occupancy_range
        self.robot_footprint_radius = self.env.robot_footprint_radius
        self.robot_footprint_radius_in_map = self.env.robot_footprint_radius_in_map
        self.robot = self.env.robots[0]
        self.base_mp_algo = base_mp_algo
        self.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        self.optimize_iter = optimize_iter

    def plan_base_motion(self, state, goal):
        x, y, theta = goal
        grid = state['occupancy_grid']

        yaw = self.robot.get_rpy()[2]
        half_occupancy_range = self.occupancy_range / 2.0
        robot_position_xy = self.robot.get_position()[:2]
        corners = [
            robot_position_xy + rotate_vector_2d(local_corner, -yaw)
            for local_corner in [
                np.array([half_occupancy_range, half_occupancy_range]),
                np.array([half_occupancy_range, -half_occupancy_range]),
                np.array([-half_occupancy_range, half_occupancy_range]),
                np.array([-half_occupancy_range, -half_occupancy_range]),
            ]
        ]
        path = plan_base_motion_2d(
            self.robot_id,
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            map_2d=grid,
            occupancy_range=self.occupancy_range,
            grid_resolution=self.grid_resolution,
            robot_footprint_radius_in_map=self.robot_footprint_radius_in_map,
            resolutions=self.base_mp_resolutions,
            obstacles=[],
            algorithm=self.base_mp_algo,
            optimize_iter=self.optimize_iter)

        return path
