from gibson2.sensors.sensor_base import BaseSensor
from gibson2.sensors.dropout_sensor_noise import DropoutSensorNoise

import numpy as np
from transforms3d.quaternions import quat2mat
import pybullet as p


class ScanSensor(BaseSensor):
    def __init__(self, env):
        super(ScanSensor, self).__init__(env)
        self.scan_noise_rate = self.config.get('scan_noise_rate', 0.0)
        self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
        self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
        assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
        self.laser_linear_range = self.config.get(
            'laser_linear_range', 10.0)
        self.laser_angular_range = self.config.get(
            'laser_angular_range', 180.0)
        self.min_laser_dist = self.config.get('min_laser_dist', 0.05)
        self.laser_link_name = self.config.get(
            'laser_link_name', 'scan_link')
        self.noise_model = DropoutSensorNoise(env)
        self.noise_model.set_noise_rate(self.scan_noise_rate)
        self.noise_model.set_noise_value(1.0)

    def get_obs(self, env):
        """
        :return: LiDAR sensor reading, normalized to [0.0, 1.0]
        """
        laser_angular_half_range = self.laser_angular_range / 2.0
        if self.laser_link_name not in env.robots[0].parts:
            raise Exception('Trying to simulate LiDAR sensor, but laser_link_name cannot be found in the robot URDF file. Please add a link named laser_link_name at the intended laser pose. Feel free to check out assets/models/turtlebot/turtlebot.urdf and examples/configs/turtlebot_p2p_nav.yaml for examples.')
        laser_pose = env.robots[0].parts[self.laser_link_name].get_pose()
        angle = np.arange(
            -laser_angular_half_range / 180 * np.pi,
            laser_angular_half_range / 180 * np.pi,
            self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
        unit_vector_local = np.array(
            [[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
        transform_matrix = quat2mat(
            [laser_pose[6], laser_pose[3], laser_pose[4], laser_pose[5]])  # [x, y, z, w]
        unit_vector_world = transform_matrix.dot(unit_vector_local.T).T

        start_pose = np.tile(laser_pose[:3], (self.n_horizontal_rays, 1))
        start_pose += unit_vector_world * self.min_laser_dist
        end_pose = laser_pose[:3] + unit_vector_world * self.laser_linear_range
        results = p.rayTestBatch(start_pose, end_pose, 6)  # numThreads = 6

        # hit fraction = [0.0, 1.0] of self.laser_linear_range
        hit_fraction = np.array([item[2] for item in results])
        hit_fraction = self.noise_model.add_noise(hit_fraction)
        scan = np.expand_dims(hit_fraction, 1)
        return scan
