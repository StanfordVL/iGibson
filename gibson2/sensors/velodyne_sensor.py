from gibson2.sensors.sensor_base import BaseSensor
from gibson2.sensors.dropout_sensor_noise import DropoutSensorNoise

import numpy as np
import os
import gibson2
from collections import OrderedDict


class VelodyneSensor(BaseSensor):
    """
    16-beam Velodyne LiDAR sensor
    """

    def __init__(self, env):
        super(VelodyneSensor, self).__init__(env)

    def get_obs(self, env):
        """
        Get velodyne LiDAR sensor reading

        :return: velodyne sensor reading
        """
        return env.simulator.renderer.get_lidar_all()
