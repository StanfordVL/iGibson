from igibson.sensors.sensor_base import BaseSensor
from igibson.sensors.dropout_sensor_noise import DropoutSensorNoise

import numpy as np
import os
import igibson
from collections import OrderedDict


class BumpSensor(BaseSensor):
    """
    Bump sensor
    """

    def __init__(self, env):
        super(BumpSensor, self).__init__(env)

    def get_obs(self, env):
        """
        Get Bump sensor reading

        :return: Bump sensor reading
        """
        has_collision = float(len(env.collision_links) > 0)
        return has_collision
