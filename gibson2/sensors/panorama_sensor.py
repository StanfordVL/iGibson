from gibson2.sensors.sensor_base import BaseSensor
from gibson2.sensors.dropout_sensor_noise import DropoutSensorNoise

import numpy as np
import os
import gibson2
from collections import OrderedDict


class PanoramaSensor(BaseSensor):
    """
    Panorama camera sensor
    """

    # TODO: warning, this sensor is extremely slow

    def __init__(self, env, modalities):
        super(PanoramaSensor, self).__init__(env)
        self.modalities = modalities

    def get_obs(self, env):
        """
        Get Panorama camera sensor reading

        :return: equirectangular images
        """
        images = []
        for modaility in self.modalities:
            images.append(env.simulator.renderer.get_equi(mode=modaility, use_robot_camera=True, fixed_orientation=True))

        return images