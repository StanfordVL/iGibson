from gibson2.sensors.sensor_base import BaseSensor

import os
import gibson2

class AudioSensor(BaseSensor):

    def __init__(self, env):
        super(AudioSensor, self).__init__(env)

    def get_obs(self, env):
        """
        Get audio sensor reading

        :return: audio sensor reading
        """
        return env.simulator.audio_system.current_output
