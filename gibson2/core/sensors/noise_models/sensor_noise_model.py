# Code adopted and modified from https://github.com/facebookresearch/habitat-sim/blob/master/habitat_sim/sensors/noise_models/sensor_noise_model.py
import abc


class SensorNoiseModel(abc.ABC):
    r"""Base class for all sensor noise models
    """

    @abc.abstractmethod
    def apply(self, sensor_observation):
        r"""Applies the noise model to the sensor observation
        :param sensor_observation: The clean sensor observation.
            Should not be modified.
        :return: The sensor observation with noise applied.
        """
        pass

    def __call__(self, sensor_observation):
        r"""Alias of `apply()`
        """
        return self.apply(sensor_observation)
