from igibson.sensors.sensor_base import BaseSensor


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
