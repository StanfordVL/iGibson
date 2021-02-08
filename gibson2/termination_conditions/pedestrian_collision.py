from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition
from gibson2.utils.utils import l2_distance
import time


class PedestrianCollision(BaseTerminationCondition):
    """
    PedestrianCollision used for navigation tasks
    Episode terminates if the robot has collided with any pedestrian
    """

    def __init__(self, config):
        super(PedestrianCollision, self).__init__(config)
        self.pedestrian_collision_threshold = self.config.get(
            'pedestrian_collision_threshold', 0.3)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if the robot has collided more than self.max_collisions_allowed times

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        success = False
        done = False
        robot_pos = env.robots[0].get_position()[:2]
        for ped in task.pedestrians:
            ped_pos = ped.get_position()[:2]
            if l2_distance(robot_pos, ped_pos) < self.pedestrian_collision_threshold:
                done = True
                break
        return done, success
