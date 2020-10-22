from abc import abstractmethod, ABC
from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene


class OutOfBound(BaseTerminationCondition):
    def __init__(self, config):
        super(OutOfBound, self).__init__(config)
        self.fall_off_thresh = self.config.get(
            'fall_off_thresh', 0.03)

    def get_termination(self, env):
        done = False
        # fall off the cliff of valid region
        if isinstance(env.scene, InteractiveIndoorScene):
            robot_z = env.robots[0].get_position()[2]
            if robot_z < (env.scene.get_floor_height() - self.fall_off_thresh):
                done = True
        success = False
        return done, success
