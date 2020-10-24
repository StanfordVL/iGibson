from abc import abstractmethod, ABC
from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition
import numpy as np


class ObjectGoal(BaseTerminationCondition):
    def __init__(self, config):
        super(ObjectGoal, self).__init__(config)
        self.success_thresh = self.config.get('success_thresh', 0.05)
        self.goal_class_id = self.config.get('goal_class_id', 255)

    def get_termination(self, env):
        seg = env.simulator.renderer.render_robot_cameras(modes='seg')[
            0][:, :, 0:1]
        seg = np.round(seg * 255.0)
        goal_seg = np.sum(seg == self.goal_class_id)
        goal_pixel_perc = goal_seg / (seg.shape[0] * seg.shape[1])
        done = goal_pixel_perc > self.success_thresh
        success = done
        return done, success
