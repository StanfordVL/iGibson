import numpy as np

from igibson.metrics.metric_base import MetricBase


class TaskMetric(MetricBase):
    def __init__(self):
        self.satisfied_predicates = []
        self.q_score = []
        self.timesteps = 0

    def start_callback(self, igbhvr_act_instance, _):
        self.render_timestep = igbhvr_act_instance.simulator.render_timestep

    def step_callback(self, igbhvr_act_inst, _):
        self.timesteps += 1
        self.satisfied_predicates.append(igbhvr_act_inst.current_goal_status)
        candidate_q_score = []
        for option in igbhvr_act_inst.ground_goal_state_options:
            predicate_truth_values = []
            for predicate in option:
                predicate_truth_values.append(predicate.evaluate())
            candidate_q_score.append(np.mean(predicate_truth_values))
        self.q_score.append(np.max(candidate_q_score))

    def gather_results(self):
        return {
            "satisfied_predicates": {
                "timestep": self.satisfied_predicates,
            },
            "q_score": {
                "timestep": self.q_score,
            },
            "time": {
                "simulator_steps": self.timesteps,
                "simulator_time": self.timesteps * self.render_timestep,
            },
        }
