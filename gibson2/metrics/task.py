import numpy as np

from gibson2.metrics.metric_base import MetricBase


class TaskMetric(MetricBase):
    def __init__(self):
        self.satisfied_predicates = []
        self.q_score = []

    def step_callback(self, igtn_task, _):
        self.satisfied_predicates.append(igtn_task.current_goal_status)
        candidate_q_score = []
        for option in igtn_task.ground_goal_state_options:
            predicate_truth_values = []
            for predicate in option:
                predicate_truth_values.append(predicate.evaluate())
            candidate_q_score.append(np.mean(predicate_truth_values))
        self.q_score.append(np.max(candidate_q_score))

    def gather_results(self):
        return {
            "satisfied_predicates": {
                "absolute": self.satisfied_predicates,
            },
            "q_score": {
                "absolute": self.q_score,
            },
        }
