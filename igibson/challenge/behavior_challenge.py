import json
import logging
import os

import bddl
import numpy as np

from igibson.envs.behavior_env import BehaviorEnv
from igibson.metrics.agent import AgentMetric
from igibson.metrics.disarrangement import KinematicDisarrangement, LogicalDisarrangement
from igibson.metrics.task import TaskMetric
from igibson.utils.utils import parse_config

logging.getLogger().setLevel(logging.WARNING)


def get_metrics_callbacks():
    metrics = [
        KinematicDisarrangement(),
        LogicalDisarrangement(),
        AgentMetric(),
        TaskMetric(),
    ]

    return (
        [metric.start_callback for metric in metrics],
        [metric.step_callback for metric in metrics],
        [metric.end_callback for metric in metrics],
        [metric.gather_results for metric in metrics],
    )


class BehaviorChallenge(object):
    def __init__(self):
        self.config_file = os.environ["CONFIG_FILE"]
        self.phase = os.environ["PHASE"]

    def submit(self, agent):
        env_config = parse_config(self.config_file)

        per_episode_metrics = {}

        if self.phase == "minival":
            # Only eval one activity in the config file
            tasks = [env_config["task"]]
        else:
            # Eval all 100 activities
            tasks = sorted(
                [
                    item
                    for item in os.listdir(os.path.join(os.path.dirname(bddl.__file__), "activity_conditions"))
                    if item != "domain_igibson.bddl"
                ]
            )
            assert len(tasks) == 100

        episode = 0
        per_episode_metrics = {}
        scene_json = os.path.join(os.path.dirname(bddl.__file__), "../utils", "activity_to_preselected_scenes.json")

        with open(scene_json) as f:
            activity_to_scenes = json.load(f)

        for task in tasks:
            assert task in activity_to_scenes
            scenes = sorted(set(activity_to_scenes[tasks[0]]))
            num_scenes = len(scenes)
            assert num_scenes <= 3

            # Evaluate 9 activity instances in the training set for now
            if num_scenes == 3:
                scene_instance_ids = {scenes[0]: range(1), scenes[1]: range(0), scenes[2]: range(0)}
            elif num_scenes == 2:
                scene_instance_ids = {scenes[0]: range(4), scenes[1]: range(5)}
            else:
                scene_instance_ids = {scenes[0]: range(9)}

            # TODO: adjust env_config['episode_length'] based on the activity
            for scene_id, instance_ids in scene_instance_ids.items():
                env_config["scene_id"] = scene_id
                for instance_id in instance_ids:
                    env = BehaviorEnv(
                        config_file=env_config,
                        mode="all",
                        action_timestep=1.0 / 30.0,
                        physics_timestep=1.0 / 120.0,
                        instance_id=instance_id,
                    )
                    start_callbacks, step_callbacks, end_callbacks, data_callbacks = get_metrics_callbacks()
                    for callback in start_callbacks:
                        callback(env.task, None)
                    agent.reset()
                    state = env.reset()
                    while True:
                        action = agent.act(state)
                        state, reward, done, info = env.step(action)
                        for callback in step_callbacks:
                            callback(env.task, None)
                        if done:
                            break
                    for callback in end_callbacks:
                        callback(env.task, None)
                    metrics_summary = {}
                    for callback in data_callbacks:
                        metrics_summary.update(callback())
                    per_episode_metrics[episode] = metrics_summary
                    episode += 1
                    env.close()

        log_path = "eval.json"
        with open(log_path, "w+") as f:
            json.dump(per_episode_metrics, f)
        print("Eval results saved to %s" % log_path)


if __name__ == "__main__":
    challenge = BehaviorChallenge()
    challenge.submit(None)
