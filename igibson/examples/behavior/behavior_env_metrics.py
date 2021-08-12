import argparse
import json
import os
import parser

import bddl

import igibson
from igibson.envs.behavior_env import BehaviorEnv
from igibson.metrics.agent import AgentMetric
from igibson.metrics.disarrangement import KinematicDisarrangement, LogicalDisarrangement
from igibson.metrics.task import TaskMetric


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


if __name__ == "__main__":

    bddl.set_backend("iGibson")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        default=os.path.join(igibson.root_path, "examples", "configs", "behavior_full_observability.yaml"),
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "simple", "gui", "iggui", "pbgui"],
        default="simple",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = BehaviorEnv(
        config_file=args.config,
        mode=args.mode,
        action_timestep=1.0 / 10.0,
        physics_timestep=1.0 / 120.0,
    )

    start_callbacks, step_callbacks, end_callbacks, data_callbacks = get_metrics_callbacks()

    per_episode_metrics = {}
    for callback in start_callbacks:
        callback(env.task, None)

    for episode in range(10):
        env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
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

    log_path = "my_igibson_run.json"

    with open(log_path, "w") as file:
        json.dump(per_episode_metrics, file)

    env.close()
