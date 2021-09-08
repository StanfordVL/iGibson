import json
import logging
import os
import shutil
from collections import defaultdict

import bddl
import numpy as np

import igibson
from igibson.envs.behavior_env import BehaviorEnv
from igibson.metrics.agent import BehaviorRobotMetric
from igibson.metrics.disarrangement import KinematicDisarrangement, LogicalDisarrangement
from igibson.metrics.task import TaskMetric
from igibson.utils.utils import parse_config

logging.getLogger().setLevel(logging.WARNING)


def get_metrics_callbacks():
    metrics = [
        KinematicDisarrangement(),
        LogicalDisarrangement(),
        BehaviorRobotMetric(),
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
        self.split = os.environ["SPLIT"]
        self.output_dir = os.environ["OUTPUT_DIR"]

    def submit(self, agent):
        env_config = parse_config(self.config_file)

        if self.split == "minival":
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

        log_path = os.path.join(self.output_dir, "per_episode_metrics.json")
        summary_log_path = os.path.join(self.output_dir, "aggregated_metrics.json")

        self_reported_log_path = os.path.join(
            self.output_dir, "..", "participant_reported_results", "per_episode_metrics.json"
        )
        self_reported_summary_log_path = os.path.join(
            self.output_dir, "..", "participant_reported_results", "aggregated_metrics.json"
        )
        if os.path.exists(self_reported_log_path):
            shutil.copyfile(self_reported_log_path, log_path)
            print("Per episode eval results copied from self-reported results %s" % log_path)
            with open(self_reported_log_path) as f:
                self_reported_log = json.load(f)
                assert len(self_reported_log) == len(tasks) * 9

        if os.path.exists(self_reported_summary_log_path):
            shutil.copyfile(self_reported_summary_log_path, summary_log_path)
            print("Aggregated eval results copied from self-reported results %s" % summary_log_path)
            return

        episode = 0
        per_episode_metrics = {}
        scene_json = os.path.join(os.path.dirname(bddl.__file__), "../utils", "activity_to_preselected_scenes.json")

        with open(scene_json) as f:
            activity_to_scenes = json.load(f)

        with open(os.path.join(igibson.ig_dataset_path, "metadata", "behavior_activity_statistics.json")) as f:
            activity_metadata = json.load(f)

        for task in tasks:
            assert task in activity_to_scenes
            scenes = sorted(set(activity_to_scenes[tasks[0]]))
            num_scenes = len(scenes)
            human_demo_mean_step = activity_metadata[task]["mean"]
            env_config["max_step"] = human_demo_mean_step * 2  # adjust env_config['max_step'] based on the human
            # demonstration, we give agent 2x steps of average human demonstration across all possible scenes

            assert num_scenes <= 3

            # Evaluate 9 activity instances in the training set for now
            if num_scenes == 3:
                scene_instance_ids = {scenes[0]: range(3), scenes[1]: range(3), scenes[2]: range(3)}
            elif num_scenes == 2:
                scene_instance_ids = {scenes[0]: range(4), scenes[1]: range(5)}
            else:
                scene_instance_ids = {scenes[0]: range(9)}

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

                    metrics_summary["task"] = task
                    per_episode_metrics[episode] = metrics_summary
                    episode += 1
                    env.close()

        with open(log_path, "w+") as f:
            json.dump(per_episode_metrics, f)
        print("Per episode eval results saved to %s" % log_path)

        aggregated_metrics = {}
        success_score = []
        simulator_time = []
        kinematic_disarrangement = []
        logical_disarrangement = []
        distance_navigated = []
        displacement_of_hands = []

        task_to_mean_success_score = defaultdict(list)
        task_scores = []

        for episode, metric in per_episode_metrics.items():
            task_to_mean_success_score[metric["task"]].append(metric["q_score"]["timestep"][-1])

        for task, scores in task_to_mean_success_score.items():
            task_scores.append(np.mean(scores))

        task_scores = sorted(task_scores, reverse=True)

        for episode, metric in per_episode_metrics.items():
            success_score.append(metric["q_score"]["timestep"][-1])
            simulator_time.append(metric["time"]["simulator_time"])
            kinematic_disarrangement.append(metric["kinematic_disarrangement"]["relative"])
            logical_disarrangement.append(metric["logical_disarrangement"]["relative"])
            distance_navigated.append(np.sum(metric["agent_distance"]["timestep"]["body"]))
            displacement_of_hands.append(
                np.sum(metric["grasp_distance"]["timestep"]["left_hand"])
                + np.sum(metric["grasp_distance"]["timestep"]["right_hand"])
            )

        aggregated_metrics["Success Score"] = np.mean(success_score)
        aggregated_metrics["Success Score Top 5"] = np.mean(np.array(task_scores)[:5])
        aggregated_metrics["Simulated Time"] = np.mean(simulator_time)
        aggregated_metrics["Kinematic Disarrangement"] = np.mean(kinematic_disarrangement)
        aggregated_metrics["Logical Disarrangement"] = np.mean(logical_disarrangement)
        aggregated_metrics["Distance Navigated"] = np.mean(distance_navigated)
        aggregated_metrics["Displacement of Hands"] = np.mean(displacement_of_hands)
        with open(summary_log_path, "w+") as f:
            json.dump(aggregated_metrics, f)
        print("Aggregated eval results saved to %s" % summary_log_path)


if __name__ == "__main__":
    challenge = BehaviorChallenge()
    challenge.submit(None)
