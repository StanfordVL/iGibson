"""
BEHAVIOR demo batch analysis script
"""

import argparse
import os
import json
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

import tasknet
from gibson2.examples.behavior.behavior_demo_replay import replay_demo
from gibson2.metrics.disarrangement import KinematicDisarrangement, LogicalDisarrangement
from gibson2.metrics.agent import AgentMetric
from gibson2.metrics.gaze import GazeMetric
from gibson2.metrics.task import TaskMetric

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run and collect an ATUS demo')
    parser.add_argument('--log_manifest', type=str,
                        help='Plain text file consisting of list of demos to replay')
    parser.add_argument('--frame_save_directory', type=str, default=None,
                        help='Path to save replay videos')
    return parser.parse_args()

def main():
    args = parse_args()

    logger = logging.getLogger()
    logger.disabled = True

    tasknet.set_backend("iGibson")

    # We use vr mode if a frame save directory is passed in
    use_vr_mode = args.frame_save_directory != ''

    demo_list = pd.read_csv(args.log_manifest)

    for idx, demo in enumerate(demo_list['demos']):

        if "replay" in demo:
            continue

        print("Replaying demo: {}, {} out of {}".format(demo, idx, len(demo_list['demos'])))

        kinematic_derangement = KinematicDisarrangement()
        logical_derangement = LogicalDisarrangement()
        agent_metric = AgentMetric()
        gaze = GazeMetric()
        task = TaskMetric()
        on_start_callbacks = [gaze.start_callback]
        on_step_callbacks = [
            kinematic_derangement.step_callback,
            logical_derangement.step_callback,
            agent_metric.step_callback,
            gaze.step_callback,
            task.step_callback
        ]

        base_path = os.path.splitext(Path(demo))[0]
        replay_path = base_path + "_replay.hdf5"
        log_path = base_path + ".json"

        if args.frame_save_directory:
            curr_frame_save_path = os.path.join(args.frame_save_directory, '{}.mp4'.format(os.path.split(demo)[1][:-5]))
        else:
            curr_frame_save_path = None

        try:
            demo_statistics = replay_demo(
                    in_log_path=demo,
                    out_log_path=replay_path,
                    frame_save_path=curr_frame_save_path,
                    start_callbacks=on_start_callbacks,
                    step_callbacks=on_step_callbacks,
                    end_callbacks = [logical_derangement.compute_relative_disarrangement],
                    mode="vr",
                    verbose=False
                )
            demo_statistics["failed"] = False
            demo_statistics["filename"] = Path(demo).name

            for callback in [kinematic_derangement, logical_derangement, agent_metric, gaze, task]:
                demo_statistics.update(callback.gather_results())

        except Exception as e:
            print("Demo failed withe error: ", e)
            demo_statistics = {
                "demo_id": Path(demo).name,
                "failed": True,
                "failure_reason": str(e)
            }

        with open(log_path, "w") as file:
            json.dump(demo_statistics, file)

if __name__ == "__main__":
    main()
