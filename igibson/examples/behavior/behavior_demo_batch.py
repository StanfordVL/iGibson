"""
BEHAVIOR demo batch analysis script
"""

import json
import logging
import os
from pathlib import Path

import bddl
import pandas as pd

from igibson.examples.behavior.behavior_demo_replay import replay_demo


def behavior_demo_batch(
    demo_root, log_manifest, out_dir, get_callbacks_callback, skip_existing=True, save_frames=False
):
    """
    Execute replay analysis functions (provided through callbacks) on a batch of BEHAVIOR demos.

    @param demo_root: Directory containing the demo files listed in the manifests.
    @param log_manifest: The manifest file containing list of BEHAVIOR demos to batch over.
    @param out_dir: Directory to store results in.
    @param get_callbacks_callback: A function that will be called for each demo that needs to return
        a four-tuple: (start_callbacks, step_callbacks, end_callbacks, data_callbacks). Each of the
        the first three callback function sets need to be compatible with the behavior_demo_replay
        API and will be used for this purpose for that particular demo. The data callbacks should
        take no arguments and return a dictionary to be included in the demo's replay data that will
        be saved in the end.
    @param skip_existing: Whether demos with existing output logs should be skipped.
    @param save_frames: Whether the demo's frames should be saved alongside statistics.
    """
    logger = logging.getLogger()
    logger.disabled = True

    bddl.set_backend("iGibson")

    demo_list = pd.read_csv(log_manifest)

    for idx, demo in enumerate(demo_list["demos"]):
        if "replay" in demo:
            continue

        demo_name = os.path.splitext(demo)[0]
        demo_path = os.path.join(demo_root, demo)
        replay_path = os.path.join(out_dir, demo_name + "_replay.hdf5")
        log_path = os.path.join(out_dir, demo_name + ".json")

        if skip_existing and os.path.exists(log_path):
            print("Skipping existing demo: {}, {} out of {}".format(demo, idx, len(demo_list["demos"])))
            continue

        print("Replaying demo: {}, {} out of {}".format(demo, idx, len(demo_list["demos"])))

        curr_frame_save_path = None
        if save_frames:
            curr_frame_save_path = os.path.join(out_dir, demo_name + ".mp4")

        try:
            start_callbacks, step_callbacks, end_callbacks, data_callbacks = get_callbacks_callback()
            demo_information = replay_demo(
                in_log_path=demo_path,
                out_log_path=replay_path,
                frame_save_path=curr_frame_save_path,
                start_callbacks=start_callbacks,
                step_callbacks=step_callbacks,
                end_callbacks=end_callbacks,
                mode="headless",
                verbose=False,
            )
            demo_information["failed"] = False
            demo_information["filename"] = Path(demo).name

            for callback in data_callbacks:
                demo_information.update(callback())

        except Exception as e:
            print("Demo failed withe error: ", e)
            demo_information = {"demo_id": Path(demo).name, "failed": True, "failure_reason": str(e)}

        with open(log_path, "w") as file:
            json.dump(demo_information, file)
