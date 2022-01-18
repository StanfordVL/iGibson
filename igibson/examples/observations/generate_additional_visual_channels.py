import logging
import os
from sys import platform

import cv2
import numpy as np
import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.vision_utils import randomize_colors, segmentation_to_rgb


def main():
    """
    Example of rendering additional sensor modalities
    Loads Rs_int (interactive) with some objects and and renders depth, normals, semantic and instance segmentation
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = os.path.join(igibson.example_config_path, "turtlebot_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    # Only load a few objects
    # config_data["load_object_categories"] = [
    #     "breakfast_table",
    #     "carpet",
    #     "sofa",
    #     "bottom_cabinet",
    #     "sink",
    #     "stove",
    #     "fridge",
    # ]
    config_data["vertical_fov"] = 90
    config_data["output"] = ["task_obs", "rgb", "depth", "normal", "seg", "ins_seg"]
    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5
    env = iGibsonEnv(config_file=config_data, mode="gui_interactive")
    # Set a better viewing direction
    env.simulator.viewer.initial_pos = [0, 1.1, 1.5]
    env.simulator.viewer.initial_view_direction = [0, 1, 0.1]
    env.simulator.viewer.reset_viewer()

    colors_ss = randomize_colors(MAX_CLASS_COUNT, bright=True)
    colors_is = randomize_colors(MAX_INSTANCE_COUNT, bright=True)

    for j in range(100):
        logging.info("Resetting environment")
        env.reset()
        for i in range(300):
            with Profiler("Environment action step"):
                action = env.action_space.sample()
                state, reward, done, info = env.step([0.1, 0.1])

                depth = state["depth"]
                cv2.imshow("Depth", depth)

                normal = state["normal"]
                cv2.imshow("Normals", cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))

                seg = state["seg"]
                seg_int = np.round(seg[:, :, 0]).astype(np.int32)
                seg_cv = segmentation_to_rgb(seg_int, MAX_CLASS_COUNT, colors=colors_ss)
                cv2.imshow("Semantic Segmentation", seg_cv)

                ins_seg = state["ins_seg"]
                ins_seg_int = np.round(ins_seg[:, :, 0]).astype(np.int32)
                ins_seg_cv = segmentation_to_rgb(ins_seg_int, MAX_INSTANCE_COUNT, colors=colors_is)
                cv2.imshow("Instance Segmentation", ins_seg_cv)

    env.close()


if __name__ == "__main__":
    main()
