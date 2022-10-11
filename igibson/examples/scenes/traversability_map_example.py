import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import igibson

from igibson.utils.assets_utils import get_scene_path


def main(selection="user", headless=False, short_exec=False):
    """
    Traversable map demo
    Loads the floor plan and obstacles for the Rs scene, and overlays them in a visual figure such that the
    highlighted area reflects the traversable (free-space) area
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    scene_id = "Ihlen_1_int"
    path = os.path.join("/data/ig_data/ig_dataset/scenes", scene_id, "layout")

    for f in range(1):
        trav_map = cv2.imread(os.path.join(path, "floor_trav_no_door_{}.png".format(f)))
        if not headless:
            plt.figure(f, figsize=(12, 12))
            plt.imshow(cv2.cvtColor(trav_map, cv2.COLOR_RGB2BGR))
            plt.title("Traversable area of {} scene".format(scene_id))

    if not headless:
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
