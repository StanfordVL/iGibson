#!/usr/bin/env python

from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.utils import parse_config
import os
import igibson
import time
import random


def test_import_igsdf():
    scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=False, object_randomization=False)
    s = Simulator(mode='headless', image_width=512,
                  image_height=512, device_idx=0)
    s.import_ig_scene(scene)

    s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
    for i in range(10):
        # if i % 100 == 0:
        #     scene.randomize_texture()
        start = time.time()
        s.step()
        end = time.time()
        print("Elapsed time: ", end - start)
        print("Frequency: ", 1 / (end - start))

    s.disconnect()


def main():
    test_import_igsdf()


if __name__ == "__main__":
    main()
