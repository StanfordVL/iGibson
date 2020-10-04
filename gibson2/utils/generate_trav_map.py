#!/usr/bin/env python

from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_ig_scene_path
from gibson2.utils.map_utils import gen_trav_map
import os
import gibson2

import time
import random
import sys

"""
script to generate all traversability maps:

for file in ../../gibson2/ig_dataset/scenes/*
  python generate_trav_map.py $(basename $file)

"""

def generate_trav_map(scene_name):
    random.seed(0)
    scene = InteractiveIndoorScene(scene_name, texture_randomization=False)
    s = Simulator(mode='headless', image_width=512,
                  image_height=512, device_idx=0)
    s.import_ig_scene(scene)
    
    for i in range(20):
        s.step()

    vertices_info, faces_info = s.renderer.dump()
    gen_trav_map(vertices_info, faces_info, output_folder=os.path.join(get_ig_scene_path(scene_name), 'layout'))


def main():
    generate_trav_map(sys.argv[1])


if __name__ == "__main__":
    main()
