#!/usr/bin/env python
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_ig_scene_non_colliding_seeds
import os
import gibson2
import time
import random

## human interaction demo


def test_import_igsdf():
    seeds = get_ig_scene_non_colliding_seeds('Rs')
    random.seed(seeds[0])
    config = parse_config(os.path.join(gibson2.root_path, '../test/test.yaml'))
    hdr_texture = os.path.join(gibson2.ig_dataset_path, 'background', 'photo_studio_01_2k.hdr')
    scene = InteractiveIndoorScene('Benevolence_1', texture_randomization=False, object_randomization=True)
    s = Simulator(mode='iggui', image_width=960,
                  image_height=720, device_idx=0, env_texture_filename=hdr_texture)
    s.import_ig_scene(scene)
    s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
    for i in range(10000):
        start = time.time()
        s.step()
        end = time.time()
        print("Elapsed time: ", end - start)
        print("Frequency: ", 1 / (end - start))
    s.disconnect()
    print("end")


def main():
    test_import_igsdf()


if __name__ == "__main__":
    main()