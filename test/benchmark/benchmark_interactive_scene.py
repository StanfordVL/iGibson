#!/usr/bin/env python

from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
import os
import gibson2
import time
import random
import matplotlib.pyplot as plt
from gibson2.utils.assets_utils import get_ig_assets_version

def benchmark_scene(scene_name):
    assets_version = get_ig_assets_version()
    print('assets_version', assets_version)
    random.seed(0)
    scene = InteractiveIndoorScene(scene_name, texture_randomization=False)
    s = Simulator(mode='headless', image_width=512,
                  image_height=512, device_idx=0)
    s.import_ig_scene(scene)

    s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
    fps = []
    for i in range(500):
        # if i % 100 == 0:
        #     scene.randomize_texture()
        start = time.time()
        s.step()
        end = time.time()
        print("Elapsed time: ", end - start)
        print("Frequency: ", 1 / (end - start))
        fps.append(1 / (end - start))
    s.disconnect()

    plt.figure()
    plt.hist(fps)
    plt.xlabel('fps')
    plt.title('Scene {} version {}'.format(scene_name, assets_version))
    plt.savefig('scene_benchmark_{}.pdf'.format(scene_name))
    print("end")


def main():
    benchmark_scene('Rs')
    benchmark_scene('Wainscott_0')


if __name__ == "__main__":
    main()
