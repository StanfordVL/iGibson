#!/usr/bin/env python

from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.utils.utils import parse_config

import os
import gibson2
import time
import random
import matplotlib.pyplot as plt
from gibson2.utils.assets_utils import get_ig_assets_version
from gibson2.utils.assets_utils import get_scene_path
from gibson2.utils.assets_utils import get_ig_scene_non_colliding_seeds

def benchmark_scene(scene_name, optimized=False):
    config = parse_config(os.path.join(gibson2.root_path, '../test/test.yaml'))
    assets_version = get_ig_assets_version()
    print('assets_version', assets_version)
    seeds = get_ig_scene_non_colliding_seeds(scene_name)
    random.seed(seeds[0])
    scene = InteractiveIndoorScene(scene_name, texture_randomization=False)
    s = Simulator(mode='headless',
                  image_width=512,
                  image_height=512,
                  device_idx=0,
                  optimized_renderer=optimized,
                  env_texture_filename=os.path.join(get_scene_path('Rs'), 'lighting', 'probes', 'probe_00.hdr'))
    s.import_ig_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
    if optimized:
        s.optimize_vertex_and_texture()
    fps = []
    physics_fps = []
    render_fps = []
    for i in range(5000):
        # if i % 100 == 0:
        #     scene.randomize_texture()
        start = time.time()
        s.step()
        physics_end = time.time()
        _ = s.renderer.render_robot_cameras(modes=('rgb'))
        end = time.time()

        #print("Elapsed time: ", end - start)
        print("Render Frequency: ", 1 / (end - physics_end))
        print("Physics Frequency: ", 1 / (physics_end - start))
        print("Step Frequency: ", 1 / (end - start))
        fps.append(1 / (end - start))
        physics_fps.append(1 / (physics_end - start))
        render_fps.append(1 / (end - physics_end))

    s.disconnect()
    plt.figure(figsize=(7,15))

    ax = plt.subplot(3, 1, 1)
    plt.hist(render_fps)
    ax.set_xlabel('Render fps')
    ax.set_title('Scene {} version {}\noptimized {}'.format(scene_name, assets_version, optimized))
    ax = plt.subplot(3, 1, 2)
    plt.hist(physics_fps)
    ax.set_xlabel('Physics fps')
    ax = plt.subplot(3, 1, 3)
    plt.hist(fps)
    ax.set_xlabel('Step fps')
    plt.savefig('scene_benchmark_{}_o_{}.pdf'.format(scene_name, optimized))


def main():
    benchmark_scene('Rs', True)
    benchmark_scene('Rs', False)
    #benchmark_scene('Wainscott_0', True)
    #benchmark_scene('Wainscott_0', False)


if __name__ == "__main__":
    main()
