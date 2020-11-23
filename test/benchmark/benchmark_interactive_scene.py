#!/usr/bin/env python

from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.utils.utils import parse_config

import os
import gibson2
import time
import random
import matplotlib.pyplot as plt
from gibson2.utils.assets_utils import get_ig_assets_version
from gibson2.utils.assets_utils import get_scene_path


def benchmark_scene(scene_name, optimized=False, first_n=200):
    config = parse_config(os.path.join(gibson2.root_path, '../test/test.yaml'))
    assets_version = get_ig_assets_version()
    print('assets_version', assets_version)
    scene = InteractiveIndoorScene(
        scene_name, texture_randomization=False, object_randomization=False)
    settings = MeshRendererSettings(
        msaa=False, enable_shadow=False, optimized=optimized)
    # scene._set_first_n_objects(first_n)
    s = Simulator(mode='headless',
                  image_width=512,
                  image_height=512,
                  device_idx=0,
                  rendering_settings=settings,
                  physics_timestep=1/240.0
                  )
    s.import_ig_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
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
    plt.figure(figsize=(7, 25))

    ax = plt.subplot(5, 1, 1)
    plt.hist(render_fps)
    ax.set_xlabel('Render fps')
    ax.set_title('Scene {} version {}\noptimized {} num_obj {}/{}'.format(
        scene_name, assets_version, optimized, first_n, scene.get_num_objects()))
    ax = plt.subplot(5, 1, 2)
    plt.hist(physics_fps)
    ax.set_xlabel('Physics fps')
    ax = plt.subplot(5, 1, 3)
    plt.hist(fps)
    ax.set_xlabel('Step fps')
    ax = plt.subplot(5, 1, 4)
    plt.plot(render_fps)
    ax.set_xlabel('Render fps with time')
    ax.set_ylabel('fps')
    ax = plt.subplot(5, 1, 5)
    plt.plot(physics_fps)
    ax.set_xlabel('Physics fps with time')
    ax.set_ylabel('fps')
    plt.savefig('scene_benchmark_{}_o_{}_{}.pdf'.format(
        scene_name, optimized, first_n))


def main():
    # for i in [0, 1,5,10,20,30,40,50,60,70]:
    benchmark_scene('Rs_int', True)
    #benchmark_scene('Rs_int', False)
    #benchmark_scene('Wainscott_0_int', True)
    #benchmark_scene('Wainscott_0_int', False)


if __name__ == "__main__":
    main()
