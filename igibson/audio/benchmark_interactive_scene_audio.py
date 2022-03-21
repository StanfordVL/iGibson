#!/usr/bin/env python

from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.robots.turtlebot_robot import Turtlebot
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.utils.constants import AVAILABLE_MODALITIES
from igibson.utils.utils import parse_config
from igibson.utils.constants import NamedRenderingPresets
import os
import igibson
import time
import random
import matplotlib.pyplot as plt
from igibson.utils.assets_utils import get_ig_assets_version
from igibson.utils.assets_utils import get_scene_path
from audio_system import AudioSystem
from igibson.objects import cube
import pickle as pkl
import numpy as np


def benchmark_scene(scene_name, optimized=False, import_robot=True, num_sources=0):
    config = parse_config(os.path.join(igibson.root_path, 'test', 'test.yaml'))
    assets_version = get_ig_assets_version()
    print('assets_version', assets_version)
    scene = InteractiveIndoorScene(
        scene_name, texture_randomization=False, object_randomization=False)
    settings = MeshRendererSettings(
        msaa=False, enable_shadow=False, optimized=optimized)
    s = Simulator(mode='headless',
                  image_width=512,
                  image_height=512,
                  device_idx=0,
                  rendering_settings=settings,
                  )
    s.import_ig_scene(scene)


    if import_robot:
        turtlebot = Turtlebot(config)
        s.import_robot(turtlebot)
        if num_sources > 0:
            audioSystem = AudioSystem(s, turtlebot, is_Viewer=False, writeToFile=False, SR = 44100)
            for i in range(num_sources):
                _,source_location = scene.get_random_point_by_room_type("living_room")
                source_location[2] = 1.7
                obj = cube.Cube(pos=source_location, dim=[0.05, 0.05, 0.05], visual_only=False, mass=0.5, color=[255, 0, 0, 1])
                s.import_object(obj)
                obj_id = obj.get_body_id()
                audioSystem.registerSource(obj_id, "440Hz_44100Hz.wav", enabled=True)
                audioSystem.setSourceRepeat(obj_id)

    s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
    fps = []
    physics_fps = []
    render_fps = []
    obj_awake = []
    audio_fps = []
    for i in range(2000):
        # if i % 100 == 0:
        #     scene.randomize_texture()
        start = time.time()
        s.step()
        audioSystem.step()
        if import_robot:
            turtlebot.apply_action(turtlebot.action_space.sample())
            # apply random actions
        physics_audio_end = time.time()
        
        if import_robot:
            _ = s.renderer.render_robot_cameras(modes=('rgb'))
        else:
            _ = s.renderer.render(modes=('rgb'))
        end = time.time()

        #print("Elapsed time: ", end - start)
        print("Render Frequency: ", 1 / (end - physics_audio_end))
        print("Physics and Audio Frequency: ", 1 / (physics_audio_end - start))
        print("Step Frequency: ", 1 / (end - start))
        fps.append(1 / (end - start))
        physics_fps.append(1 / (physics_audio_end - start))
        render_fps.append(1 / (end - physics_audio_end))
        obj_awake.append(s.body_links_awake)
    s.disconnect()
    plt.figure(figsize=(7, 25))

    ax = plt.subplot(6, 1, 1)
    plt.hist(render_fps)
    ax.set_xlabel('Render fps')
    ax.set_title('Scene {} version {}\noptimized {} num_obj {}\n import_robot {}'.format(
        scene_name, assets_version, optimized, scene.get_num_objects(), import_robot))
    ax = plt.subplot(6, 1, 2)
    plt.hist(physics_fps)
    ax.set_xlabel('Physics and Audio fps')
    ax = plt.subplot(6, 1, 3)
    plt.hist(fps)
    ax.set_xlabel('Step fps')
    ax = plt.subplot(6, 1, 4)
    plt.plot(render_fps)
    ax.set_xlabel('Render fps with time')
    ax.set_ylabel('fps')
    ax = plt.subplot(6, 1, 5)
    plt.plot(physics_fps)
    ax.set_xlabel('Physics fps with time, converge to {}'.format(np.mean(physics_fps[-100:])))
    ax.set_ylabel('fps')
    ax = plt.subplot(6, 1, 6)
    plt.plot(obj_awake)
    ax.set_xlabel('Num object links awake, converge to {}'.format(np.mean(obj_awake[-100:])) )

    plt.savefig('scene_benchmark_{}_o_{}_r_{}_n_{}.pdf'.format(
        scene_name, optimized, import_robot, num_sources))

def main():
    benchmark_scene('Rs_int', optimized=True, import_robot=False)
    benchmark_scene('Rs_int', optimized=True, import_robot=True, num_sources=0)
    benchmark_scene('Rs_int', optimized=True, import_robot=True, num_sources=1)
    benchmark_scene('Rs_int', optimized=True, import_robot=True, num_sources=2)
    benchmark_scene('Rs_int', optimized=True, import_robot=True, num_sources=10)
    

    # scenes = ["Beechwood_0_int",
    #           "Beechwood_1_int",
    #           "Benevolence_0_int",
    #           "Benevolence_1_int",
    #           "Benevolence_2_int",
    #           "Ihlen_0_int",
    #           "Ihlen_1_int",
    #           "Merom_0_int",
    #           "Merom_1_int",
    #           "Pomaria_0_int",
    #           "Pomaria_1_int",
    #           "Pomaria_2_int",
    #           "Rs_int",
    #           "Wainscott_0_int",
    #           "Wainscott_1_int"]

    # for scene in scenes:
    #     benchmark_scene(scene, True)

if __name__ == "__main__":
    main()
