#!/usr/bin/env python

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

import igibson
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots.fetch import Fetch
from igibson.robots.turtlebot_robot import Turtlebot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_assets_version
from igibson.utils.utils import parse_config


def benchmark_scene(scene_name, optimized=False, import_robot=True, import_ycb=False):
    config = parse_config(os.path.join(igibson.root_path, "../tests/test.yaml"))
    assets_version = get_ig_assets_version()
    print("assets_version", assets_version)
    if import_ycb:
        scene = InteractiveIndoorScene(
            scene_name,
            ignore_visual_shape=True,
            load_object_categories=["countertop", "coffee_table", "fridge"],
        )
    else:
        scene = InteractiveIndoorScene(
            scene_name,
            ignore_visual_shape=True,
        )
    settings = MeshRendererSettings(msaa=False, enable_shadow=False, optimized=optimized)
    s = Simulator(
        mode="headless",
        image_width=128,
        image_height=128,
        device_idx=0,
        rendering_settings=settings,
        render_timestep=1 / 30.0,
        physics_timestep=1 / 120.0,
    )
    start = time.time()
    s.import_scene(scene)

    if import_robot:
        robot = Fetch(config)
        s.import_robot(robot)
        robot.set_position([1000, 1000, 1000])

    if import_ycb:
        for i in range(5):
            for j in range(3):
                obj = YCBObject("003_cracker_box")
                s.import_object(obj)
                obj.set_position([-0.7 + i * 0.1, -1.5 + 0.3 * j, 0.53])
                p.changeDynamics(obj.get_body_id(), -1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)

    s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
    fps = []
    physics_fps = []
    render_fps = []
    obj_awake = []
    for i in range(2000):
        start = time.time()
        s.step()
        if import_robot:
            # apply random actions
            action = robot.action_space.sample()
            action[:] = 0
            robot.apply_action(action)

        physics_end = time.time()
        if import_robot:
            _ = s.renderer.render_robot_cameras(modes=("rgb"))
        else:
            _ = s.renderer.render(modes=("rgb"))
        end = time.time()

        # print("Elapsed time: ", end - start)
        print("Render Frequency: ", 1 / (end - physics_end))
        print("Physics Frequency: ", 1 / (physics_end - start))
        print("Step Frequency: ", 1 / (end - start))
        fps.append(1 / (end - start))
        physics_fps.append(1 / (physics_end - start))
        render_fps.append(1 / (end - physics_end))
        obj_awake.append(s.body_links_awake)

    s.disconnect()
    plt.figure(figsize=(7, 25))

    ax = plt.subplot(7, 1, 1)
    plt.hist(render_fps)
    ax.set_xlabel("Render fps")
    ax.set_title(
        "Scene {} version {}\noptimized {} num_obj {}\n import_robot {}".format(
            scene_name, assets_version, optimized, scene.get_num_objects(), import_robot
        )
    )
    ax = plt.subplot(7, 1, 2)
    plt.hist(physics_fps)
    ax.set_xlabel("Physics fps")
    ax = plt.subplot(7, 1, 3)
    plt.hist(fps)
    ax.set_xlabel("Step fps")
    ax = plt.subplot(7, 1, 4)
    plt.plot(render_fps)
    ax.set_xlabel("Render fps with time")
    ax.set_ylabel("fps")
    ax = plt.subplot(7, 1, 5)
    plt.plot(physics_fps)
    ax.set_xlabel("Physics fps with time, converge to {}".format(np.mean(physics_fps[-100:])))
    ax.set_ylabel("fps")
    ax = plt.subplot(7, 1, 6)
    plt.plot(fps)
    ax.set_xlabel("Step fps with time, converge to {}".format(np.mean(fps[-100:])))
    ax.set_ylabel("fps")
    ax = plt.subplot(7, 1, 7)
    plt.plot(obj_awake)
    ax.set_xlabel("Num object links awake, converge to {}".format(np.mean(obj_awake[-100:])))

    plt.savefig("scene_benchmark_{}_o_{}_r_{}_ycb_{}.pdf".format(scene_name, optimized, import_robot, import_ycb))


def main():
    benchmark_scene("Rs_int", optimized=True, import_robot=True, import_ycb=True)
    benchmark_scene("Rs_int", optimized=True, import_robot=True, import_ycb=False)
    benchmark_scene("Beechwood_0_int", optimized=True, import_robot=True, import_ycb=False)


if __name__ == "__main__":
    main()
