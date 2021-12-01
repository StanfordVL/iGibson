#!/usr/bin/env python

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots.turtlebot_robot import Turtlebot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_assets_version
from igibson.utils.utils import parse_config


def benchmark_scene(scene_name, num_processes):
    def make_env(rank, seed=0, config=None):
        def _init() -> iGibsonEnv:
            env = iGibsonEnv(
                config_file=config,
                mode="headless",
                action_timestep=1 / 30.0,
                physics_timestep=1 / 120.0,
                device_idx=rank // 8 if num_processes == 64 else 0,
                rendering_settings=MeshRendererSettings(msaa=False, enable_shadow=False, optimized=True),
            )
            env.seed(seed + rank)
            return env

        return _init

    config = parse_config(os.path.join(os.path.dirname(__file__), "benchmark.yaml"))
    config["scene_id"] = scene_name
    if num_processes == 1:
        env = iGibsonEnv(
            config_file=config,
            mode="headless",
            action_timestep=1 / 30.0,
            physics_timestep=1 / 120.0,
            rendering_settings=MeshRendererSettings(msaa=False, enable_shadow=False, optimized=True),
        )
    else:
        env = SubprocVecEnv([make_env(rank=i, seed=0, config=config) for i in range(num_processes)])

    env.reset()
    fps = []
    for i in range(2000):
        start = time.time()
        env.step([None] * num_processes)
        end = time.time()
        # print("Elapsed time: ", end - start)
        print("Step Frequency: ", 1 / (end - start))
        # print(env.robots[0].get_position())
        fps.append(1 / (end - start))
    env.close()
    print("{}, {}: Best SPS:".format(scene_name, num_processes), np.max(fps) * num_processes)

    plt.figure(figsize=(2, 25))
    ax = plt.subplot(2, 1, 1)
    plt.hist(fps)
    ax.set_xlabel("Step fps")
    ax.set_title("Scene {} num_proc_{}".format(scene_name, num_processes))
    ax = plt.subplot(2, 1, 2)
    plt.plot(fps)
    ax.set_xlabel("Step fps with time, converge to {}".format(np.mean(fps[-100:])))
    ax.set_ylabel("fps")

    plt.savefig("scene_benchmark_{}_num_proc_{}.pdf".format(scene_name, num_processes))


def main():
    for n_proc in [1, 16, 64]:
        for scene in ["Benevolence_0_int", "Rs_int", "Beechwood_0_int"]:
            benchmark_scene(scene, num_processes=n_proc)

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
