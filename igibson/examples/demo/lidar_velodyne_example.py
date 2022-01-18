import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import igibson
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.robots.turtlebot_robot import Turtlebot
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config


def main():
    config = parse_config(os.path.join(igibson.example_config_path, "turtlebot_demo.yaml"))
    settings = MeshRendererSettings()
    s = Simulator(mode="headless", image_width=256, image_height=256, rendering_settings=settings)

    scene = StaticIndoorScene("Rs", build_graph=True, pybullet_load_texture=True)
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    turtlebot.apply_action([0.1, -0.1])
    s.step()
    lidar = s.renderer.get_lidar_all()
    print(lidar.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(lidar[:, 0], lidar[:, 1], lidar[:, 2], s=3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    s.disconnect()


if __name__ == "__main__":
    main()
