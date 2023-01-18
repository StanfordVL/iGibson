"""
Demo for testing VR body based on torso tracker
"""
import logging

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator


def main():
    s = Simulator(mode="vr", rendering_settings=MeshRendererSettings(enable_shadow=True, optimized=True))
    scene = EmptyScene()
    s.import_scene(scene)
    vr_agent = BehaviorRobot()

    # Main simulation loop
    while True:
        s.step()
        vr_agent.apply_action()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
