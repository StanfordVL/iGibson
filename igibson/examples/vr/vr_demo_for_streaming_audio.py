""" This is a VR demo in a simple scene consisting of some objects to interact with, and space to move around.
Can be used to verify everything is working in VR, and iterate on current VR designs.
"""
import logging
import os

import pybullet as p
import pybullet_data
import numpy as np

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot, Turtlebot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.audio_system import AudioSystem
from igibson.agents.av_nav.ppo.ppo_trainer import PPOTrainer
from igibson.agents.av_nav.utils.environment import AVNavRLEnv
from igibson.objects.visual_marker import VisualMarker
import torch

# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config


hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main(selection="user", headless=False, short_exec=False):
    

    # VR rendering settings
    bvr_config = os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml")
    exp_config = "C:/Users/capri28/Documents/iGibson-dev/igibson/examples/vr/audiogoal_continuous.yaml"

    env = AVNavRLEnv(config_file=exp_config, mode='headless', scene_splits=['Rs_int'])
    config = parse_config(bvr_config)
    bvr_robot = BehaviorRobot(**config["robot"])
    env.simulator.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0.5, 0, 0.7], [0, 0, 0, 1])

    # bvr_robot.set_position_orientation([0.5, 0, 0.7], [0, 0, 0, 1])

    env.audio_system.registerSource(
                bvr_robot._parts['eye'].body_id,
                "",
                enabled=True,
                repeat=True
            )

    # Main simulation loop
    env.reset()

    initial_pos = env.robots[0].get_position()
    initial_orn = env.robots[0].get_orientation()
    

     # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        enable_pbr=True,
        msaa=True,
        light_dimming_factor=1.0,
    )

    s = SimulatorVR(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))
    scene = InteractiveIndoorScene(
        "Rs_int", load_object_categories=["walls", "floors"], load_room_types=["kitchen"]
    )
    s.import_scene(scene)

    config = parse_config(bvr_config)
    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0.5, 0, 0.7], [0, 0, 0, 1])

    turtle_cfg = os.path.join(igibson.configs_path, "turtlebot_interactive_nav.yaml")
    config = parse_config(turtle_cfg)
    listener = Turtlebot(**config["robot"])
    s.import_object(listener)
    listener.set_position_orientation(initial_pos, initial_orn)

    

    # Audio System Initialization!
    # audioSystem = AudioSystem(s, listener, acousticMesh, is_Viewer=False, writeToFile="mic_out", stream_input=True)
    # audioSystem.registerSource(
    #             bvr_robot._parts['eye'].body_id,
    #             "",
    #             enabled=True,
    #             repeat=True
    #         )
    # # Runs for 30 seconds, then saves output audio to file. 
    # for i in range(1000):
    #     bvr_robot.apply_action(s.gen_vr_robot_action())
    #     s.step()
    #     audioSystem.step()
    # audioSystem.disconnect()
    # s.disconnect()
    while True:
        action = env.action_space.sample()
        env.step(action)
        # End demo by pressing overlay toggle
        bvr_robot.apply_action(s.gen_vr_robot_action())
        s.step()
        if env.simulator.query_vr_event("left_controller", "overlay_toggle"):
            break
    env.audio_system.disconnect()
    env.simulator.disconnect()
    s.disconnect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
