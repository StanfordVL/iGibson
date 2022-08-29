import sys
import os
import numpy as np
import igibson
from igibson.agents.av_nav.utils.environment import AVNavRLEnv
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_ig_scene_path
from igibson.robots import BehaviorRobot, Turtlebot
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene, StaticIndoorScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.objects import cube
from igibson.objects.articulated_object import ArticulatedObject
from igibson.utils.assets_utils import get_ig_model_path
from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.matterport_acoustic_mesh import getMatterportAcousticMesh
from igibson.objects.visual_marker import VisualMarker
import audio
import wave
import pybullet as p
import time
import pyaudio
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config


hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def load_vr_first_example():
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
        "Rs_int", load_object_categories=["walls", "floors", "ceilings"], load_room_types=["kitchen"]
    )
    bvr_config = os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml")
    s.import_scene(scene)

    config = parse_config(bvr_config)
    bvr_robot = BehaviorRobot(**config["robot"])
    s.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0.5, 0, 0.7], [0, 0, 0, 1])
    acousticMesh = getIgAcousticMesh(s)

    turtle_cfg = os.path.join(igibson.configs_path, "turtlebot_interactive_nav.yaml")
    config = parse_config(turtle_cfg)
    listener = Turtlebot(**config["robot"])
    s.import_object(listener)
    listener.set_position_orientation([0.5, 0.2, 0.7], [0, 0, 0, 1])

    # Runs for 30 seconds, then saves output audio to file. 
    for i in range(1000):
        bvr_robot.apply_action(s.gen_vr_robot_action())
        s.step()
    s.disconnect()

def load_igibson_first_example():
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
        msaa=False,
        light_dimming_factor=1.0,
    )

    bvr_config = os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml")
    exp_config = "audiogoal_continuous.yaml"

    # create an iGibsonEnv and load a tutlebot in it
    env = AVNavRLEnv(config_file=exp_config, mode='vr', scene_splits=['Rs_int'], rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))
    
    # load bvr robot in it as the main vr robot
    config = parse_config(bvr_config)
    bvr_robot = BehaviorRobot(**config["robot"])
    env.simulator.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0.5, 0, 0.7], [0, 0, 0, 1])
    env.simulator.switch_main_vr_robot(bvr_robot)

    # Main simulation loop
    obs = env.reset()

    while True:
        #
        env.step(np.array([0., 0.]))
        bvr_robot.apply_action(env.simulator.gen_vr_robot_action())

        if env.simulator.query_vr_event("left_controller", "overlay_toggle"):
            break

    env.simulator.disconnect()

def main():
    load_vr_first_example()

if __name__ == '__main__':
    main()