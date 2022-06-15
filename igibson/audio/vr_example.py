import cv2
import sys
import os
import numpy as np
import igibson
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_ig_scene_path
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene, StaticIndoorScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.objects import cube
from igibson.objects.articulated_object import ArticulatedObject
from igibson.utils.assets_utils import get_ig_model_path
from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.matterport_acoustic_mesh import getMatterportAcousticMesh
from igibson.objects.ycb_object import YCBObject
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
import audio
import wave
import pybullet as p
import time
from audio_system import AudioSystem
import pyaudio

""" VR playground containing various objects. This playground operates in the
Rs_int PBR scene.

Important - VR functionality and where to find it:

1) Most VR functions can be found in the igibson/simulator.py
2) The BehaviorRobot and its associated parts can be found in igibson/robots/behavior_robot.py
3) VR utility functions are found in igibson/utils/vr_utils.py
4) The VR renderer can be found in igibson/render/mesh_renderer.py
5) The underlying VR C++ code can be found in vr_mesh_render.h and .cpp in igibson/render/cpp
"""


# Whether to use VR mode or not
VR_MODE = True
# Set to false to load entire Rs_int scene
LOAD_PARTIAL = False
# Set to true to print out render, physics and overall frame FPS
PRINT_STATS = True
# Set to true to use VR hand instead of gripper
USE_GRIPPER = False

# HDR files for PBR rendering
hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main():
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
    # VR system settings
    # Change use_vr to toggle VR mode on/off
    vr_settings = VrSettings()
    if not VR_MODE:
        vr_settings.turn_off_vr_mode()
    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=vr_settings)
    scene = InteractiveIndoorScene("Rs_int")
    # Turn this on when debugging to speed up loading
    if LOAD_PARTIAL:
        scene._set_first_n_objects(10)
    s.import_ig_scene(scene)
    obj_id = (scene.objects_by_category["loudspeaker"][0]).body_ids[0]

    acousticMesh = getIgAcousticMesh(s)
    # Audio System Initialization!
    audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_Viewer=True, writeToFile=False, SR = 44100, num_probes=5)
    # Attach wav file to imported cube obj
    audioSystem.registerSource(obj_id, "10s_speech_sample", enabled=True)
    # Ensure source continuously repeats
    audioSystem.setSourceRepeat(obj_id)

    if not VR_MODE:
        camera_pose = np.array([0, -3, 1.2])
        view_direction = np.array([0, 1, 0])
        s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        s.renderer.set_fov(90)

    if VR_MODE:
        # Create a BehaviorRobot and it will handle all initialization and importing under-the-hood
        # Change USE_GRIPPER to switch between the BRHand and the BRGripper (see robots/behavior_robot.py for more details)
        vr_agent = BehaviorRobot(s, use_gripper=USE_GRIPPER)
        # Since vr_height_offset is set, we will use the VR HMD true height plus this offset instead of the z coordinate of start_pos
        s.set_vr_start_pos([0, 0, 0], vr_height_offset=-0.1)
    source_id = vr_agent.parts['eye'].get_body_id()
    audioSystem.registerSource(source_id, None, enabled=True)
    # Main simulation loop
    while True:
        s.step(print_stats=PRINT_STATS)
        audioSystem.step()

        if VR_MODE:
            # Update VR objects
            vr_agent.apply_action()

    s.disconnect()


if __name__ == "__main__":
    main()

