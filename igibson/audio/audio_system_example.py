import sys
import os
import numpy as np
import igibson
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
from audio_system import AudioSystem
import pyaudio
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config


hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def vr_input_example():
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

    

    # Audio System Initialization!
    audioSystem = AudioSystem(s, listener, acousticMesh, is_Viewer=False, writeToFile="example", stream_input=True)
    audioSystem.registerSource(
                bvr_robot._parts['eye'].body_id,
                "",
                enabled=True,
                repeat=True
            )
    # Runs for 30 seconds, then saves output audio to file. 
    for i in range(1000):
        bvr_robot.apply_action(s.gen_vr_robot_action())
        s.step()
        audioSystem.step()
    audioSystem.disconnect()
    s.disconnect()

def ig_example():
    s = Simulator(mode='gui_interactive', image_width=512, image_height=512, device_idx=0)
    scene = InteractiveIndoorScene('Rs_int', texture_randomization=False, object_randomization=False)
    s.import_scene(scene)

    obj_id = (scene.objects_by_category["loudspeaker"][0]).get_body_ids()[0]

    acousticMesh = getIgAcousticMesh(s)

    # Audio System Initialization!
    audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_Viewer=True, writeToFile="example")
    # Attach wav file to imported cube obj
    audioSystem.registerSource(obj_id, "440Hz_44100Hz.wav", enabled=True)

    # Visualize reverb probes!
    for i in range(len(scene.floor_heights)):
        for probe_pos in audioSystem.probe_key_to_pos_by_floor[i].values():
            z = scene.floor_heights[i] + 1.7
            pos = [probe_pos[0], probe_pos[1], z]
            obj = cube.Cube(pos=pos, dim=[0.1, 0.1, 0.1], visual_only=True, mass=0, color=[255, 0, 0, 1])
            s.import_object(obj)

    # This section is entirely optional - it simply tries to stream audio live
    def pyaudCallback(in_data, frame_count, time_info, status):
        return (bytes(audioSystem.current_output), pyaudio.paContinue)
    pyaud = pyaudio.PyAudio()
    stream = pyaud.open(rate=audioSystem.SR, frames_per_buffer=audioSystem.framesPerBuf, format=pyaudio.paInt16, channels=2, output=True, stream_callback=pyaudCallback)

    # Runs for 30 seconds, then saves output audio to file. 
    for i in range(1000):
        s.step()
        audioSystem.step()
    audioSystem.disconnect()
    s.disconnect()
    
def mp3d_example():
    s = Simulator(mode='gui_interactive', image_width=512, image_height=512, device_idx=0)
    scene = StaticIndoorScene('17DRP5sb8fy')
    s.import_scene(scene)

    acousticMesh = getMatterportAcousticMesh(s, "/cvgl/group/Gibson/matterport3d-downsized/v2/17DRP5sb8fy/sem_map.png")

    obj = cube.Cube(pos=[0, 0, 2], dim=[0.3, 0.3, 0.3], visual_only=True, mass=0, color=[1,1,0,1])
    s.import_object(obj)
    obj_id = obj.get_body_id()
    # Audio System Initialization!
    audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_Viewer=True, writeToFile="example")
    # Attach wav file to imported cube obj
    audioSystem.registerSource(obj_id, "440Hz_44100Hz.wav", enabled=True)
    # Ensure source continuously repeats
    audioSystem.setSourceRepeat(obj_id)

    # Visualize reverb probes!
    for i in range(len(scene.floor_heights)):
        for probe_pos in audioSystem.probe_key_to_pos_by_floor[i].values():
            z = scene.floor_heights[i] + 1.7
            pos = [probe_pos[0], probe_pos[1], z]
            obj = cube.Cube(pos=pos, dim=[0.1, 0.1, 0.1], visual_only=True, mass=0, color=[255, 0, 0, 1])
            s.import_object(obj)

    # This section is entirely optional - it simply tries to stream audio live
    def pyaudCallback(in_data, frame_count, time_info, status):
        return (bytes(audioSystem.current_output), pyaudio.paContinue)
    pyaud = pyaudio.PyAudio()
    stream = pyaud.open(rate=audioSystem.SR, frames_per_buffer=audioSystem.framesPerBuf, format=pyaudio.paInt16, channels=2, output=True, stream_callback=pyaudCallback)

    # Runs for 30 seconds, then saves output audio to file. 
    for i in range(4000):
        s.step()
        audioSystem.step()

    audioSystem.disconnect()
    s.disconnect()
    
def main():
    #mp3d_example()
    #ig_example()
    vr_input_example()

if __name__ == '__main__':
    main()