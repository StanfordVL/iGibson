import cv2
import sys
import os
import numpy as np
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_ig_scene_path
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.objects import cube
from igibson.objects.articulated_object import ArticulatedObject
from igibson.utils.assets_utils import get_ig_model_path
import audio
import wave
import pybullet as p
import time
from audio_system import AudioSystem
import pyaudio



def main():
    s = Simulator(mode='iggui', image_width=512, image_height=512, device_idx=0)
    scene = InteractiveIndoorScene('Rs_int', texture_randomization=False, object_randomization=False)
    s.import_ig_scene(scene)

    obj_id = (scene.objects_by_category["loudspeaker"][0]).body_ids[0]
    # Audio System Initialization!
    audioSystem = AudioSystem(s, s.viewer, is_Viewer=True, writeToFile=True, SR = 44100, num_probes=5)
    # Attach wav file to imported cube obj
    audioSystem.registerSource(obj_id, "440Hz_44100Hz.wav", enabled=True)
    # Ensure source continuously repeats
    audioSystem.setSourceRepeat(obj_id)

    s.attachAudioSystem(audioSystem)

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
    audioSystem.disconnect()
    s.disconnect()
    


if __name__ == '__main__':
    main()