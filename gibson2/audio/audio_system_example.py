import cv2
import sys
import os
import numpy as np
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.profiler import Profiler
from gibson2.utils.assets_utils import get_ig_scene_path
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.stadium_scene import StadiumScene
from gibson2.objects import cube
import pyaudio
import audio
import wave
import pybullet as p
import time
from audio_system import AudioSystem
from scipy.io.wavfile import read, write


def main():
    s = Simulator(mode='gui', image_width=512, image_height=512, device_idx=0)
    scene = InteractiveIndoorScene('Rs_int', texture_randomization=False, object_randomization=False)
    s.import_ig_scene(scene)

    _,source_location = scene.get_random_point_by_room_type("living_room")
    source_location[2] = 1.7
    obj = cube.Cube(pos=source_location, dim=[0.2, 0.2, 0.2], visual_only=False, mass=0.5, color=[255, 0, 0, 1])
    obj_id = s.import_object(obj)

    audioSystem = AudioSystem(s, s.viewer, is_Viewer=True, writeToFile=True, SR = 44100)
    audioSystem.registerSource(obj_id, "440Hz_44100Hz.wav", enabled=True)
    audioSystem.setSourceRepeat(obj_id)

    s.attachAudioSystem(audioSystem)


    for i in range(5000):
        s.step()
        audioSystem.step()
    audioSystem.disconnect()
    s.disconnect()
    


if __name__ == '__main__':
    main()