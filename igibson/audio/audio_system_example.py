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
import pyaudio
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
    #_,source_location = scene.get_random_point_by_room_type("living_room")
    #source_location[2] = 1.7
    #model = "fe613d2a63582e126e5a8ef1ff6470a3"
    #model_path = get_ig_model_path("loudspeaker","fe613d2a63582e126e5a8ef1ff6470a3")
    #filename = os.path.join(model_path, model + ".urdf")
    #obj = ArticulatedObject(filename)
    #obj = cube.Cube(pos=source_location, dim=[0.2, 0.2, 0.2], visual_only=False, mass=0.5, color=[255, 0, 0, 1])
    #obj_id = s.import_object(obj)

    # Audio System Initialization!
    audioSystem = AudioSystem(s, s.viewer, is_Viewer=True, writeToFile=True, SR = 44100)
    # Attach wav file to imported cube obj
    audioSystem.registerSource(obj_id, "440Hz_44100Hz.wav", enabled=True)
    # Ensure source continuously repeats
    audioSystem.setSourceRepeat(obj_id)

    s.attachAudioSystem(audioSystem)

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