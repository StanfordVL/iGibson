import cv2
import sys
import os
import numpy as np
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
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
from igibson.audio.acoustic_material_mapping import ResonanceMaterialToId
from igibson.render.viewer import Viewer
import audio
import wave
import pybullet as p
import matplotlib.pyplot as plt
import time
from audio_system import AudioSystem, AcousticMesh
from math import cos, sin, atan2, sqrt, pi, factorial

def mp3d_example():
    s = Simulator(mode='iggui', image_width=512, image_height=512, device_idx=0)
    scene = StaticIndoorScene('1pXnuDYAj8r')
    #scene = StadiumScene()
    s.import_scene(scene)



    acousticMesh = getMatterportAcousticMesh(s, "/cvgl/group/Gibson/matterport3d-downsized/v2/1pXnuDYAj8r/sem_map.png")#AcousticMesh()
    #acousticMesh = AcousticMesh()
    #transparent_id =ResonanceMaterialToId["Metal"]
    #Make mesh transparent so we only render direct sound
    #acousticMesh.materials = np.ones(acousticMesh.materials.shape) * transparent_id

    # 10.7, -1.9, 3.5 (small: 6.3, 6.9)
    obj = cube.Cube(pos=[10.7, -1.9, 2], dim=[0.05, 0.05, 0.05], visual_only=True, mass=0, color=[1,0,0,1])
    obj_id = s.import_object(obj)[0]

    s.viewer = Viewer(simulator=s, renderer=s.renderer, initial_pos=[10.7, -1.9, 1.2])
    s.viewer.renderer = s.renderer

    # Audio System Initialization, with reverb/reflections off
    audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_Viewer=True, writeToFile=True, SR = 44100, num_probes=5, renderAmbisonics=True, renderReverbReflections=True)
    # Attach wav file to imported cube obj
    audioSystem.registerSource(obj_id, "gettysburg.wav", enabled=True)
    # Ensure source continuously repeats
    audioSystem.setSourceRepeat(obj_id)

    s.attachAudioSystem(audioSystem)

    state = True
    for i in range(70):
        if i % 68 == 0:
            audioSystem.setSourceEnabled(obj_id, state)
            state = not state
        s.step()
    rendered_audio = audioSystem.complete_output[::2]
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(rendered_audio, Fs=44100)
    plt.savefig('audio_spectrogram')
    audioSystem.disconnect()
    s.disconnect()
    
def main():
    mp3d_example()
    #ig_example()

if __name__ == '__main__':
    main()