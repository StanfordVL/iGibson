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

src_f = "10s_speech_sample.wav"

def plotPairedSpectrograms(nameAndData, title):
    num_plots = len(nameAndData)
    fig, axs = plt.subplots(1, num_plots, figsize=[6.4,2.1])
    if title:
        fig.suptitle(title)
    for i, ax in enumerate(axs):
        ax.specgram(nameAndData[i][1], Fs=44100, scale='dB', NFFT=int(0.025*44100), noverlap=int(0.015*44100))
        ax.title.set_text(nameAndData[i][0])
        ax.set_xlim([0,2])
        ax.set_yscale('log')
        ax.set_ylim([20,1*10**4])
        #ax.set_ylim([0,10000])
    plt.savefig(title)
    
def plotSpectrogram(data, name):
    plt.specgram(data, Fs=44100, scale='dB')
    plt.xlim([0,10])
    plt.ylim([0,20000])
    plt.yscale('log')
    plt.savefig(name)
    plt.close()

def renderAndPlot(scene_id, pos, name, reverb_gain, mat_override=None):
    s = Simulator(mode='gui_interactive', image_width=512, image_height=512, device_idx=0)
    scene = StaticIndoorScene(scene_id)
    #scene = StadiumScene()
    s.import_scene(scene)

    acousticMesh = getMatterportAcousticMesh(s, "/cvgl/group/Gibson/matterport3d-downsized/v2/{}/sem_map.png".format(scene_id))#AcousticMesh()
    if mat_override is not None:
        acousticMesh.materials = np.ones(acousticMesh.materials.shape) * mat_override

    # 10.7, -1.9, 3.5 (small: 6.3, 6.9)
    pos3d = pos + [1.2]
    obj = cube.Cube(pos=pos3d, dim=[0.05, 0.05, 0.05], visual_only=True, mass=0, color=[1,0,0,1])
    s.import_object(obj)
    obj_id = obj.get_body_id()

    pos3d = pos + [1.2]
    s.viewer = Viewer(simulator=s, renderer=s.renderer, initial_pos=pos3d)
    s.viewer.renderer = s.renderer

    # Audio System Initialization, with reverb/reflections off
    audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_Viewer=True, writeToFile=name, SR = 44100, num_probes=5, renderAmbisonics=True, renderReverbReflections=True)
    # Attach wav file to imported cube obj
    
    audioSystem.registerSource(obj_id, src_f, enabled=True, reverb_gain=reverb_gain)
    # Ensure source continuously repeats
    audioSystem.setSourceRepeat(obj_id, False)

    for i in range(int(10.5 / 0.03)):
        s.step()
        audioSystem.step()
    rendered_audio = audioSystem.complete_output[::2]
    plotSpectrogram(rendered_audio, name)
    audioSystem.disconnect()
    s.disconnect()

    return rendered_audio

def mp3d_example():
    src_audio = np.frombuffer(wave.open(src_f, 'rb').readframes(np.inf), dtype=np.int16)
    plotSpectrogram(src_audio, 'src_spectrogram')

    s_g2_metal = renderAndPlot('1pXnuDYAj8r', [ 1.7, -9.5 ], 'small_gain2_metal', 2, mat_override=ResonanceMaterialToId['Metal'])
    l_g2_metal = renderAndPlot('1pXnuDYAj8r', [ 10.7, -1.9], 'large_gain2_metal', 2, mat_override=ResonanceMaterialToId['Metal'])
    plotPairedSpectrograms([("Source", src_audio), ("Small Room", s_g2_metal), ("Large Room", l_g2_metal)] , "Gain 2 (Metal Mesh)")
    s = renderAndPlot('1pXnuDYAj8r', [ 1.7, -9.5 ], 'small_gain1', 1, mat_override=None)
    l = renderAndPlot('1pXnuDYAj8r', [ 10.7, -1.9], 'large_gain1', 1, mat_override=None)
    plotPairedSpectrograms([("Source", src_audio), ("Small Room", s), ("Large Room", l)] , "Gain 1")
    s = renderAndPlot('1pXnuDYAj8r', [ 1.7, -9.5 ], 'small_gain2', 2, mat_override=None)
    l = renderAndPlot('1pXnuDYAj8r', [ 10.7, -1.9], 'large_gain2', 2, mat_override=None)
    plotPairedSpectrograms([("Source", src_audio), ("Small Room", s), ("Large Room", l)] , "Gain 2")
    s = renderAndPlot('1pXnuDYAj8r', [ 1.7, -9.5 ], 'small_gain4', 4, mat_override=None)
    l = renderAndPlot('1pXnuDYAj8r', [ 10.7, -1.9], 'large_gain4', 4, mat_override=None)
    plotPairedSpectrograms([("Source", src_audio), ("Small Room", s), ("Large Room", l)] , "Gain 4")
    s = renderAndPlot('1pXnuDYAj8r', [ 1.7, -9.5 ], 'small_gain6', 6, mat_override=None)
    l = renderAndPlot('1pXnuDYAj8r', [ 10.7, -1.9], 'large_gain6', 6, mat_override=None)
    plotPairedSpectrograms([("Source", src_audio), ("Small Room", s), ("Large Room", l)] , "Gain 6")


def main():
    mp3d_example()
    #ig_example()

if __name__ == '__main__':
    main()