from igibson.robots.fetch import Fetch
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene, StaticIndoorScene
from igibson.objects.ycb_object import YCBObject
from igibson.objects import cube
from igibson.utils.utils import parse_config
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from igibson.utils.mesh_util import ortho
import cv2
from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.matterport_acoustic_mesh import getMatterportAcousticMesh
from igibson.utils.mesh_util import lookat, mat2xyz, ortho, perspective, quat2rotmat, safemat2quat, xyz2mat, xyzw2wxyz
from audio_system import AudioSystem
import librosa
import librosa.display



SPECTROGRAM_DURATION_S=1


class FakeViewer:
    def __init__(self):
        self.px, self.py, self.pz = 0, 1, 0
        self.phi, self.theta = 0, 0


def worldToPixel(xy, map_size, res=0.01):
    px = int(xy[1] / res) + map_size // 2
    py = -1 * int(xy[0] / res) + map_size // 2
    return [px, py]

def pixelToWorld(pxy, map_size, res=0.01):
    y = (pxy[0] - map_size // 2) * res
    x = -1 * (pxy[1] - map_size // 2) * res
    return [x, y]
    
def plot_spectrogram(spec, fname="spectrogram.png"):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, y_axis='log', x_axis='time', ax=ax, sr=44100, hop_length=160)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.savefig(fname)

def spec_power(spec):
    rms = librosa.feature.rms(S=spec, frame_length=512, hop_length=160)
    avg_rms = np.sum(rms) / np.size(rms)
    return avg_rms



def main():
    scene_choices = {
        "Rs_int": "ig",
        #"1LXtFkjw3qL: "mp3d"",
        #"1pXnuDYAj8r: "mp3d"",
        #"Ihlen_0_int": "mp3d",
    }
    scene_trav_map_size = {
        "1pXnuDYAj8r": 4000,
        "1LXtFkjw3qL": 3600,
        "Rs_int": 1000,
        #"Ihlen_0_int": 2000
    }
    for scene_id, scene_type in scene_choices.items():
        map_size = scene_trav_map_size[scene_id]
        settings = MeshRendererSettings(enable_shadow=False, msaa=False)
        s = Simulator(mode="headless", image_width=map_size, image_height=map_size, rendering_settings=settings)

        if scene_type == "ig":
            scene = InteractiveIndoorScene(scene_id, texture_randomization=False, object_randomization=False, trav_map_resolution = 0.1,)
            s.import_scene(scene)
            acousticMesh = getIgAcousticMesh(s)
            camera_pose = np.array([0, 0, -10])
            view_direction = np.array([0, 0, 1])
        else:
            scene = StaticIndoorScene(scene_id,  trav_map_resolution = 0.1)
            s.import_scene(scene)
            acousticMesh = getMatterportAcousticMesh(s, "/cvgl/group/Gibson/matterport3d-downsized/v2/{}/sem_map.png".format(scene_id))
            camera_pose = np.array([0, 0, -1.0])
            view_direction = np.array([0, 0, 1.0])
        
        points = scene.get_points_grid(10)[0]
        
        
        fake_viewer = FakeViewer()
        fake_viewer.px = points[0][0] + 0.1
        fake_viewer.py = points[0][1] + 0.1
        fake_viewer.pz = points[0][2]
        audioSystem = AudioSystem(s, fake_viewer, acousticMesh, is_Viewer=True, renderReverbReflections=True, renderAmbisonics=True, spectrogram_window_len=SPECTROGRAM_DURATION_S)
        
        obj1 = cube.Cube(pos=points[0], dim=[0.1,0.1,0.01], visual_only=True, mass=0, color=[1,0,0,1])
        s.import_object(obj1)
        obj_id = obj1.get_body_ids()[0]
        
        # Attach wav file to imported cube obj
        audioSystem.registerSource(obj_id, "440Hz_44100Hz.wav", enabled=True)
        # Ensure source continuously repeats
        audioSystem.setSourceRepeat(obj_id)
        s.step()
        audioSystem.step()
        silence = False
        for i in range(int(SPECTROGRAM_DURATION_S / s.render_timestep)):
            fake_viewer.px += 0.1
            fake_viewer.py += 0.1
            s.step()
            audioSystem.step()

        spectrograms = audioSystem.get_spectrogram()
        for channel in range(spectrograms.shape[2]):
            plot_spectrogram(spectrograms[:,:,channel], fname="spec_ch_"+str(channel))


        s.disconnect()

if __name__ == "__main__":
    main()
