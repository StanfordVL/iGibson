import sys
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

N_FFT = 512
librosa_FS = 22050
freq_bins = np.arange(0, 1 + N_FFT / 2) * librosa_FS / N_FFT

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

def compute_stft(signal, signal_sr):
    hop_length = 160
    win_length = 400
    signal_resampled = librosa.resample(signal, signal_sr, librosa_FS)
    stft = np.abs(librosa.stft(signal_resampled, n_fft=N_FFT, hop_length=hop_length, win_length=win_length))
    return stft
    
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

def img_mtx_to_overlay(overlay):
    overlay_scaled = overlay.copy()
    max_val = np.max(overlay[np.nonzero(overlay)])
    min_val = np.min(overlay[np.nonzero(overlay)])
    for i in range(overlay_scaled.shape[0]):
        for j in range(overlay_scaled.shape[1]):
            if overlay_scaled[i, j] != 0:
                scaled_val = 255 * (overlay_scaled[i, j] - min_val) / (max_val - min_val)
                overlay_scaled[i, j] = scaled_val

    overlay_scaled = overlay_scaled.astype(np.uint8)
    overlay_cmap = cv2.applyColorMap(overlay_scaled, cv2.COLORMAP_JET)
    overlay_cmap = cv2.cvtColor(overlay_cmap, cv2.COLOR_RGB2RGBA)
    for i in range(overlay.shape[0]):
        for j in range(overlay.shape[1]):
            if overlay[i, j] == 0:
                overlay_cmap[i, j] = [0, 0, 0, 0]

    return overlay_cmap

def main():
    scene_choices = {
        #"Rs_int": "ig",
        #"1LXtFkjw3qL: "mp3d"",
        "1pXnuDYAj8r": "mp3d",
        #"Ihlen_0_int": "mp3d",
    }
    scene_trav_map_size = {
        "1pXnuDYAj8r": 4000,
        #"1LXtFkjw3qL": 3600,
        #"Rs_int": 1000,
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
            camera_pose = np.array([0, 0, -0.7])
            view_direction = np.array([0, 0, 0.7])
        else:
            scene = StaticIndoorScene(scene_id,  trav_map_resolution = 0.1)
            s.import_scene(scene)
            acousticMesh = getMatterportAcousticMesh(s, "/cvgl/group/Gibson/matterport3d-downsized/v2/{}/sem_map.png".format(scene_id))
            camera_pose = np.array([0, 0, -1.0])
            view_direction = np.array([0, 0, 1.0])
        
        renderer = s.renderer
        trav_res = scene.trav_map_default_resolution
        points = scene.get_points_grid(100)[0]
        
        
        fake_viewer = FakeViewer()
        audioSystem = AudioSystem(s, fake_viewer, acousticMesh, is_Viewer=True, renderReverbReflections=True, renderAmbisonics=False, spectrogram_window_len=s.render_timestep)

        #s.step()
        
        obj_pos = [7, -4.5, -0.05869176]
        obj1 = cube.Cube(pos=obj_pos, dim=[0.1,0.1,0.01], visual_only=True, mass=0, color=[255,255,255,1])
        s.import_object(obj1)
        obj_id = obj1.get_body_ids()[0]
        
        # Attach wav file to imported cube obj
        audioSystem.registerSource(obj_id, "440Hz_44100Hz.wav", enabled=True)
        # Ensure source continuously repeats
        audioSystem.setSourceRepeat(obj_id)
        s.step()
        audioSystem.step()

        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
        # cache original P and recover for robot cameras
        p_range = map_size / 200.0
        renderer.P = ortho(-p_range, p_range, -p_range, p_range, -10, 20.0)
        frame, three_d = renderer.render(modes=("rgb", "3d"))
        depth = -three_d[:, :, 2]
        # white bg
        frame[depth == 0] = 1.0
        frame = cv2.flip(frame, 0)
        bg = (frame[:, :, 0:3][:, :, ::-1] * 255).astype(np.uint8)
        overlay = np.zeros((bg.shape[0], bg.shape[1]))
        occl_overlay = np.zeros((bg.shape[0], bg.shape[1]))
        cv2.imwrite("floorplan/{}.png".format(scene_id), bg)
        
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2RGBA)
        max_intensity = -1 * np.inf
        min_intensity = np.inf
        max_pt = None
        for idx, world_point in enumerate(points):
            fake_viewer.px = world_point[0]
            fake_viewer.py = world_point[1]
            fake_viewer.pz = world_point[2] + 0.5
            intensity = 0
            for sample in range(10):
                s.step()
                audioSystem.step()
                spectrograms = audioSystem.get_spectrogram()
                intensity += spec_power(spectrograms[:,:,0])
                intensity += spec_power(spectrograms[:,:,1])
            if intensity == 0.0:
                continue

            if intensity > max_intensity:
                max_intensity = intensity
                max_pt = [fake_viewer.px, fake_viewer.py, fake_viewer.pz]
            min_intensity = min(min_intensity, intensity)
            px, py = worldToPixel(world_point, map_size, trav_res)
            square_radius = 30
            for i in range(-square_radius,square_radius+1):
                for j in range(-square_radius,square_radius+1):
                    if overlay[px + i, py + j]:
                        overlay[px + i, py + j] = (intensity + overlay[px + i, py + j]) / 2
                        occl_overlay[px + i, py + j] = (occl_overlay[px + i, py + j]+audioSystem.occl_intensity) / 2
                    else:
                        overlay[px + i, py + j] = intensity
                        occl_overlay[px + i, py + j] = audioSystem.occl_intensity

        print("Max RMS = " + str(max_intensity) + " Min RMS = " + str(min_intensity))
        print("Obj 0 at" + str(obj_pos))
        print("Max RMS at " + str(max_pt))
        intensity_overlay = img_mtx_to_overlay(overlay)
        cv2.imwrite("floorplan/{}_intensity_bare.png".format(scene_id), intensity_overlay)

        occl_overlay = img_mtx_to_overlay(occl_overlay)
        cv2.imwrite("floorplan/{}_occl_bare.png".format(scene_id), occl_overlay)
  
        img = cv2.addWeighted(bg, 1, intensity_overlay, 0.4, 0)
        cv2.imwrite("floorplan/{}_intensity.png".format(scene_id), img)

        img = cv2.addWeighted(bg, 1, occl_overlay, 0.4, 0)
        cv2.imwrite("floorplan/{}_occl.png".format(scene_id), img)

        s.disconnect()

if __name__ == "__main__":
    main()
