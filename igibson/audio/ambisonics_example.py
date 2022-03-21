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
import audio
import wave
import pybullet as p
import time
from audio_system import AudioSystem, AcousticMesh
from math import cos, sin, atan2, sqrt, pi, factorial

class Position(object):
    def __init__(self, x1, x2, x3, c_type):
        assert c_type.lower() in ['cartesian', 'polar']

        self.x, self.y, self.z = 0., 0., 0.
        self.phi, self.nu, self.r = 0., 0., 0.
        if c_type == 'cartesian':
            self.set_cartesian(x1, x2, x3)
        else:
            self.set_polar(x1, x2, x3)

    def clone(self):
        return Position(self.x, self.y, self.z, 'cartesian')

    def set_cartesian(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.calc_polar()
        self.calc_cartesian()

    def set_polar(self, phi, nu, r):
        self.phi, self.nu, self.r = float(phi), float(nu), float(r)
        self.calc_cartesian()
        self.calc_polar()

    def calc_cartesian(self):
        self.x = self.r * cos(self.phi) * cos(self.nu)
        self.y = self.r * sin(self.phi) * cos(self.nu)
        self.z = self.r * sin(self.nu)

    def calc_polar(self):
        self.phi = atan2(self.y, self.x)
        self.nu = atan2(self.z, sqrt(self.x**2+self.y**2))
        self.r = sqrt(self.x**2+self.y**2+self.z**2)

    def rotate(self, rot_matrix):
        pos = np.dot(rot_matrix, np.array([self.x, self.y, self.z]).reshape(3, 1))
        self.x, self.y, self.z = pos[0], pos[1], pos[2]
        self.calc_polar()
        self.calc_cartesian()

    def set_radius(self, radius):
        self.r = radius
        self.calc_cartesian()

    def coords(self, c_type):
        if c_type == 'cartesian':
            return np.array([self.x, self.y, self.z])
        elif c_type == 'polar':
            return np.array([self.phi, self.nu, self.r])
        else:
            raise ValueError('Unknown coordinate type. Use cartesian or polar.')

    def print_position(self, c_type=None):
        if c_type is None or c_type == 'cartesian':
            print('Cartesian (x,y,z): (%.2f, %.2f, %.2f)' % (self.x, self.y, self.z))
        if c_type is None or c_type == 'polar':
            print('Polar (phi,nu,r):  (%.2f, %.2f, %.2f)' % (self.phi, self.nu, self.r))

CHANNEL_ORDERING = ['FURSE_MALHAM', 'SID', 'ACN']
NORMALIZATION = ['MAX_N', 'SN3D', 'N3D']
DEFAULT_ORDERING = 'ACN'
DEFAULT_NORMALIZATION = 'SN3D'
DEFAULT_RATE = 44100
DEFAULT_RADIUS = 3.
DEFAULT_ORDER = 1

class AmbiFormat(object):
    def __init__(self,
                 ambi_order=DEFAULT_ORDER,
                 sample_rate=DEFAULT_RATE,
                 radius=DEFAULT_RADIUS,
                 ordering=DEFAULT_ORDERING,
                 normalization=DEFAULT_NORMALIZATION):
        self.order = ambi_order
        self.num_channels = int((ambi_order+1)**2)
        self.radius = radius
        self.sample_rate = sample_rate
        self.ordering = ordering
        self.normalization = normalization


def degree_order_to_index(order, degree, ordering=DEFAULT_ORDERING):
    assert -order <= degree <= order
    assert ordering in CHANNEL_ORDERING

    def acn_idx(n, m):
        return n*(n+1)+m

    def sid_idx(n, m):
        idx_order = [1+i*2 for i in range(n)] + [n*2] + list(reversed([i*2 for i in range(n)]))
        return idx_order[m+n] + n**2

    def fm_idx(n, m):
        if n == 1:
            idx_order = [1, 2, 0]
        else:
            idx_order = list(reversed([2*(i+1) for i in range(n)])) + [0] + [1+i*2 for i in range(n)]
        return idx_order[m+n] + n**2

    if ordering == 'ACN':
        return acn_idx(order, degree)
    elif ordering == 'FURSE_MALHAM':
        return fm_idx(order, degree)
    else:
        return sid_idx(order, degree)


def index_to_degree_order(index, ordering=DEFAULT_ORDERING):
    assert ordering in CHANNEL_ORDERING
    order = int(sqrt(index))
    index -= order**2
    if ordering == 'ACN':
        degree = index - order
        return order, degree
    elif ordering == 'FURSE_MALHAM':
        if order == 1:
            mapping = [1, -1, 0]
            degree = mapping[index]
        else:
            degree = (int(index)+1)/2
            if index % 2 == 0:
                degree = -degree
        return order, degree
    else:
        degree = (int(index)+1)/2
        if index % 2 == 0:
            degree = -degree
        return order, degree
    
def normalization_factor(index, ordering=DEFAULT_ORDERING, normalization=DEFAULT_NORMALIZATION):
    assert ordering in CHANNEL_ORDERING
    assert normalization in NORMALIZATION

    def max_norm(n, m):
        assert n <= 3
        if n == 0:
            return 1/sqrt(2.)
        elif n == 1:
            return 1.
        elif n == 2:
            return 1. if m == 0 else 2. / sqrt(3.)
        else:
            return 1. if m == 0 else (sqrt(45. / 32) if m in [1, -1] else 3. / sqrt(5.))

    def sn3d_norm(n, m):
        return sqrt((2. - float(m == 0)) * float(factorial(n-abs(m))) / float(factorial(n+abs(m))))

    def n3d_norm(n, m):
        return sn3d_norm(n, m) * sqrt((2*n+1) / (4.*pi))

    order, degree = index_to_degree_order(index, ordering)
    if normalization == 'MAX_N':
        return max_norm(order, degree)
    elif normalization == 'N3D':
        return n3d_norm(order, degree)
    elif normalization == 'SN3D':
        return sn3d_norm(order, degree)


def spherical_harmonic_mn(order, degree, phi, nu, normalization=DEFAULT_NORMALIZATION):
    from scipy.special import lpmv
    norm = normalization_factor(degree_order_to_index(order, degree), normalization=normalization)
    sph = (-1)**degree * norm * \
        lpmv(abs(degree), order, np.sin(nu)) * \
        (np.cos(abs(degree) * phi) if degree >= 0 else np.sin(abs(degree) * phi))
    return sph


def spherical_harmonics(position, max_order, ordering=DEFAULT_ORDERING, normalization=DEFAULT_NORMALIZATION):
    assert isinstance(position, Position)

    num_channels = int((max_order+1)**2)
    output = np.zeros((num_channels,))
    for i in range(num_channels):
        order, degree = index_to_degree_order(i, ordering)
        output[i] = spherical_harmonic_mn(order, degree, position.phi, position.nu, normalization)
    return output


def spherical_harmonics_matrix(positions, max_order, ordering=DEFAULT_ORDERING, normalization=DEFAULT_NORMALIZATION):
    assert isinstance(positions, list) and all([isinstance(p, Position) for p in positions])

    num_channels = int((max_order + 1) ** 2)
    Y = np.zeros((len(positions), num_channels))
    for i, p in enumerate(positions):
        Y[i] = spherical_harmonics(p, max_order, ordering, normalization)
    return Y

DECODING_METHODS = ['projection', 'pseudoinv']
DEFAULT_DECODING = 'projection'


class AmbiDecoder(object):
    def __init__(self, speakers_pos, ambi_format, method=DEFAULT_DECODING):
        assert method in DECODING_METHODS
        if isinstance(speakers_pos, Position):
            speakers_pos = [speakers_pos]
        assert isinstance(speakers_pos, list) and all([isinstance(p, Position) for p in speakers_pos])
        self.speakers_pos = speakers_pos
        self.sph_mat = spherical_harmonics_matrix(speakers_pos,
                                                  ambi_format.order,
                                                  ambi_format.ordering,
                                                  ambi_format.normalization)
        self.method = method
        if self.method == 'pseudoinv':
            self.pinv = np.linalg.pinv(self.sph_mat)

    def decode(self, ambi):
        if self.method == 'projection':
            return np.dot(ambi, self.sph_mat.T)
        if self.method == 'pseudoinv':
            return np.dot(ambi, self.pinv)

def spherical_mesh(angular_res):
    # X -> front, Y -> left, Z -> top
    # phi -> horizontal angle from X counterclockwise
    phi_rg = np.flip(np.arange(-180., 180, angular_res) / 180. * np.pi, 0)
    # nu -> elevation angle from from X towards top
    nu_rg = np.arange(0., 0.1, angular_res) / 180. * np.pi

    phi_mesh, nu_mesh = np.meshgrid(phi_rg, nu_rg)

    return phi_mesh, nu_mesh


class SphericalAmbisonicsVisualizer(object):
    def __init__(self, data, rate=22050, window=0.1, angular_res=2.0):
        self.window = window
        self.angular_res = angular_res
        self.data = data
        self.phi_mesh, self.nu_mesh = spherical_mesh(angular_res)
        mesh_p = [Position(phi, nu, 0.25, 'polar') for phi, nu in zip(self.phi_mesh.reshape(-1), self.nu_mesh.reshape(-1))]

        # set up decoder
        ambi_order = np.sqrt(data.shape[1]) - 1
        self.decoder = AmbiDecoder(mesh_p, AmbiFormat(ambi_order=ambi_order, sample_rate=rate), method='projection')

        # compute spherical energy averaged over consecutive chunks of "window" secs
        self.window_frames = int(self.window * rate)
        self.n_frames = data.shape[0] // self.window_frames
        self.output_rate = float(rate)// self.window_frames
        self.frame_dims = self.phi_mesh.shape
        self.cur_frame = -1

    def visualization_rate(self):
        return self.output_rate

    def mesh(self):
        return self.nu_mesh, self.phi_mesh

    def get_next_frame(self):
        self.cur_frame += 1
        if self.cur_frame >= self.n_frames:
            return None

        # Decode ambisonics on a grid of speakers
        chunk_ambi = self.data[self.cur_frame * self.window_frames:((self.cur_frame + 1) * self.window_frames), :]
        decoded = self.decoder.decode(chunk_ambi)

        # Compute RMS at each speaker
        rms = np.sqrt(np.mean(decoded ** 2, 0)).reshape(self.phi_mesh.shape)
        return np.flipud(rms)

    def loop_frames(self):
        while True:
            rms = self.get_next_frame()
            if rms is None:
                break
            yield rms

# PARAMETERS FOR BUILDING INTENSITY MAP
DURATION = 1.0 # may choose shorter DURATION because eventually only the first sample matters 
ANGULAR_RES = 5 # resolution for binning the estimated angles

IGIBSON_OFFSET = 0

def mp3d_example():
    s = Simulator(mode='iggui', image_width=512, image_height=512, device_idx=0)
    scene = StaticIndoorScene('17DRP5sb8fy')
    #scene = StadiumScene()
    s.import_scene(scene)



    acousticMesh = getMatterportAcousticMesh(s, "/cvgl/group/Gibson/matterport3d-downsized/v2/17DRP5sb8fy/sem_map.png")#AcousticMesh()
    #acousticMesh = AcousticMesh()
    #transparent_id =ResonanceMaterialToId["Transparent"]
    #Make mesh transparent so we only render direct sound
    #acousticMesh.materials = np.ones(acousticMesh.materials.shape) * transparent_id

    # Audio System Initialization, with reverb/reflections off
    audioSystem = AudioSystem(s, s.viewer, acousticMesh, is_Viewer=True, writeToFile=True, SR = 44100, num_probes=5, renderAmbisonics=True, renderReverbReflections=False)
    # -4.1,3.1,1.2
    obj = cube.Cube(pos=[-1, 1, 1.2], dim=[0.05, 0.05, 0.05], visual_only=True, mass=0, color=[1,0,0,1])
    s.import_object(obj)
    obj_id = obj.get_body_id()

    # Attach wav file to imported cube obj
    audioSystem.registerSource(obj_id, "250Hz_44100Hz.wav", enabled=True)
    # Ensure source continuously repeats
    audioSystem.setSourceRepeat(obj_id)
    audioSystem.setSourceNearFieldEffectGain(obj_id, 1.1)

    # Runs for 30 seconds, then saves output audio to file. 
    numberOfSteps = 3
    audio_window = np.zeros((16, 1470 * numberOfSteps))
    viewer_pos = audioSystem.get_pos()
    r = 1.5
    state = True
    
    for i in range(1000):
        theta = i / 20
        #obj.set_position([viewer_pos[0] + r*cos(theta), viewer_pos[1] + r*sin(theta), viewer_pos[2]])
        if i % 50 == 0:
            state = not state
            audioSystem.setSourceEnabled(obj_id, state)
        s.step()
        audioSystem.step()
        audio_window[:,(i % numberOfSteps) * 1470:((i % numberOfSteps)+1) * 1470] = np.asarray(audioSystem.ambisonic_output)
        if i % numberOfSteps == 0:
            ambiSpherical = SphericalAmbisonicsVisualizer(audio_window.transpose(), DEFAULT_RATE, window=numberOfSteps/30,
                                                        angular_res=ANGULAR_RES)

            azimuth_range = np.flip(np.arange(-180., 180, ANGULAR_RES), 0)
            # looping over chunks of sound, break after the very first chunk to get angles from the direct sound
            for frame_count, rms in enumerate(ambiSpherical.loop_frames()):
                # there is only 1 elevation angle i.e., 0 degrees, so use that with index 0
                estimated_angle = azimuth_range[np.argmax(rms[0, :])] + IGIBSON_OFFSET
                # just use the first chunk
                if frame_count == 0:
                    break

            listener_pos, listener_ori = audioSystem.get_pos(), audioSystem.get_ori()
            source_pos,_ = p.getBasePositionAndOrientation(obj_id)
            #print("estimated source-listener angle: {}".format(estimated_angle))    
            print("estimated max energy angle: %3d   " % (estimated_angle))
            #print(np.mean(audio_data))
            #if estimated_angle == 170:
            #    print(np.mean(audio_data))
            
            # NOTE: this azimuth angle is uncalibrated/relative, you need to calibrate it so that you get the absolute
            # angle as per the coordinate system you want it to be in. An easy way to do it is to look at adjacent
            # nodes and see what angle this code estimates and what angle it is as per your coordinate system and then
            # add the necessary offset to the estimate
    audioSystem.disconnect()
    s.disconnect()
    
def main():
    mp3d_example()
    #ig_example()

if __name__ == '__main__':
    main()