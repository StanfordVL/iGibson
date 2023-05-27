from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.render.mesh_renderer.instances import InstanceGroup
from igibson.utils.utils import l2_distance
from igibson.objects import cube

import igibson.audio.default_config as config
# import igibson.audio as audio
from igibson.audio import audio
import librosa

from skimage.measure import block_reduce
import wave
import numpy as np
import pybullet as p
from scipy.io.wavfile import write
import scipy.io.wavfile as wavfile
import transforms3d as tf3d

# Mesh object required to be populated for AudioSystem
class AcousticMesh:
    # flattened vertex position array
    verts = None
    # flattened list of vertex indices of triangle faces
    faces = None
    # An array of integers, corresponding to the ResonanceAudio MaterialName enum value for each
    # face in the mesh
    materials = None

class AudioSystem(object):
    """
    AudioSystem manages sound simulation.
    It manages a set of audio objects and their corresponding audio buffers.
    It also interfaces with ResonanceAudio to perform the simulaiton to the listener.
    """
    def __init__(self,
                 simulator,
                 listener,
                 acousticMesh,
                 is_Viewer=False,
                 is_VR_Viewer=False,
                 writeToFile="",
                 SR= config.SAMPLE_RATE,
                 num_probes=config.NUM_REVERB_PROBES,
                 occl_multiplier=config.OCCLUSION_MULTIPLIER,
                 spectrogram_window_len=config.SPECTROGRAM_WINDOW_LEN,
                 renderAmbisonics=False,
                 renderReverbReflections=True,
                 stream_audio=False,
                 stream_input=False,
                 ):
        """
        :param scene: iGibson scene
        :param pybullet: pybullet client
        :param simulator: Simulator object
        :param listener: Audio receiver, either a Viewer object or any subclass of BaseRobot
        :param mesh: AudioMesh object, populated by caller. This mesh is primarily used for reverb/reflection baking.
        :param SR: ResonanceAudio sample rate
        :param num_probes: Determines number of reverb/reflections probes in the scene. Actual number is num_probes ^ 2
        """
        self.scene = simulator.scene
        self.SR = SR
        self.s = simulator
        self.listener = listener
        self.writeToFile = writeToFile
        self.renderAmbisonics = renderAmbisonics
        self.reverb = renderReverbReflections
        self.occl_multiplier = occl_multiplier
        self.occl_intensity = -1
        self.num_ambisonic_channels = 4
        _, self.bg_noise = wavfile.read("new_noise.wav")
        self.bg_noise = (self.bg_noise.reshape((-1,)) * 32768.0).astype(np.int16)

        def getViewerOrientation():
            #from numpy-quaternion github
            ct = np.cos(self.listener.theta / 2.0)
            cp = np.cos(self.listener.phi / 2.0)
            st = np.sin(self.listener.theta / 2.0)
            sp = np.sin(self.listener.phi / 2.0)
            return [-1*sp*st, st*cp, sp*ct, cp*ct]

        if is_Viewer:
            self.get_pos = lambda: [self.listener.px, self.listener.py, self.listener.pz]
            self.get_ori = getViewerOrientation 
        elif is_VR_Viewer:
            self.get_pos = lambda: self.s.get_data_for_vr_device("hmd")[1]
            def get_ori():
                get_vr_ori = lambda: self.s.get_data_for_vr_device("hmd")[2]
                lis_ori = get_vr_ori()
                start = [lis_ori[3], lis_ori[0],lis_ori[1],lis_ori[2]]
                delta = tf3d.quaternions.axangle2quat([0, 1, 0], np.pi/2)
                final = tf3d.quaternions.qmult(start, delta)
                delta2 = tf3d.quaternions.axangle2quat([0, 0, 1], -np.pi/2)
                final = tf3d.quaternions.qmult(final, delta2)
                final = [final[1], final[2],final[3],final[0]]
                return final
            self.get_ori = get_ori
        else:
            def get_pos():
                pose = np.zeros((3,))
                pose[:2] = self.listener.eyes.get_position()[:2]
                pose[2] = 0.73
                return pose
            self.get_pos = get_pos #self.listener.eyes.get_position

            def get_ori():
                lis_ori = self.listener.eyes.get_orientation()
                # convert to [w,x,y,z]
                start = [lis_ori[3], lis_ori[0],lis_ori[1],lis_ori[2]]
                # rotate along y axis
                delta = tf3d.quaternions.axangle2quat([0, 1, 0], np.pi/2)
                final = tf3d.quaternions.qmult(start, delta)
                # rotate along 
                delta2 = tf3d.quaternions.axangle2quat([0, 0, 1], -np.pi/2)
                final = tf3d.quaternions.qmult(final, delta2)
                final = [final[1], final[2],final[3],final[0]]
                return final
            self.get_ori = get_ori#self.listener.eyes.get_orientation


        #TODO: Here we assume an integer number of audio frames per simulator time step. 
        #Usually true, but if not, is this even a problem?

        self.framesPerBuf =  int(SR / (1 / self.s.render_timestep))
        # spectrogram taken over longer time windows
        if self.renderAmbisonics:
            self.curr_audio_by_channel = np.zeros((self.num_ambisonic_channels, self.framesPerBuf))
            self.window_by_channel = np.zeros((self.num_ambisonic_channels, int(SR * spectrogram_window_len)))
        else:
            self.curr_audio_by_channel = np.zeros((2, self.framesPerBuf))
            self.window_by_channel = np.zeros((2, int(SR * spectrogram_window_len)))

        audio.InitializeSystem(self.framesPerBuf, SR)

        #Get reverb and reflection properties at equally spaced point in grid along traversible map
        self.probe_key_to_pos_by_floor, self.current_probe_key = [], None
        if self.reverb:
            if acousticMesh.faces is None or acousticMesh.verts is None or acousticMesh.materials is None:
                raise ValueError('Invalid audioMesh')
            #Load scene mesh without dynamic objects
            audio.LoadMesh(int(acousticMesh.verts.size / 3), int(acousticMesh.faces.size / 3), acousticMesh.verts, acousticMesh.faces, acousticMesh.materials, 0.9) #Scattering coefficient needs tuning?
            points_grid = self.scene.get_points_grid(num_probes)
            for floor in points_grid.keys():
                self.probe_key_to_pos_by_floor.append({})
                for i, sample_position in enumerate(points_grid[floor]):
                    key = "floor " + str(floor) + " probe " + str(i)
                    if is_Viewer:
                        #add arbitrary height
                        sample_position[2] += 1.7
                    else:
                        sample_position[2] += self.get_pos()[2]
                    audio.RegisterReverbProbe(key, sample_position, *config.REV_PROBE_PARAMS)
                    self.probe_key_to_pos_by_floor[floor][key] = sample_position[:2]

            self.current_probe_key = self.getClosestReverbProbe(self.get_pos())
            audio.SetRoomPropertiesFromProbe(self.current_probe_key)
            # Save some memory
            audio.DeleteMesh()
        else:
            audio.DisableRoomEffects()

        #We can only intelligently avoid doble-counting collisions with individual objects on interactive scenes
        self.alwaysCountCollisionIDs = set()
        self.single_occl_hit_per_obj = isinstance(self.scene, InteractiveIndoorScene)
        if self.single_occl_hit_per_obj:
            #Since the walls are all assigned one obj_id, we need to make sure not to automatically skip counting duplicate collisions with these ids
            for category in ["walls", "floors", "ceilings"]:
                for obj in self.s.scene.objects_by_category[category]: 
                    self.alwaysCountCollisionIDs.add(obj)

        self.sourceToEnabled, self.sourceToBuffer, self.sourceToRepeat,  self.sourceToResonanceID = {}, {}, {}, {}
        self.current_output, self.complete_output = [0]*(4*self.framesPerBuf), []

        # Try to stream audio live
        if stream_audio or stream_input:
            import pyaudio
            self.streaming_input = []
            # self.mic_audio = []
            def pyaudOutputCallback(in_data, frame_count, time_info, status):
                return (bytes(self.current_output), pyaudio.paContinue)
            def pyaudInputCallback(in_data, frame_count, time_info, status):
                b = frame_count
                a = len(in_data)
                self.streaming_input = in_data
                return (in_data, pyaudio.paContinue)
            pyaud = pyaudio.PyAudio()
            info = pyaud.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                if (pyaud.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print("Input Device id ", i, " - ", pyaud.get_device_info_by_host_api_device_index(0, i).get('name'))
            self.out_stream = pyaud.open(rate=self.SR, frames_per_buffer=self.framesPerBuf, format=pyaudio.paInt16, channels=2, output=True, stream_callback=pyaudOutputCallback)
            if stream_input:
                self.in_stream = pyaud.open(rate=self.SR, frames_per_buffer=self.framesPerBuf, input_device_index = 1,format=pyaudio.paInt16, channels=1, input=True, stream_callback=pyaudInputCallback)
                # in_stream.start_stream()
    def getClosestReverbProbe(self, pos):
        floor = 0
        for i in range(len(self.scene.floor_heights)):
            if self.scene.floor_heights[i] > pos[2]:
                floor = i - 1
                break
        if floor < 0:
            print("Floor height error, cannot match closest reverb probe")
        min_dist, min_probe = np.inf, None
        #TODO: This is very inefficient
        for probe_key, probe_pos in self.probe_key_to_pos_by_floor[floor].items():
            dist = l2_distance(probe_pos, pos[:2])
            if dist < min_dist:
                min_dist = dist
                min_probe = probe_key
        return min_probe

    def registerSource(self,
                       source_obj_id,
                       audio_fname,
                       enabled=False,
                       repeat=True,
                       min_distance=config.DEFAULT_MIN_FALLOFF_DISTANCE,
                       max_distance=config.DEFAULT_MAX_FALLOFF_DISTANCE,
                       source_gain=config.DEFAULT_SOURCE_GAIN,
                       near_field_gain=config.DEFAULT_NEAR_FIELD_GAIN,
                       reverb_gain=config.DEFAULT_ROOM_EFFECTS_GAIN,
                    ):
        print("Initializing source object " + str(source_obj_id) + " from file: " + audio_fname)
        print("near field gain is", near_field_gain)
        if source_obj_id in self.sourceToEnabled:
            raise Exception('Object {} has already been registered with source {}, and we currently only support one audio stream per source.'.format(source_obj_id, audio_fname))

        source_pos,_ = p.getBasePositionAndOrientation(source_obj_id)
        source_id = audio.InitializeSource(source_pos, min_distance, max_distance, source_gain, near_field_gain, reverb_gain)
        buffer = None
        if audio_fname:
            buffer =  wave.open(audio_fname, 'rb')
            if buffer.getframerate() != self.SR:
                raise Exception('Object {} with source {} has SR {}, which does not match the system SR of {}.'.format(source_obj_id, audio_fname, buffer.getframerate(),self.SR))
            if buffer.getnchannels() != 1:
                raise Exception('Source {} has {} channels, 1 expected.'.format(audio_fname, buffer.getnchannels()))

        self.sourceToResonanceID[source_obj_id] = source_id
        self.sourceToEnabled[source_obj_id] = enabled
        self.sourceToRepeat[source_obj_id] = repeat
        self.sourceToBuffer[source_obj_id] = buffer
        audio.SetSourceOcclusion(self.sourceToResonanceID[source_obj_id], 0)

    def setSourceEnabled(self, source_obj_id, enabled=True):
        self.sourceToEnabled[source_obj_id] = enabled
    
    def setSourceRepeat(self, source_obj_id, repeat=True):
        self.sourceToRepeat[source_obj_id] = repeat

    def setSourceNearFieldEffectGain(self, source_obj_id, gain):
        audio.SetNearFieldEffectGain(self.sourceToResonanceID[source_obj_id], gain)

    def readSource(self, source, nframes):
        #This conversion to numpy is inefficient and unnecessary
        #TODO: is np.int16 limiting?
        buffer = self.sourceToBuffer[source]
        if buffer is None:
            # streaming input
            return np.frombuffer(self.streaming_input, dtype=np.int16)
        return np.frombuffer(buffer.readframes(nframes), dtype=np.int16)

    def step(self):
        listener_pos = self.get_pos()
        for source, buffer in self.sourceToBuffer.items():
            if self.sourceToEnabled[source]:
                source_audio = self.readSource(source, self.framesPerBuf)
                # self.mic_audio.extend(source_audio.tolist())
                if source_audio.size < self.framesPerBuf:
                    if self.sourceToRepeat[source] and buffer is not None:
                        buffer.rewind()
                        audio_to_append = self.readSource(source, self.framesPerBuf - source_audio.size)
                        source_audio = np.append(source_audio, audio_to_append)
                    else:
                        num_pad = self.framesPerBuf - source_audio.size
                        source_audio = np.pad(source_audio, (0, num_pad), 'constant')
                        self.setSourceEnabled(source, enabled=False)
                #TODO: Source orientation!
                source_pos,_ = p.getBasePositionAndOrientation(source)
                audio.SetSourcePosition(self.sourceToResonanceID[source], [source_pos[0], source_pos[1], source_pos[2]])
                occl_hits, hit_num = 0, 0
                hit_objects = set()
                while hit_num < 12:
                    rayHit = p.rayTestBatch([source_pos], [listener_pos], reportHitNumber=hit_num, fractionEpsilon=0.01)
                    hit_id = rayHit[0][0]
                    if hit_id == -1: #add collision with listener
                        break
                    if hit_id != source:
                        if hit_id not in hit_objects:
                            occl_hits += 1
                        if self.single_occl_hit_per_obj and hit_id not in self.alwaysCountCollisionIDs:
                            hit_objects.add(hit_id)
                    hit_num += 1
                self.occl_intensity = occl_hits*self.occl_multiplier
                audio.SetSourceOcclusion(self.sourceToResonanceID[source], self.occl_intensity)
                audio.ProcessSource(self.sourceToResonanceID[source], self.framesPerBuf, source_audio)
            else:
                audio.ProcessSource(self.sourceToResonanceID[source], self.framesPerBuf, np.zeros(self.framesPerBuf, dtype=np.int16))
        # print("eye", self.get_ori())
        audio.SetListenerPositionAndRotation(listener_pos, self.get_ori())
        if self.reverb:
            closest_probe_key = self.getClosestReverbProbe(listener_pos)
            if closest_probe_key != self.current_probe_key:
                audio.SetRoomPropertiesFromProbe(closest_probe_key)
                self.current_probe_key = closest_probe_key

        self.current_output = audio.ProcessListener(self.framesPerBuf)
        # adding noise
        # start_idx = int(np.random.choice(int(self.bg_noise.shape[0] - len(self.current_output)), 1)[0])
        # if start_idx % 2 != 0:
        #     start_idx -= 1
        # noise = self.bg_noise[start_idx:(start_idx + len(self.current_output))]
        # self.current_output = self.current_output + noise * 0.855 #
        # self.current_output[self.current_output > 32768] = 32768
        # self.current_output[self.current_output < -32768] = -32768

        if self.renderAmbisonics:
            self.ambisonic_output = audio.RenderAmbisonics(self.framesPerBuf)
            self.curr_audio_by_channel = np.array(self.ambisonic_output[:self.num_ambisonic_channels])
        else:
            self.curr_audio_by_channel[0] = np.array(self.current_output[::2], dtype=np.float32, order='C') / 32768.0
            self.curr_audio_by_channel[1] = np.array(self.current_output[1::2], dtype=np.float32, order='C') / 32768.0

        if self.writeToFile != "":
            self.complete_output.extend(self.current_output)
    
    def reset(self):
        self.save_audio()
        
        for source, _ in self.sourceToBuffer.items():
            audio.DestroySource(self.sourceToResonanceID[source])

        self.sourceToEnabled, self.sourceToBuffer, self.sourceToRepeat,  self.sourceToResonanceID = {}, {}, {}, {}
        self.current_output, self.complete_output = [], []
    
    def save_audio(self):
        if self.writeToFile != "":
            deinterleaved_audio = np.array([self.complete_output[::2], self.complete_output[1::2]], dtype=np.int16).T
            # print(deinterleaved_audio)
            write(self.writeToFile + '.wav', self.SR, deinterleaved_audio)
            # write('supp_video_results/test_mic.wav', self.SR , np.array(self.mic_audio, dtype=np.int16))
    
    def get_spectrogram(self):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(2, 2), func=np.mean)
            return stft

        spectrogram_per_channel = []
        for mono_idx in range(self.curr_audio_by_channel.shape[0]):
            self.window_by_channel[mono_idx] = np.append(self.window_by_channel[mono_idx,self.framesPerBuf:], self.curr_audio_by_channel[mono_idx])
            spectrogram_per_channel.append(np.log1p(compute_stft(self.window_by_channel[mono_idx])))

        spectrogram = np.stack(spectrogram_per_channel, axis=-1)

        return spectrogram

    def disconnect(self):
        self.save_audio()    
        audio.ShutdownSystem()