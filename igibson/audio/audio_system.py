from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.render.mesh_renderer.instances import InstanceGroup
from igibson.utils.utils import l2_distance
from igibson.objects import cube

import audio

import wave
import numpy as np
import pybullet as p
from scipy.io.wavfile import write

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
    def __init__(self, simulator, listener, acousticMesh, is_Viewer=False, writeToFile=False, SR=44100, num_probes=10):
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

        if acousticMesh.faces is None or acousticMesh.verts is None or acousticMesh.materials is None:
            raise ValueError('Invalid audioMesh')

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
        else:
            self.get_pos = self.listener.eyes.get_position
            self.get_ori = self.listener.eyes.get_orientation

        #TODO: Here we assume an integer number of audio frames per simulator time step. 
        #Usually true, but if not, is this even a problem?

        self.framesPerBuf =  int(SR / (1 / self.s.render_timestep)) 
        audio.InitializeSystem(self.framesPerBuf, SR)

        #Load scene mesh without dynamic objects
        audio.LoadMesh(int(acousticMesh.verts.size / 3), int(acousticMesh.faces.size / 3), acousticMesh.verts, acousticMesh.faces, acousticMesh.materials, 0.9) #Scattering coefficient needs tuning?

        #Get reverb and reflection properties at equally spaced point in grid along traversible map
        self.probe_key_to_pos_by_floor = []

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
                audio.RegisterReverbProbe(key, sample_position)
                self.probe_key_to_pos_by_floor[floor][key] = sample_position[:2]

        print("Finished computing reverb probes")
        self.current_probe_key = self.getClosestReverbProbe(self.get_pos())

        #We can only intelligently avoid doble-counting collisions with individual objects on interactive scenes
        self.alwaysCountCollisionIDs = set()
        self.single_occl_hit_per_obj = isinstance(self.scene, InteractiveIndoorScene)
        if self.single_occl_hit_per_obj:
            #Since the walls are all assigned one obj_id, we need to make sure not to automatically skip counting duplicate collisions with these ids
            for category in ["walls", "floors", "ceilings"]:
                for obj in self.s.scene.objects_by_category[category]: 
                    self.alwaysCountCollisionIDs.add(obj)

        self.sourceToEnabled, self.sourceToBuffer, self.sourceToRepeat,  self.sourceToResonanceID = {}, {}, {}, {}
        self.current_output, self.complete_output = [], []

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

    def registerSource(self, source_obj_id, audio_fname, enabled=False, repeat=True):
        print("Initializing source object " + str(source_obj_id) + " from file: " + audio_fname)
        if source_obj_id in self.sourceToEnabled:
            raise Exception('Object {} has already been registered with source {}, and we currently only support one audio stream per source.'.format(source_obj_id, audio_fname))

        buffer =  wave.open(audio_fname, 'rb')
        if buffer.getframerate() != self.SR:
            raise Exception('Object {} with source {} has SR {}, which does not match the system SR of {}.'.format(source_obj_id, audio_fname, buffer.getframerate(),self.SR))
        if buffer.getnchannels() != 1:
            raise Exception('Source {} has {} channels, 1 expected.'.format(audio_fname, buffer.getnchannels()))

        source_pos,_ = p.getBasePositionAndOrientation(source_obj_id)
        source_id = audio.InitializeSource(source_pos, 0.1, 10)
        self.sourceToResonanceID[source_obj_id] = source_id
        self.sourceToEnabled[source_obj_id] = enabled
        self.sourceToRepeat[source_obj_id] = repeat
        self.sourceToBuffer[source_obj_id] = buffer
        audio.SetSourceOcclusion(self.sourceToResonanceID[source_obj_id], 0)

    def setSourceEnabled(self, source_obj_id, enabled=True):
        self.sourceToEnabled[source_obj_id] = enabled
    
    def setSourceRepeat(self, source_obj_id, repeat=True):
        self.sourceToRepeat[source_obj_id] = repeat

    def readSource(self, source, nframes):
        #This conversion to numpy is inefficient and unnecessary
        #TODO: is np.int16 limiting?
        buffer = self.sourceToBuffer[source]
        return np.frombuffer(buffer.readframes(self.framesPerBuf), dtype=np.int16)

    def step(self):
        listener_pos = self.get_pos()
        for source, buffer in self.sourceToBuffer.items():
            if self.sourceToEnabled[source]:
                source_audio = self.readSource(source, self.framesPerBuf)
                if source_audio.size < self.framesPerBuf:
                    if self.sourceToRepeat[source]:
                        buffer.rewind()
                        audio_to_append = self.readSource(source, self.framesPerBuf - source_audio.size)
                        source_audio = np.append(source_audio, audio_to_append)
                    else:
                        num_pad = source_audio.size - self.framesPerBuf
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
                audio.SetSourceOcclusion(self.sourceToResonanceID[source], occl_hits)
                audio.ProcessSource(self.sourceToResonanceID[source], self.framesPerBuf, source_audio)

        audio.SetListenerPositionAndRotation(listener_pos, self.get_ori())
        closest_probe_key = self.getClosestReverbProbe(listener_pos)
        if closest_probe_key != self.current_probe_key:
            print("Updating Reverb/Reflection properties to probe " + closest_probe_key)
            audio.SetRoomPropertiesFromProbe(closest_probe_key)
            self.current_probe_key = closest_probe_key

        self.current_output = audio.ProcessListener(self.framesPerBuf)

        if self.writeToFile:
            self.complete_output.extend(self.current_output)
    
    def disconnect(self):
        deinterleaved_audio = np.array([self.complete_output[::2], self.complete_output[1::2]], dtype=np.int16).T
        write("Audio_Out.wav", self.SR, deinterleaved_audio)