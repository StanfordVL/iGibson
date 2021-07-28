from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.render.mesh_renderer.instances import InstanceGroup

import audio

import wave
import numpy as np
import pybullet as p
from scipy.io.wavfile import write

#This is an inelegant (and potentially unsustainable) way to map materials, and should be directly unified
#with the C enums in resonance audio 

#enum MaterialName {
#  kTransparent = 0,
#  kAcousticCeilingTiles,
#  kBrickBare,
#  kBrickPainted,
#  kConcreteBlockCoarse,
#  kConcreteBlockPainted,
#  kCurtainHeavy,
#  kFiberGlassInsulation,
#  kGlassThin,
#  kGlassThick,
#  kGrass,
#  kLinoleumOnConcrete,
#  kMarble,
#  kMetal,
#  kParquetOnConcrete,
#  kPlasterRough,
#  kPlasterSmooth,
#  kPlywoodPanel,
#  kPolishedConcreteOrTile,
#  kSheetrock,
#  kWaterOrIceSurface,
#  kWoodCeiling,
#  kWoodPanel,
#  kUniform,
#  kNumMaterialNames
#};

#We hardcode these mappings because there is no clean way to directly access the materials of the iGibson mesh if they are loaded with
#randomization off
iGibsonToResonanceMaterialMap = {
    "walls": 5,             #kConcreteBlockPainted
    "floors": 22,           #kWoodPanel
    "ceilings": 5,          #kConcreteBlockPainted
    "window": 8             #kGlassThick
}


def getMesh(simulator):
    vert, face = simulator.renderer.dump()

    vert_flattened = np.empty((vert.size,), dtype=vert.dtype)
    vert_flattened[0::3] = vert[:,0]
    vert_flattened[1::3] = vert[:,1]
    vert_flattened[2::3] = vert[:,2]

    face_flattened = np.empty((face.size,), dtype=face.dtype)
    face_flattened[0::3] = face[:,0]
    face_flattened[1::3] = face[:,1]
    face_flattened[2::3] = face[:,2]

    material_indices = np.ones(face.shape[0]) * 22  #TODO: Implement material-specific customization

    return vert_flattened, face_flattened, material_indices

def dumpFromRenderer(renderer, obj_pb_ids):
    instances_vertices = []
    instances_faces = []
    len_v = 0
    for instance in renderer.instances:
        if instance.pybullet_uuid in obj_pb_ids:
            vertex_info, face_info = instance.dump()
            for v, f in zip(vertex_info, face_info):
                instances_vertices.append(v)
                instances_faces.append(f + len_v)
                len_v += len(v)
    instances_vertices = np.concatenate(instances_vertices, axis=0)
    instances_faces = np.concatenate(instances_faces, axis=0)

    return instances_vertices, instances_faces

def getSceneSkeletonMesh(simulator):
    vert, face, mat = [], [], []
    for category in ["walls", "floors", "ceilings", "window"]:
        for obj in simulator.scene.objects_by_category[category]:
            for id in obj.body_ids:
                obj_verts, obj_faces = dumpFromRenderer(simulator.renderer, [id])
                mat.extend([iGibsonToResonanceMaterialMap[category]] * len(obj_faces))
                vert.extend(obj_verts)
                face.extend(obj_faces)
   
    vert, face = np.asarray(vert), np.asarray(face)
    vert_flattened = np.empty((vert.size,), dtype=vert.dtype)
    vert_flattened[0::3] = vert[:,0]
    vert_flattened[1::3] = vert[:,1]
    vert_flattened[2::3] = vert[:,2]

    face_flattened = np.empty((face.size,), dtype=face.dtype)
    face_flattened[0::3] = face[:,0]
    face_flattened[1::3] = face[:,1]
    face_flattened[2::3] = face[:,2]

    material_indices = np.asarray(mat) 

    return vert_flattened, face_flattened, material_indices

class AudioSystem(object):
    """
    AudioSystem manages sound simulation.
    It manages a set of audio objects and their corresponding audio buffers.
    It also interfaces with ResonanceAudio to perform the simulaiton to the listener.
    """
    def __init__(self, simulator, listener, is_Viewer=False, writeToFile=False, SR=44100):
        """
        :param scene: iGibson scene
        :param pybullet: pybullet client
        :param simulator: Simulator object
        :param listener: Audio receiver, either a Viewer object or any subclass of BaseRobot
        :param SR: ResonanceAudio sample rate
        """
        assert isinstance(simulator.scene, InteractiveIndoorScene), 'AudioSystem can only be called with InteractiveIndoorScene loaded in the Simulator'

        self.scene = simulator.scene
        self.SR = SR
        self.s = simulator
        self.listener = listener
        #self.pyaud = pyaudio.PyAudio()
        self.writeToFile = writeToFile

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

        verts, faces, materials = getSceneSkeletonMesh(simulator)

        #TODO: Here we assume an integer number of audio frames per simulator time step. 
        #Usually true, but if not, is this even a problem?

        self.framesPerBuf =  int(SR / (1 / self.s.render_timestep)) 
        audio.InitializeSystem(self.framesPerBuf, SR)

        #Load scene mesh without dynamic objects
        audio.LoadMesh(int(verts.size / 3), int(faces.size / 3), verts, faces, materials, 0.9) #Scattering coefficient needs tuning?

        #Get reverb and reflection properties for each room
        for room_ins in self.s.scene.room_ins_name_to_ins_id.keys():
            #TODO: get a better sample position (center of room?)
            _, sample_position = self.s.scene.get_random_point_by_room_instance(room_ins)
            if is_Viewer:
                #add arbitrary height
                sample_position[2] += 1.7
            else:
                sample_position[2] += self.get_pos()[2]

            audio.RegisterReverbProbe(room_ins, sample_position)

        print("Finished computing reverb probes")
        self.room = self.s.scene.get_room_instance_by_point(np.array(self.get_pos())[:2])
        self.alwaysCountCollisionIDs = set()
        #Since the walls are all assigned one obj_id, we need to make sure not to automatically skip counting duplicate collisions with these ids
        for category in ["walls", "floors", "ceilings"]:
            for obj in self.s.scene.objects_by_category[category]: 
                self.alwaysCountCollisionIDs.add(obj)

        self.sourceToEnabled, self.sourceToBuffer, self.sourceToRepeat,  self.sourceToResonanceID = {}, {}, {}, {}
        self.current_output, self.complete_output = [], []


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
                    #print(rayHit)
                    hit_id = rayHit[0][0]
                    if hit_id == -1: #add collision with listener
                        break
                    if hit_id != source:
                        if hit_id not in hit_objects:
                            occl_hits += 1
                        if hit_id not in self.alwaysCountCollisionIDs:
                            hit_objects.add(hit_id)
                    hit_num += 1
                audio.SetSourceOcclusion(self.sourceToResonanceID[source], occl_hits)
                audio.ProcessSource(self.sourceToResonanceID[source], self.framesPerBuf, source_audio)

        audio.SetListenerPositionAndRotation(listener_pos, self.get_ori())
        curr_room = self.scene.get_room_instance_by_point(np.array(listener_pos[:2]))
        if curr_room != self.room and curr_room != None:
            print("Updating Room Properties to room " + curr_room)
            audio.SetRoomPropertiesFromProbe(curr_room)
            self.room = curr_room

        self.current_output = audio.ProcessListener(self.framesPerBuf)

        if self.writeToFile:
            self.complete_output.extend(self.current_output)
    
    def disconnect(self):
        deinterleaved_audio = np.array([self.complete_output[::2], self.complete_output[1::2]], dtype=np.int16).T
        write("Audio_Out.wav", self.SR, deinterleaved_audio)