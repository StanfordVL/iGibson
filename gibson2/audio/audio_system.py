from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator


import audio

import pyaudio
import wave
import numpy as np





    
def getSceneSkeletonMesh(self, scene_id):
    static_scene = InteractiveIndoorScene(  scene_id, 
                                            texture_randomization=False,
                                            object_randomization=False,
                                            load_object_categories=["walls", "floors", "ceilings", "door", "window"])
    static_s = Simulator(mode='headless', device_idx=0)
    static_s.import_ig_scene(static_scene)

    vert, face = static_s.renderer.dump()

    vert_flattened = np.empty((vert.size,), dtype=vert.dtype)
    vert_flattened[0::3] = vert[:,0]
    vert_flattened[1::3] = vert[:,1]
    vert_flattened[2::3] = vert[:,2]

    face_flattened = np.empty((face.size,), dtype=face.dtype)
    face_flattened[0::3] = face[:,0]
    face_flattened[1::3] = face[:,1]
    face_flattened[2::3] = face[:,2]

    material_indices = np.ones(face.shape[0]) * 22  #TODO: Implement material-specific customization

    static_s.disconnect()

    return vert_flattened, face_flattened, material_indices


class AudioSystem(object):
    """
    AudioSystem manages sound simulation.
    It manages a set of audio objects and their corresponding audio buffers.
    It also interfaces with ResonanceAudio to perform the simulaiton to the listener.
    """
    def __init__(self, scene, pybullet, simulator, listener, is_Viewer=False, SR = 44100):
        """
        :param scene: iGibson scene
        :param pybullet: pybullet client
        :param simulator: Simulator object
        :param listener: Audio receiver, either a Viewer object or any subclass of BaseRobot
        :param SR: ResonanceAudio sample rate
        """
        self.scene = scene
        self.SR = SR
        self.p = pybullet
        self.s = simulator
        self.listener = listener
        self.pyaud = pyaudio.PyAudio()

        if is_Viewer:
            self.get_pos = lambda: [self.listener.px, self.listener.py, self.listener.pz]
        else:
            #TODO: Should this position be directly sourced from URDF, or is `eyes' a reasonable compromise
            self.get_pos = self.listener.eyes.get_position


        assert isinstance(scene, InteractiveIndoorScene), 'AudioSystem can only be called with InteractiveIndoorScene'

        #TODO: Implement unique materials
        verts, faces, materials = getSceneSkeletonMesh(scene.scene_id)

        #TODO: Here we assume an integer number of audio frames per simulator time step. 
        #Usually true, but if not, is this even a problem?

        framesPerBuf =  int(SR / (1 / self.s.render_timestep)) 
        audio.InitializeSystem(framesPerBuf, SR)

        #Load scene mesh without dynamic objects
        audio.LoadMesh(int(verts.size / 3), int(faces.size / 3), verts, faces, materials, 0.9) #Scattering coefficient needs tuning?

        #Get reverb and reflection properties for each room
        for room_ins in scene.room_ins_name_to_ins_id.keys():
            #TODO: get a better sample position (center of room?)
            sample_position = scene.get_random_point_by_room_instance(room_ins)
            if is_Viewer:
                #add arbitrary height
                sample_position[2] += 1.7
            else:
                sample_position[2] += self.get_pos()
                
            audio.RegisterReverbProbe(room_ins, sample_position)
      

        self.alwaysCountCollisionIDs = set()
        #Since the walls are all assigned one obj_id, we need to make sure not to automatically skip counting duplicate collisions with these ids
        for category in ["walls", "floors", "ceilings"]:
            for obj in scene.objects_by_category[category]: 
                self.alwaysCountCollisionIDs.add(obj)

        self.sourceToEnabled, self.sourceToBuffer, self.sourceToRepeat,  self.sourceToResonanceID = {}, {}, {}


    def registerSource(self, source_obj_id, audio_fname, enabled=False, repeat=True):
        if source_obj_id in self.sourceToEnabled:
            raise Exception('Object {} has already been registered with source {}, and we currently only support one audio stream per source.'.format(source_obj_id, audio_fname))

        buffer =  wave.open(audio_fname, 'rb')
        if buffer.getframerate() != self.SR:
            raise Exception('Object {} with source {} has SR {}, which does not match the system SR of {}.'.format(source_obj_id, audio_fname, buffer.getframerate(),self.SR))
        if buffer.getnchannels() != 1:
            raise Exception('Source {} has {} channels, 1 expected.'.format(audio_fname, buffer.getnchannels()))

        source_pos,_ = self.p.getBasePositionAndOrientation(source_obj_id)
        source_id = audio.InitializeSource(source_pos, 0.1, 10)
        self.sourceToResonanceID[source_obj_id] = source_id
        self.sourceToEnabled[source_obj_id] = enabled
        self.sourceToRepeat[source_obj_id] = repeat
        self.sourceToBuffer[source_obj_id] = buffer


    def setSourceEnabled(self, source_obj_id, enabled=True):
        self.sourceToEnabled[source_obj_id] = enabled
    
    def setSourceRepeat(self, source_obj_id, repeat=True):
        self.sourceToRepeat[source_obj_id] = repeat