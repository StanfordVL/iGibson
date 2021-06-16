import cv2
import sys
import os
import numpy as np
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.profiler import Profiler
from gibson2.utils.assets_utils import get_ig_scene_path
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.stadium_scene import StadiumScene
from gibson2.objects import cube
import pyaudio
import audio
import wave
import pybullet as p
import time
from scipy.io.wavfile import read, write


SR = 44100



def getSceneSkeleton(scene_name):
    static_scene = InteractiveIndoorScene(scene_name, texture_randomization=False, object_randomization=False, load_object_categories=["walls", "floors", "ceilings", "door", "window"])
    static_s = Simulator(mode='headless', device_idx=0)
    static_s.import_ig_scene(static_scene)

    vert, face = static_s.renderer.dump()
    print("Vertices Shape: "+ str(vert.shape))
    print("Faces Shape: "+ str(face.shape))

    vert_flattened = np.empty((vert.size,), dtype=vert.dtype)
    vert_flattened[0::3] = vert[:,0]
    vert_flattened[1::3] = vert[:,1]
    vert_flattened[2::3] = vert[:,2]

    face_flattened = np.empty((face.size,), dtype=face.dtype)
    face_flattened[0::3] = face[:,0]
    face_flattened[1::3] = face[:,1]
    face_flattened[2::3] = face[:,2]

    material_indices = np.ones(face.shape[0]) * 22

    assert(vert_flattened.size / 3 == vert.shape[0])
    assert(face_flattened.size / 3 == face.shape[0])

    static_s.disconnect()

    return vert_flattened, face_flattened, material_indices


obj_id = -1
timestep = 0
def main():
    global obj_id
    global alwaysCountCollisionIDs

    #if len(sys.argv) > 1:
    #    model_path = sys.argv[1]
    #else:
    #    model_path = os.path.join(get_ig_scene_path('Rs_int'))
    wf = wave.open("440Hz_44100Hz.wav", 'rb')
    print(wf.getparams())

    verts, faces, materials = getSceneSkeleton('Rs_int')

    s = Simulator(mode='iggui', image_width=512, image_height=512, device_idx=0)
    scene = InteractiveIndoorScene('Rs_int', texture_randomization=False, object_randomization=False)#StadiumScene()
    s.import_ig_scene(scene)#s.import_scene(scene)#
    alwaysCountCollisionIDs = set()
    for category in ["walls", "floors", "ceilings"]:
        for obj in scene.objects_by_category[category]:
            alwaysCountCollisionIDs.add(obj)
    
    

    _,source_location = scene.get_random_point_by_room_type("living_room")
    source_location[2] = 1.7
    obj = cube.Cube(pos=source_location, dim=[0.2, 0.2, 0.2], visual_only=False, mass=0.5, color=[255, 0, 0, 1])
    obj_id = s.import_object(obj)


    #_, head_pos = scene.get_random_point_by_room_type("kitchen")
    #_, source_location = scene.get_random_point_by_room_type("living_room")

    #head_pos = np.array([1, 1, 1])#scene.get_random_point_by_room_type("kitchen")
    #source_location = np.array([0, 0, 1])#scene.get_random_point_by_room_type("living_room")

    #head_positions = [scene.get_random_point_by_room_type("living_room")[1], scene.get_random_point_by_room_type("bedroom")[1]]

    #out_arr = audio.InitializeFromMeshAndTest(vert.shape[0], face.shape[0],
    #vert_flattened, face_flattened,
    #material_indices,
    #0.9, "ClapSound.wav", source_location, head_pos)

    framesPerBuf =  int(SR / (1 / s.render_timestep)) #DEAL WITH THIS LATER
    audio.InitializeSystem(framesPerBuf, SR)
    audio.LoadMesh(int(verts.size / 3), int(faces.size / 3), verts, faces, materials, 0.9, source_location)
    source_id = audio.InitializeSource(source_location, 0.1, 10)
    #audio.SetSourceListenerDirectivity(source_id, 0.3, 1.1)
    #audio.SetListenerPosition(head_pos)
    

    sr, wav_in = read("440Hz_44100Hz.wav")
    print(sr)
    print(wav_in.size)

    num_pad = wav_in.size % framesPerBuf
    wav_in_padded = np.pad(wav_in, (0, num_pad), 'constant')
    audio_to_write =  np.array([])


    #print(head_positions)

    #head_pos_int = np.ones((3,500))
    #head_pos_int[0,:] = np.linspace(head_positions[0][0],head_positions[1][0], num=500)
    #head_pos_int[1,:] = np.linspace(head_positions[0][1],head_positions[1][1], num=500)

    #idx = 0
    #for pos_idx in range(head_pos_int.shape[1]):
    #    if idx >= wav_in_padded.size:
    #       idx=0
    #    audio.SetListenerPosition(head_pos_int[:,pos_idx])
    #    out_audio = audio.ProcessSourceAndListener(source_id, framesPerBuf, wav_in_padded[idx:idx+framesPerBuf])
    #    idx += framesPerBuf
    #    audio_to_write = np.append(audio_to_write, out_audio)#audio_to_write[idx*2: idx*2 + 64] = out_audio
        
    #deinterleaved_audio = np.array([audio_to_write[::2], audio_to_write[1::2]], dtype=np.int16).T
    #deinterleaved_audio[:, 0] = audio_to_write[::2]
    #deinterleaved_audio[:, 1] = audio_to_write[1::2]
    #write("440Hz_44100Hz_Out.wav", SR, deinterleaved_audio)

    print("Initing pyaudio")
    idx = 0
    pyaud = pyaudio.PyAudio()
    print("Inited pyaudio")
    
    def getNextNFrames(n):
        global idx
        print(idx)
        if idx >= wav_in_padded.size:
           idx=0
        aud =  wav_in_padded[idx:idx+n]
        idx += n
        return aud


    def callback(in_data, frame_count, time_info, status):
        global obj_id
        global timestep
        global alwaysCountCollisionIDs
        start_t = time.time()
        source_pos,_ = p.getBasePositionAndOrientation(obj_id)
        listener_pos = [s.viewer.px, s.viewer.py, s.viewer.pz]
        audio.SetSourcePosition(source_id, [source_pos[0], source_pos[1], source_pos[2]])
        #from numpy-quaternion github
        ct = np.cos(s.viewer.theta / 2.0)
        cp = np.cos(s.viewer.phi / 2.0)
        st = np.sin(s.viewer.theta / 2.0)
        sp = np.sin(s.viewer.phi / 2.0)
        audio.SetListenerPositionAndRotation(listener_pos, [-1*sp*st, st*cp, sp*ct,cp*ct])
        occl_hits, hit_num = 0, 0
        hit_ids = set()
        while hit_num < 12:
            rayHit = p.rayTestBatch([source_pos], [listener_pos], reportHitNumber=hit_num, fractionEpsilon=0.01)
            #print(rayHit)
            hit_id = rayHit[0][0]
            if hit_id == -1: #add collision with listener
                break
            if hit_id != obj_id:
                if hit_id not in hit_ids:
                    occl_hits += 1
                if hit_id not in alwaysCountCollisionIDs:
                    hit_ids.add(hit_id)
            hit_num += 1
        #print(occl_hits)
        #if rayHit and occl_hits <= 16:
            #print(rayHit[0])
            #print(rayHit[0][0])
        #    if rayHit[0][0] != obj_id:
        #        occl_hits += 1
       # 	rayHit = p.rayTest(source_pos, listener_pos)
        #	print(tmp)
        data = wf.readframes(frame_count)
        if len(data) < frame_count*wf.getsampwidth(): 
            wf.rewind()
            data = wf.readframes(frame_count)
        out_audio = audio.ProcessSourceAndListener(source_id, frame_count, np.frombuffer(data, dtype=np.int16))#This is inefficient

        if timestep % 10 == 0:
            #print("Left volume  = " + str(np.linalg.norm(out_audio[0::2])))
            #print("Right volume = " + str(np.linalg.norm(out_audio[1::2])))
            print(occl_hits)
            print(time.time() - start_t)
            timestep = 0
        timestep += 1
        return (out_audio.tobytes(), pyaudio.paContinue)


    stream = pyaud.open(rate=SR, frames_per_buffer=framesPerBuf, format=pyaudio.paInt16, channels=2, output=True,stream_callback=callback)


    np.random.seed(0)
    _,(px,py,pz) = scene.get_random_point()
    s.viewer.px = px
    s.viewer.py = py
    s.viewer.pz = 1.7
    s.viewer.update()
    
    stream.start_stream()
    while True:
        s.step()
    s.disconnect()
    


if __name__ == '__main__':
    main()
