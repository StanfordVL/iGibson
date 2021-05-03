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
import pyaudio
import audio
import wave
from scipy.io.wavfile import read, write

def main():

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_ig_scene_path('Rs_int'))

    s = Simulator(mode='headless', image_width=512, image_height=512, device_idx=0)
    scene = StadiumScene()#InteractiveIndoorScene('Rs_int', texture_randomization=False, object_randomization=False)
    s.import_scene(scene)#import_ig_scene(scene)

    #vert, face = renderer.dump()

    #printf("Vertices Shape: "+ str(vert.shape))
    #printf("Faces Shape: "+ str(face.shape))

    #renderer.add_instance(0)
    #print(renderer.visual_objects, renderer.instances)
    #print(renderer.materials_mapping, renderer.mesh_materials)

    vert, face = s.renderer.dump()
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

    #_, head_pos = scene.get_random_point_by_room_type("kitchen")
    #_, source_location = scene.get_random_point_by_room_type("living_room")

    head_pos = np.array([1, 1, 1])#scene.get_random_point_by_room_type("kitchen")
    source_location = np.array([0, 0, 1])#scene.get_random_point_by_room_type("living_room")

    print(vert_flattened[:10])
    print(face_flattened[:10])

    #out_arr = audio.InitializeFromMeshAndTest(vert.shape[0], face.shape[0],
    #vert_flattened, face_flattened,
    #material_indices,
    #0.9, "ClapSound.wav", source_location, head_pos)

    framesPerBuf = 480
    audio.InitializeSystem(framesPerBuf, 44100)
    audio.LoadMesh(vert.shape[0], face.shape[0], vert_flattened, face_flattened, material_indices, 0.9, source_location)
    source_id = audio.InitializeSource(source_location, 0.1, 10)
    audio.SetListenerPosition(head_pos)
    

    sr, wav_in = read("440Hz_44100Hz.wav")
    print(sr)
    #wav_out = wave.open("ClapOut.wav", mode='wb')
    #print(wav.getparams())
    #wav_out.setnchannels(2)
    #wav_out.setframerate(48000)
    #wav_out.setsampwidth(2)

    print(wav_in.size)

    num_pad = wav_in.size % framesPerBuf
    wav_in_padded = np.pad(wav_in, (0, num_pad), 'constant')
    audio_to_write =  np.array([])

    print(wav_in_padded.size)

    head_pos_int = np.ones((3,500))
    head_pos_int[0,:] = np.linspace(100,-100, num=500)
    head_pos_int[1,:] = np.linspace(100,-100, num=500)

    idx = 0
    for pos_idx in range(head_pos_int.shape[1]):
        if idx >= wav_in_padded.size:
           idx=0
        audio.SetListenerPosition(head_pos_int[:,pos_idx])
        out_audio = audio.ProcessSourceAndListener(source_id, framesPerBuf, wav_in_padded[idx:idx+framesPerBuf])
        idx += framesPerBuf
        audio_to_write = np.append(audio_to_write, out_audio)#audio_to_write[idx*2: idx*2 + 64] = out_audio
        
    deinterleaved_audio = np.array([audio_to_write[::2], audio_to_write[1::2]], dtype=np.int16).T
    #deinterleaved_audio[:, 0] = audio_to_write[::2]
    #deinterleaved_audio[:, 1] = audio_to_write[1::2]
    write("440Hz_44100Hz_Out.wav", 44100, deinterleaved_audio)



#    p = pyaudio.PyAudio()
#    idx = 0
#    pos_idx = 0
##    def callback(in_data, frame_count, time_info, status):
##        if idx >= wav_in_padded.size:
#           idx=0
#        if pos_idx == 2000:
##            pos_idx = 0
#        audio.SetListenerPosition(head_pos[:,pos_idx])
#        out_audio = audio.ProcessSourceAndListener(source_id, frame_count, wav_in_padded[idx:idx+frame_count])
#        idx += frame_count
#        pos_idx+=1
#        return (out_audio.tobytes(), pyaudio.paContinue)


#    stream = p.open(rate=48000, format=pyaudio.paInt16, channels=2, output=True,stream_callback=callback)



#    while stream.is_active():
#        time.sleep(0.1)

    #received_audio = np.zeros(32)
    #read_audio = wav.readframes(32)
    #while read_audio != b'':
    #    read_audio = wav.readframes(32)
    #    print(read_audio)
    
    #while 
    #audio.ProcessSourceAndListener(source_id, 32, py::array_t<int16> input_arr)





    
    if np.all((audio_to_write == 0)):
        print("Got all zero")

if __name__ == '__main__':
    main()
