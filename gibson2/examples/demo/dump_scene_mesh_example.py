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
import audio
from scipy.io.wavfile import write

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

    head_pos = np.array([2, 2, 1])#scene.get_random_point_by_room_type("kitchen")
    source_location = np.array([0, 0, 1])#scene.get_random_point_by_room_type("living_room")

    print(vert_flattened[:10])
    print(face_flattened[:10])

    out_arr = audio.InitializeFromMeshAndTest(vert.shape[0], face.shape[0],
    vert_flattened, face_flattened,
    material_indices,
    0.9, "ClapSound.wav", source_location, head_pos)

    if np.all((out_arr == 0)):
        print("Got all zero")
    print(out_arr)
    print(out_arr.shape)
    print(out_arr.dtype)
    

if __name__ == '__main__':
    main()
