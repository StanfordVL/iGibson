import cv2
import sys
import os
import numpy as np
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.profiler import Profiler
from gibson2.utils.assets_utils import get_ig_scene_path
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene


def main():

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_ig_scene_path('Rs_int'))

    s = Simulator(mode='headless', image_width=512, image_height=512, device_idx=0)
    scene = InteractiveIndoorScene('Rs_int', texture_randomization=False, object_randomization=False)
    s.import_ig_scene(scene)

    #vert, face = renderer.dump()

    #printf("Vertices Shape: "+ str(vert.shape))
    #printf("Faces Shape: "+ str(face.shape))

    #renderer.add_instance(0)
    #print(renderer.visual_objects, renderer.instances)
    #print(renderer.materials_mapping, renderer.mesh_materials)

    vert, face = s.renderer.dump()
    print("Vertices Shape: "+ str(vert.shape))
    print("Faces Shape: "+ str(face.shape))




if __name__ == '__main__':
    main()
