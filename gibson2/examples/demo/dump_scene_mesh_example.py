import cv2
import sys
import os
import numpy as np
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.profiler import Profiler
from gibson2.utils.assets_utils import get_ig_scene_path


def main():
    global _mouse_ix, _mouse_iy, down, view_direction

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_ig_scene_path('Rs_int'))

    renderer = MeshRenderer(width=512, height=512)
    renderer.load_object(model_path)

    vert, face = renderer.dump()

    printf("Vertices Shape: "+ str(vert.shape))
    printf("Faces Shape: "+ str(face.shape))

    renderer.add_instance(0)
    print(renderer.visual_objects, renderer.instances)
    print(renderer.materials_mapping, renderer.mesh_materials)


    printf("Vertices Shape: "+ str(vert.shape))
    printf("Faces Shape: "+ str(face.shape))


    renderer.release()


if __name__ == '__main__':
    main()
