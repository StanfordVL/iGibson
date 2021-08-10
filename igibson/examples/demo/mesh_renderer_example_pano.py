import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.utils.assets_utils import get_scene_path


def main():
    global _mouse_ix, _mouse_iy, down, view_direction

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path("Rs"), "mesh_z_up.obj")

    settings = MeshRendererSettings(enable_pbr=False)
    renderer = MeshRenderer(width=512, height=512, rendering_settings=settings)
    renderer.load_object(model_path)

    renderer.add_instance(0)
    print(renderer.visual_objects, renderer.instances)
    print(renderer.materials_mapping, renderer.mesh_materials)

    px = 0
    py = 0.2

    camera_pose = np.array([px, py, 0.5])
    view_direction = np.array([0, -1, -0.3])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)

    img = renderer.get_equi()
    print(img.shape)
    plt.imshow(img)
    plt.show()

    renderer.release()


if __name__ == "__main__":
    main()
