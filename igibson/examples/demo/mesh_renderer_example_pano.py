import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.utils.assets_utils import get_scene_path


def main():
    global _mouse_ix, _mouse_iy, down, view_direction

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path("Allensville"), "mesh_z_up.obj")

    settings = MeshRendererSettings(enable_pbr=False)
    renderer = MeshRenderer(width=1024, height=1024, rendering_settings=settings)
    renderer.load_object(model_path)

    renderer.add_instance(0)
    print(renderer.visual_objects, renderer.instances)
    print(renderer.materials_mapping, renderer.mesh_materials)

    camera_pose = (
        np.array(
            json.load(open("/data2/gibson_tiny/Allensville/pano/points/point_p000001.json"))[-1]["camera_rt_matrix"]
        )
        .astype(np.float32)
        .reshape((4, 4))
    )
    print(camera_pose)

    camera_pose = np.linalg.inv(camera_pose)
    renderer.V = camera_pose

    img = renderer.get_equi()
    img2 = np.array(
        Image.open("/data2/gibson_tiny/Allensville/pano/rgb/point_p000001_view_equirectangular_domain_rgb.png")
    )
    print(img.shape)
    Image.fromarray((img * 255).astype(np.uint8)).save("equi.png")
    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.subplot(3, 1, 2)
    plt.imshow(img2)
    plt.subplot(3, 1, 3)
    plt.imshow(np.abs(img2[:, :, :3] - img[:, :, :3]))
    plt.show()

    renderer.release()


if __name__ == "__main__":
    main()
