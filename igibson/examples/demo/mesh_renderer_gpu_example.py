import sys
import os
import numpy as np
from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_scene_path
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path('Rs'), 'mesh_z_up.obj')

    renderer = MeshRendererG2G(width=512, height=512, device_idx=0)
    renderer.load_object(model_path)
    renderer.add_instance(0)

    print(renderer.visual_objects, renderer.instances)
    print(renderer.materials_mapping, renderer.mesh_materials)
    camera_pose = np.array([0, 0, 1.2])
    view_direction = np.array([1, 0, 0])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)
    for i in range(3000):
        with Profiler('Render'):
            frame = renderer.render(modes=('rgb', 'normal', '3d'))

    print(frame)
    img_np = frame[0].flip(0).data.cpu().numpy().reshape(
        renderer.height, renderer.width, 4)
    normal_np = frame[1].flip(0).data.cpu().numpy().reshape(
        renderer.height, renderer.width, 4)
    plt.imshow(np.concatenate([img_np, normal_np], axis=1))
    plt.show()


if __name__ == '__main__':
    main()
