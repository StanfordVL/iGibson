import cv2
import sys
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from gibson2.core.render.profiler import Profiler
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    model_path = sys.argv[1]
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
            frame = renderer.render_to_tensor(modes=('rgb', 'normal'))

    print(frame)
    img_np = frame[0].flip(0).data.cpu().numpy().reshape(renderer.height, renderer.width, 4)
    normal_np = frame[1].flip(0).data.cpu().numpy().reshape(renderer.height, renderer.width, 4)
    plt.imshow(np.concatenate([img_np, normal_np], axis=1))
    plt.show()

    renderer.release()