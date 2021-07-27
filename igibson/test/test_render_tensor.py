from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

import numpy as np
import os
import igibson
import GPUtil
import time
from igibson.utils.assets_utils import download_assets
from igibson.utils.assets_utils import get_ig_model_path
from PIL import Image
import matplotlib.pyplot as plt
import torch

def test_tensor_render_rendering():
    w = 800
    h = 600
    setting = MeshRendererSettings(enable_pbr=False, msaa=True)
    renderer = MeshRendererG2G(w, h, rendering_settings=setting)
    test_dir = os.path.join(igibson.assets_path, 'test')
    renderer.load_object(os.path.join(test_dir, 'mesh/bed1a77d92d64f5cbbaaae4feed64ec1_new.obj'))
    renderer.add_instance(0)

    renderer.set_camera([0, 0, 1.2], [0, 1, 1.2], [0, 1, 0])
    renderer.set_fov(90)
    tensor, tensor2 = renderer.render(modes=('rgb', 'normal'))

    img_np = tensor.flip(0).data.cpu().numpy().reshape(h, w, 4)
    img_np2 = tensor2.flip(0).data.cpu().numpy().reshape(h, w, 4)

    # plt.subplot(1,2,1)
    # plt.imshow(img_np)
    # plt.subplot(1,2,2)
    # plt.imshow(img_np2)
    # plt.show()
    assert (np.allclose(np.mean(img_np.astype(np.float32), axis=(0, 1)),
                       np.array([131.71548, 128.34981, 121.81708, 255.86292]), rtol=1e-3))

    # print(np.mean(img_np.astype(np.float32), axis = (0,1)))
    # print(np.mean(img_np2.astype(np.float32), axis = (0,1)))
    renderer.release()

