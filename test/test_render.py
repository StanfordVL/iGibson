from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
import numpy as np
import torch
import os
from gibson2 import assets
import matplotlib.pyplot as plt

dir = os.path.join(os.path.dirname(assets.__file__), 'test')

def test_render_loading_cleaning():
    renderer = MeshRenderer(width=800, height=600)
    renderer.release()


def test_render_rendering():
    renderer = MeshRenderer(width=800, height=600)
    renderer.load_object(os.path.join(dir, 'mesh/bed1a77d92d64f5cbbaaae4feed64ec1_new.obj'))
    renderer.add_instance(0)
    renderer.set_camera([0, 0, 1.2], [0, 1, 1.2], [0, 1, 0])
    renderer.set_fov(90)
    rgb, _, seg, _ = renderer.render()
    #plt.imshow(np.concatenate([rgb, seg], axis=1)) # uncomment these two lines to show the rendering results
    #plt.show()
    assert (np.allclose(np.mean(rgb, axis=(0, 1)), np.array([0.51661223, 0.5035339, 0.4777793, 1.]), rtol=1e-3))
    renderer.release()
