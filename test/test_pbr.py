from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, MeshRendererSettings
import numpy as np
import os
import gibson2
import GPUtil
import time
from gibson2.utils.assets_utils import download_assets
from gibson2.utils.assets_utils import get_ig_model_path
from PIL import Image


def test_render_pbr():
    hdr_texture = os.path.join(gibson2.ig_dataset_path, 'background', 'quattro_canti_2k.hdr')
    model_path = os.path.join(get_ig_model_path('sink', 'sink_1'), 'shape', 'visual')
    settings = MeshRendererSettings(msaa=True, enable_shadow=True, env_texture_filename=hdr_texture, env_texture_filename3=hdr_texture)
    renderer = MeshRenderer(width=1024, height=1024, vertical_fov=90, rendering_settings=settings)
    renderer.set_light_position_direction([0,0,10], [0,0,0])
    i = 0

    for fn in os.listdir(model_path):
        if fn.endswith('obj'):
            renderer.load_object(os.path.join(model_path, fn), scale=[1, 1, 1])
            renderer.add_instance(i)
            i += 1
            renderer.instances[-1].use_pbr = True
            renderer.instances[-1].use_pbr_mapping = True


    renderer.set_camera([1.5,1.5,1.5], [0,0,0], [0, 0, 1])
    frame = renderer.render(modes=('rgb', 'normal'))
    Image.fromarray((255*np.concatenate(frame, axis=1)[:,:,:3]).astype(np.uint8)).save('test_render.png')

