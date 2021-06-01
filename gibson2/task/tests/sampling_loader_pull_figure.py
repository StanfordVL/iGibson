from gibson2.task.task_base import iGTNTask
from IPython import embed
import numpy as np
from PIL import Image
from gibson2.simulator import Simulator
import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
import os

import tasknet
tasknet.set_backend("iGibson")

task_choices = [
    "packing_lunches_filtered",
    "assembling_gift_baskets_filtered",
    "organizing_school_stuff_filtered",
    "re-shelving_library_books_filtered",
    "serving_hors_d_oeuvres_filtered",
    "putting_away_toys_filtered",
    "putting_away_Christmas_decorations_filtered",
    "putting_dishes_away_after_cleaning_filtered",
    "cleaning_out_drawers_filtered",
]
task = 'pull_figure_chair_no_clutter'
task_id = 0
scene = 'Rs_int'
num_init = 1

camera_pos = np.array([1.2, -2.8, 1.5])
viewing_direction = np.array([-0.2, -0.8, -0.6])

hdr_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

# VR rendering settings
vr_rendering_settings = MeshRendererSettings(
    optimized=True,
    fullscreen=False,
    env_texture_filename=hdr_texture,
    env_texture_filename2=hdr_texture2,
    env_texture_filename3=background_texture,
    light_modulation_map_filename=light_modulation_map_filename,
    enable_shadow=True,
    enable_pbr=True,
    msaa=False,
    light_dimming_factor=1.0
)
igtn_task = iGTNTask(task, task_instance=task_id)
simulator = Simulator(mode='headless', image_width=1280,
                      image_height=720, rendering_settings=vr_rendering_settings)

for num_init in range(6):
    scene_kwargs = {
        'urdf_file': '{}_neurips_task_{}_{}_{}'.format(scene, task, task_id, num_init),
    }
    init_success = igtn_task.initialize_simulator(
        scene_id=scene,
        simulator=simulator,
        load_clutter=True,
        should_debug_sampling=False,
        scene_kwargs=scene_kwargs,
        online_sampling=False,
    )

    igtn_task.simulator.sync()
    igtn_task.simulator.renderer.set_camera(
        camera_pos, camera_pos + viewing_direction, [0, 0, 1])
    rgb = igtn_task.simulator.renderer.render(modes=('rgb'))[0][:, :, :3]
    Image.fromarray((rgb * 255).astype(np.uint8)
                    ).save('{}_{}.png'.format(task, num_init))
