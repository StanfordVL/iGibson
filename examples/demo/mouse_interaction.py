#!/usr/bin/env python
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_ig_scene_non_colliding_seeds
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
import os
import gibson2
import time
import random
import sys

# human interaction demo


def test_import_igsdf():
    seeds = get_ig_scene_non_colliding_seeds('Rs_int')
    random.seed(seeds[0])
    config = parse_config(os.path.join(gibson2.root_path, '../test/test.yaml'))
    hdr_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'lighting', 'probes', 'probe_02.hdr')
    hdr_texture2 = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'lighting', 'probes', 'probe_03.hdr')
    light_modulation_map_filename = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'light_fusion_map.png')
    background_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

    scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=False, object_randomization=False)
    # scene._set_first_n_objects(10)
    settings = MeshRendererSettings(env_texture_filename=hdr_texture,
                                    env_texture_filename2=hdr_texture2,
                                    env_texture_filename3=background_texture,
                                    light_modulation_map_filename=light_modulation_map_filename,
                                    enable_shadow=True, msaa=True,
                                    light_dimming_factor=1.0)
    s = Simulator(mode='iggui', image_width=960,
                  image_height=720, device_idx=0, rendering_settings=settings)

    s.import_ig_scene(scene)

    while True:
        start = time.time()
        s.step()
        end = time.time()
        print("Elapsed time: ", end - start)
        print("Frequency: ", 1 / (end - start))
    s.disconnect()
    print("end")


def main():
    test_import_igsdf()


if __name__ == "__main__":
    main()
