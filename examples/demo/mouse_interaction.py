#!/usr/bin/env python
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
import os
import gibson2
import time
import random
import sys
import matplotlib.pyplot as plt
import pybullet as p
# human interaction demo


def test_import_igsdf():
    hdr_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
    hdr_texture2 = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
    light_modulation_map_filename = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
    background_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

    scene = InteractiveIndoorScene(
        'Beechwood_0_int', texture_randomization=False, object_randomization=False)
    #scene._set_first_n_objects(10)
    settings = MeshRendererSettings(env_texture_filename=hdr_texture,
                                    env_texture_filename2=hdr_texture2,
                                    env_texture_filename3=background_texture,
                                    light_modulation_map_filename=light_modulation_map_filename,
                                    enable_shadow=True, msaa=True,
                                    light_dimming_factor=1.0)
    s = Simulator(mode='iggui', image_width=960,
                  image_height=720, device_idx=0, rendering_settings=settings)

    #s.viewer.min_cam_z = 1.0

    s.import_ig_scene(scene)
    fpss = []
    
    for i in range(3000):
        if i == 2500:
            logId = p.startStateLogging(loggingType=p.STATE_LOGGING_PROFILE_TIMINGS, fileName='trace_beechwood')
        start = time.time()
        s.step()
        end = time.time()
        print("Elapsed time: ", end - start)
        print("Frequency: ", 1 / (end - start))
        fpss.append(1 / (end - start))
    p.stopStateLogging(logId)
    s.disconnect()
    print("end")
    
    plt.plot(fpss)
    plt.show()

def main():
    test_import_igsdf()


if __name__ == "__main__":
    main()
