import os 

import tasknet as tn 
from tasknet.task_base import TaskNetTask
import gibson2
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings 
from gibson2.objects.articulated_object import URDFObject


class iGTNTaskInstance(TaskNetTask):
    def __init__(self, atus_activity):
        super().__init__(atus_activity)

    def initialize_scene(self):             # NOTE can't have the same method name right 
        '''
        Get scene populated with objects such that scene satisfies initial conditions 
        '''
        self.scene_name, self.scene = self.initialize(InteractiveIndoorScene, URDFObject)

        hdr_texture = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
        hdr_texture2 = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
        light_modulation_map_filename = os.path.join(
            gibson2.ig_dataset_path, 'scenes', self.scene_name, 'layout', 'floor_lighttype_0.png')
        background_texture = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

        settings = MeshRendererSettings(env_texture_filename=hdr_texture,
                                        env_texture_filename2=hdr_texture2,
                                        env_texture_filename3=background_texture,
                                        light_modulation_map_filename=light_modulation_map_filename,
                                        enable_shadow=True, msaa=True,
                                        light_dimming_factor=1.0)
        self.simulator = Simulator(mode='iggui', image_width=960, image_height=720, device_idx=0, rendering_settings=settings)
    
        self.simulator.viewer.min_cam_z = 1.0
        self.simulator.import_ig_scene(self.scene)


def main():
    igtn_task_instance = iGTNTaskInstance('demo1')
    igtn_task_instance.initialize_scene()

    for i in range(100):
        igtn_task_instance.simulator.step()
    print('TASK SUCCESS:', igtn_task_instance.check_success())
    igtn_task_instance.simulator.disconnect()
                
    
if __name__ == '__main__':
    main()

    

