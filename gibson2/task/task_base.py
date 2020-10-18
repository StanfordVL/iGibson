import tasknet as tn 
from tasknet.task_base import TaskNetTask
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings 


scene_path = 'd:\\gibson2_assets\\gibson_v2_selected\\'         # TODO where to source this from? 
scene_path = 'd:\\ig_dataset\\scenes'


class iGTNTaskInstance(TaskNetTask):
    def __init__(self, atus_activity, simulator):
        super().__init__(atus_activity)
        self.simulator = simulator 

    def initialize_scene(self):             # NOTE can't have the same method name right 
        '''
        Get scene populated with objects such that scene satisfies initial conditions 
        '''
        self.scene = self.initialize(scene_path, InteractiveIndoorScene)
        self.simulator.import_scene(self.scene)


def main():
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    sim = Simulator(mode='gui', image_width=1000, image_height=1000, device_idx=0, rendering_settings=settings)
    igtn_task_instance = iGTNTaskInstance('demo1', sim)
    igtn_task_instance.initialize_scene()
    for i in range(30):
        sim.step()
    print('TASK SUCCESS:', igtn_task_instance.check_success())
    sim.disconnect()

    
if __name__ == '__main__':
    main()

    

