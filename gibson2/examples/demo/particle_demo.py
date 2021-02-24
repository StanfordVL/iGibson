from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from gibson2.render.profiler import Profiler
from gibson2.objects.articulated_object import URDFObject
import gibson2
import os
from gibson2.utils.assets_utils import get_ig_model_path
from gibson2.object_states.factory import prepare_object_states
import pybullet as p

def main():
    config = parse_config(os.path.join(gibson2.example_config_path, 'turtlebot_demo.yaml'))
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, optimized=True)
    s = Simulator(mode='gui', image_width=512,
                  image_height=512, rendering_settings=settings)

    scene = InteractiveIndoorScene('Rs_int',
                              build_graph=True,
                              pybullet_load_texture=True)
    scene._set_first_n_objects(3)
    s.import_ig_scene(scene)

    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)
    model_path = os.path.join(get_ig_model_path('sink', 'sink_1'), 'sink_1.urdf')

    sink = URDFObject(filename=model_path,
                     category='sink',
                     name='sink_1',
                     scale=np.array([0.8,0.8,0.8]),
                     abilities={'toggleable': {}, 'water_source': {}}
                     )

    s.import_object(sink)
    sink.set_position([1,1,0.8])

    block = YCBObject(name='036_wood_block')
    s.import_object(block)
    block.set_position([1, 1, 1.8])
    block.abilities = ["soakable", "cleaning_tool"]
    prepare_object_states(block, abilities={"soakable": {}, "cleaning_tool": {}})
    # assume block can soak water

    model_path = os.path.join(get_ig_model_path('table', '19898'), '19898.urdf')
    desk = URDFObject(filename=model_path,
                     category='table',
                     name='19898',
                     scale=np.array([0.8, 0.8, 0.8]),
                     abilities={'dustable': {}}
                     )

    s.import_object(desk)
    desk.set_position([1, -2, 0.4])

    for _ in range(100):
        p.stepSimulation()

    desk.states['dirty'].set_value(True)

    for i in range(10000):
        with Profiler('Simulator step'):
            turtlebot.apply_action([0.1, 0.1])
            s.step()
            rgb = s.renderer.render_robot_cameras(modes=('rgb'))
    s.disconnect()


if __name__ == '__main__':
    main()
