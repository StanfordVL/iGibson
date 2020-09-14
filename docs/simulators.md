# Simulators

### Overview

**Simulator** maintains an instance of **Renderer** and **PhysicsEngine** and provides APIs to import **Scene**, **Object** and **Robot** into both of them and keep them synchronized at all time.

Some key functions are the following:
- `load`: initialize PyBullet physics engine and MeshRenderer
- `import_scene`: import the scene into PyBullet by calling `scene.load`, and then import it into MeshRenderer by calling `self.renderer.add_instance`. If the scene is interactive (`is_interactive=True`), all the objects in the scene will be imported as well.
- `import_{object, articulated_object, robot}`: import the object, articulated object and robot into the simulator in a similar manner
- `sync`: synchronize the poses of the dynamic objects (including the robots) between PyBullet and MeshRenderer. Specifically, it calls `update_position` for each object, in which it retrieve the object's pose in PyBullet, and then update its pose accordingly in MeshRenderer.

If `Simulator` uses `gui` mode, by default it will also maintain a `Viewer`, which essentially is a virtual camera in the scene that can render images. More info about the `Viewer` can be found here: [gibson2/render/viewer.py](https://github.com/StanfordVL/iGibson/blob/master/gibson2/render/viewer.py).

Most of the code can be found here: [gibson2/simulator.py](https://github.com/StanfordVL/iGibson/blob/master/gibson2/simulator.py).

### Examples
In this example, we import a `BuildingScene`, a `Turtlebot`, and ten `YCBObject` into the simulator. The code can be found here: [examples/demo/simulator_example.py](https://github.com/StanfordVL/iGibson/blob/master/examples/demo/simulator_example.py)

```python
from gibson2.robots.robot_locomotors import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.indoor_scene import IndoorScene
from gibson2.objects.object_base import YCBObject
from gibson2.utils.utils import parse_config
import pybullet as p
import numpy as np
from gibson2.render.profiler import Profiler


def main():
    config = parse_config('../configs/turtlebot_demo.yaml')
    s = Simulator(mode='gui', image_width=512, image_height=512)
    scene = BuildingScene('Rs',
                          build_graph=True,
                          pybullet_load_texture=True)
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    for _ in range(10):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)
        obj.set_position_orientation(np.random.uniform(low=0, high=2, size=3), [0,0,0,1])

    for i in range(10000):
        with Profiler('Simulator step'):
            turtlebot.apply_action([0.1,0.1])
            s.step()
            rgb = s.renderer.render_robot_cameras(modes=('rgb'))

    s.disconnect()


if __name__ == '__main__':
    main()
```
