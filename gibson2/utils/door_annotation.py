import yaml
from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.utils.utils import parse_config
from gibson2.core.physics.interactive_objects import InteractiveObj
import gibson2

if __name__ == '__main__':
    s = Simulator(mode='gui')
    scene = BuildingScene('Ohoopee')
    s.import_scene(scene)

    door = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components',
                                       'realdoor.urdf'),
                          scale=0.3)
    s.import_interactive_object(door)

    x = p.addUserDebugParameter('x', -10, 10, 0)
    y = p.addUserDebugParameter('y', -10, 10, 0)
    z = p.addUserDebugParameter('z', -5, 5, 0)
    rotate = p.addUserDebugParameter('rotate', -np.pi, np.pi, 0)

    while True:
        x_pos = p.readUserDebugParameter(x)
        y_pos = p.readUserDebugParameter(y)
        z_pos = p.readUserDebugParameter(z)
        rotate_pos = p.readUserDebugParameter(rotate)
        door.set_position_rotation([x_pos, y_pos, z_pos],
                                   p.getQuaternionFromEuler([0, 0, rotate_pos]))
        s.step()
