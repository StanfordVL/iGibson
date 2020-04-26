from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject
from gibson2.utils.utils import parse_config
import pybullet as p
import numpy as np
from gibson2.core.render.profiler import Profiler
import cv2


def main():
    config = parse_config('../configs/turtlebot_demo.yaml')
    s = Simulator(mode='gui', image_width=512, image_height=512)
    scene = BuildingScene('17DRP5sb8fy',
                          build_graph=True,
                          pybullet_load_texture=True)
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    cracker_box_sem_class = 256 * 15  # (r, g, b) = (0, 0, 16 * 15 = 240)
    for _ in range(10):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj, class_id=cracker_box_sem_class)
        obj.set_position_orientation(np.random.uniform(
            low=0, high=2, size=3), [0, 0, 0, 1])

    cv2.namedWindow('SemSeg')
    for i in range(1000000000):
        with Profiler('Simulator step'):
            turtlebot.apply_action([0.1, 0.1])
            s.step()
            frame = s.renderer.render_robot_cameras(modes=('rgb', 'normal', '3d', 'seg'))
            # the semantic class of cracker box should appear to be blue
            cv2.imshow('SemSeg', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))

    s.disconnect()


if __name__ == '__main__':
    main()
