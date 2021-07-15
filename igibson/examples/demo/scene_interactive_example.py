from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
import numpy as np


def main():
    s = Simulator(mode='gui', image_width=512,
                  image_height=512, device_idx=0)
    scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=False, object_randomization=False)
    s.import_ig_scene(scene)

    np.random.seed(0)
    for _ in range(10):
        pt = scene.get_random_point_by_room_type('living_room')[1]
        print('random point in living_room', pt)

    for _ in range(1000):
        s.step()
    s.disconnect()


if __name__ == '__main__':
    main()
