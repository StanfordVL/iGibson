from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main():
    s = Simulator(mode='gui', image_width=512,
                  image_height=512, device_idx=0)

    for random_seed in range(10):
        scene = InteractiveIndoorScene('Rs_int',
                                       texture_randomization=False,
                                       object_randomization=True,
                                       object_randomization_idx=random_seed)
        s.import_ig_scene(scene)
        for i in range(1000):
            s.step()
        s.reload()

    s.disconnect()


if __name__ == '__main__':
    main()
