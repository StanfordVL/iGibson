from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main():
    s = Simulator(mode='gui', image_width=512,
                  image_height=512, device_idx=0)
    scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=True, object_randomization=False)
    s.import_ig_scene(scene)

    for i in range(10000):
        if i % 1000 == 0:
            scene.randomize_texture()
        s.step()
    s.disconnect()


if __name__ == '__main__':
    main()
