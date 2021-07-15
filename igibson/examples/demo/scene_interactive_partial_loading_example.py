from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main():
    s = Simulator(mode='gui', image_width=512,
                  image_height=512, device_idx=0)
    scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=False, object_randomization=False,
        load_object_categories=['chair'], load_room_types=['living_room'])
    s.import_ig_scene(scene)

    for _ in range(1000):
        s.step()
    s.disconnect()


if __name__ == '__main__':
    main()
