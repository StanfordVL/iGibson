from gibson2 import object_states
from gibson2.object_states.factory import prepare_object_states
from gibson2.objects.ycb_object import YCBObject
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator


def main():
    s = Simulator(mode='gui', device_idx=0)
    scene = InteractiveIndoorScene(
        'Rs_int', urdf_file="cleaning_demo",
    )
    s.import_ig_scene(scene)

    # Let the user view the frozen scene in the UI for purposes of comparison.
    try:
        while True:
            pass
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
