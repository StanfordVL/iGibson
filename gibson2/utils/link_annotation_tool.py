import os

import gibson2
from pynput import keyboard
import numpy as np

from gibson2.objects.articulated_object import URDFObject
from gibson2.objects.visual_marker import VisualMarker
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.simulator import Simulator
from gibson2.utils.assets_utils import download_assets

download_assets()

# Categories that you want to annotate.
CATEGORIES = ["stove", "oven", "microwave", "sink"]


def get_category_directory(category):
    return os.path.join(gibson2.ig_dataset_path, 'objects', category)


def get_obj(folder):
    return URDFObject(os.path.join(folder, os.path.basename(folder) + ".urdf"), name="obj", model_path=folder)


def main():
    for cat in CATEGORIES:
        cd = get_category_directory(cat)
        for objdir in os.listdir(cd):
            objdirfull = os.path.join(cd, objdir)

            s = Simulator(mode='gui')
            scene = EmptyScene()
            s.import_scene(scene)
            obj = get_obj(objdirfull)
            s.import_object(obj)
            obj_pos = [0, 0, 1]
            obj.set_position(obj_pos)

            m = VisualMarker(radius=0.02)
            s.import_object(m)

            m_pos = np.array([0., 0., 2.])
            m.set_position(m_pos)

            step_size = 0.01
            while True:
                with keyboard.Events() as events:
                    # Block at most one second
                    event = events.get(1.0)
                    if event is None or not isinstance(event, keyboard.Events.Press) or not hasattr(event.key, "char"):
                        continue

                    if event.key.char == "w":
                        print("Moving forward one")
                        m_pos += np.array([0, 1, 0]) * step_size
                    elif event.key.char == "a":
                        print("Moving left one")
                        m_pos += np.array([-1, 0, 0]) * step_size
                    elif event.key.char == "s":
                        print("Moving back one")
                        m_pos += np.array([0, -1, 0]) * step_size
                    elif event.key.char == "d":
                        print("Moving right one")
                        m_pos += np.array([1, 0, 0]) * step_size
                    elif event.key.char == "q":
                        print("Moving up one")
                        m_pos += np.array([0, 0, 1]) * step_size
                    elif event.key.char == "z":
                        print("Moving down one")
                        m_pos += np.array([0, 0, -1]) * step_size
                    elif event.key.char == "h":
                        print("Sizing to 0.1")
                        step_size = 0.1
                    elif event.key.char == "j":
                        print("Sizing to 0.01")
                        step_size = 0.01
                    elif event.key.char == "k":
                        print("Sizing to 0.001")
                        step_size = 0.001
                    elif event.key.char == "c":
                        break

                print("New position:", m_pos)
                m.set_position(m_pos)

            px, py, pz = tuple(np.array(m.get_position()) - np.array(obj_pos))
            print("('%s', '%s'): [%.3f, %.3f, %.3f]," % (cat, objdir, px, py, pz))

            input("Hit enter to continue.")

            s.disconnect()

if __name__ == "__main__":
    main()
