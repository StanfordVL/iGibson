import os

import numpy as np
import pdb

import gibson2
from gibson2 import object_states
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.articulated_object import URDFObject
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.simulator import Simulator
from gibson2.utils.assets_utils import download_assets

download_assets()


def main():
    s = Simulator(mode='gui')

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        cabinet_filename = os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')

        microwave_dir = os.path.join(gibson2.ig_dataset_path, 'objects/microwave/7128/')
        microwave_filename = os.path.join(microwave_dir, '7128.urdf')

        plate_dir = os.path.join(gibson2.ig_dataset_path, 'objects/plate/plate_000/')
        plate_filename = os.path.join(plate_dir, 'plate_000.urdf')

        apple_dir = os.path.join(gibson2.ig_dataset_path, 'objects/apple/00_0/')
        apple_filename = os.path.join(apple_dir, '00_0.urdf')

        cabinet = ArticulatedObject(filename=cabinet_filename, scale=2)
        s.import_object(cabinet)
        cabinet.set_position([0, 0, 0.5])

        for _ in range(100):
            s.step()

        microwave = URDFObject(filename=microwave_filename, category="microwave", model_path=microwave_dir,
                               scale=np.array([0.5, 0.5, 0.5]))
        s.import_object(microwave)
        assert microwave.states[object_states.OnTop].set_value(cabinet, True, use_ray_casting_method=True)

        microwave.states[object_states.Open].set_value(True)

        for _ in range(100):
            s.step()

        print("Microwave placed")

        for i in range(3):
            plate = URDFObject(filename=plate_filename, category="plate", model_path=plate_dir,
                               bounding_box=[0.25, 0.25, 0.05])
            s.import_object(plate)

            # Put the 1st plate in the microwave
            if i == 0:
                assert plate.states[object_states.Inside].set_value(microwave, True, use_ray_casting_method=True)
            else:
                assert plate.states[object_states.OnTop].set_value(microwave, True, use_ray_casting_method=True)

            print("Plate %d placed." % i)

            for _ in range(100):
                s.step()

            for j in range(3):
                apple = URDFObject(filename=apple_filename, category="apple", model_path=apple_dir)
                s.import_object(apple)
                assert apple.states[object_states.OnTop].set_value(plate, True, use_ray_casting_method=True)

                for _ in range(100):
                    s.step()

        while True:
            s.step()
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()