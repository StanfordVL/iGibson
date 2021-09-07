import os

import numpy as np

import igibson
from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets

download_assets()


def main():
    s = Simulator(mode="gui")
    scene = EmptyScene()
    s.import_scene(scene)

    try:
        sample_microwave_plates_apples(s)
        sample_boxes_on_shelf(s)

        while True:
            s.step()
    finally:
        s.disconnect()


def sample_microwave_plates_apples(simulator):
    cabinet_filename = os.path.join(igibson.assets_path, "models/cabinet2/cabinet_0007.urdf")

    microwave_dir = os.path.join(igibson.ig_dataset_path, "objects/microwave/7128/")
    microwave_filename = os.path.join(microwave_dir, "7128.urdf")

    plate_dir = os.path.join(igibson.ig_dataset_path, "objects/plate/plate_000/")
    plate_filename = os.path.join(plate_dir, "plate_000.urdf")

    apple_dir = os.path.join(igibson.ig_dataset_path, "objects/apple/00_0/")
    apple_filename = os.path.join(apple_dir, "00_0.urdf")

    cabinet = URDFObject(filename=cabinet_filename, category="cabinet", scale=np.array([2.0, 2.0, 2.0]))
    simulator.import_object(cabinet)
    cabinet.set_position([0, 0, 0.5])
    for _ in range(100):
        simulator.step()

    microwave = URDFObject(
        filename=microwave_filename, category="microwave", model_path=microwave_dir, scale=np.array([0.5, 0.5, 0.5])
    )
    simulator.import_object(microwave)
    assert microwave.states[object_states.OnTop].set_value(cabinet, True, use_ray_casting_method=True)
    microwave.states[object_states.Open].set_value(True)
    for _ in range(100):
        simulator.step()
    print("Microwave placed")

    for i in range(3):
        plate = URDFObject(
            filename=plate_filename, category="plate", model_path=plate_dir, bounding_box=[0.25, 0.25, 0.05]
        )
        simulator.import_object(plate)

        # Put the 1st plate in the microwave
        if i == 0:
            assert plate.states[object_states.Inside].set_value(microwave, True, use_ray_casting_method=True)
        else:
            assert plate.states[object_states.OnTop].set_value(microwave, True, use_ray_casting_method=True)

        print("Plate %d placed." % i)

        for _ in range(100):
            simulator.step()

        for j in range(3):
            apple = URDFObject(filename=apple_filename, category="apple", model_path=apple_dir)
            simulator.import_object(apple)
            assert apple.states[object_states.OnTop].set_value(plate, True, use_ray_casting_method=True)

            for _ in range(100):
                simulator.step()


def sample_boxes_on_shelf(simulator):
    shelf_dir = os.path.join(igibson.ig_dataset_path, "objects/shelf/1170df5b9512c1d92f6bce2b7e6c12b7/")
    shelf_filename = os.path.join(shelf_dir, "1170df5b9512c1d92f6bce2b7e6c12b7.urdf")

    cracker_box_dir = os.path.join(igibson.ig_dataset_path, "objects/cracker_box/cracker_box_000/")
    cracker_box_filename = os.path.join(cracker_box_dir, "cracker_box_000.urdf")

    shelf = URDFObject(
        filename=shelf_filename, category="shelf", model_path=shelf_dir, bounding_box=np.array([1.0, 0.4, 2.0])
    )
    simulator.import_object(shelf)
    shelf.set_position([2, 2, 1])
    for _ in range(100):
        simulator.step()
    print("Shelf placed")

    for i in range(10):
        cracker_box = URDFObject(
            filename=cracker_box_filename,
            category="cracker_box",
            model_path=cracker_box_dir,
            bounding_box=np.array([0.2, 0.05, 0.3]),
        )
        simulator.import_object(cracker_box)
        cracker_box.states[object_states.Inside].set_value(shelf, True, use_ray_casting_method=True)

        print("Box %d placed." % i)

        for _ in range(100):
            simulator.step()


if __name__ == "__main__":
    main()
