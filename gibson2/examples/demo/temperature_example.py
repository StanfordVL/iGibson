import os

import gibson2
import numpy as np
from gibson2 import object_states
from gibson2.object_states.factory import prepare_object_states
from gibson2.objects.articulated_object import URDFObject
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.simulator import Simulator
from gibson2.utils.assets_utils import download_assets

download_assets()


def main():
    s = Simulator(mode='gui', image_height=512, image_width=512)

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        stove_dir = os.path.join(
            gibson2.ig_dataset_path, 'objects/stove/101930/')
        stove_urdf = os.path.join(stove_dir, "101930.urdf")
        stove = URDFObject(stove_urdf, name="stove", category="stove", model_path=stove_dir)
        s.import_object(stove)
        stove.set_position([0, 0, 0.782])
        stove.states[object_states.ToggledOn].set_value(True)

        microwave_dir = os.path.join(
            gibson2.ig_dataset_path, 'objects/microwave/7128/')
        microwave_urdf = os.path.join(microwave_dir, "7128.urdf")
        microwave = URDFObject(microwave_urdf, name="microwave", category="microwave", model_path=microwave_dir)
        s.import_object(microwave)
        microwave.set_position([2, 0, 0.401])
        microwave.states[object_states.ToggledOn].set_value(True)

        oven_dir = os.path.join(
            gibson2.ig_dataset_path, 'objects/oven/7120/')
        oven_urdf = os.path.join(oven_dir, "7120.urdf")
        oven = URDFObject(oven_urdf, name="oven", category="oven", model_path=oven_dir)
        s.import_object(oven)
        oven.set_position([-2, 0, 0.816])
        oven.states[object_states.ToggledOn].set_value(True)

        tray_dir = os.path.join(
            gibson2.ig_dataset_path, 'objects/tray/tray_000/')
        tray_urdf = os.path.join(tray_dir, 'tray_000.urdf')
        tray = URDFObject(tray_urdf, name="tray", category="tray", model_path=tray_dir, scale=np.array([0.1,0.1,0.1]))
        s.import_object(tray)
        tray.set_position([0, 0, 1.5])

        apple_dir = os.path.join(
            gibson2.ig_dataset_path, 'objects/apple/00_0/')
        apple_urdf = os.path.join(apple_dir, "00_0.urdf")
        apple = URDFObject(apple_urdf, name="apple", category="apple", model_path=apple_dir, texture_procedural_generation=True)
        s.import_object(apple)
        apple.set_position([0, 0, 1.6])


        # Run simulation for 1000 steps
        while True:
            s.step()
            print("Apple Temperature: ", apple.states[object_states.Temperature].get_value())
            print("Apple Cooked: ", apple.states[object_states.Cooked].get_value())
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
