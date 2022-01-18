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
    s = Simulator(mode="iggui", image_width=1280, image_height=720)

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        stove_dir = os.path.join(igibson.ig_dataset_path, "objects/stove/101930/")
        stove_urdf = os.path.join(stove_dir, "101930.urdf")
        stove = URDFObject(stove_urdf, name="stove", category="stove", model_path=stove_dir)
        s.import_object(stove)
        stove.set_position([0, 0, 0.782])
        stove.states[object_states.ToggledOn].set_value(True)

        microwave_dir = os.path.join(igibson.ig_dataset_path, "objects/microwave/7128/")
        microwave_urdf = os.path.join(microwave_dir, "7128.urdf")
        microwave = URDFObject(microwave_urdf, name="microwave", category="microwave", model_path=microwave_dir)
        s.import_object(microwave)
        microwave.set_position([2, 0, 0.401])
        microwave.states[object_states.ToggledOn].set_value(True)

        oven_dir = os.path.join(igibson.ig_dataset_path, "objects/oven/7120/")
        oven_urdf = os.path.join(oven_dir, "7120.urdf")
        oven = URDFObject(oven_urdf, name="oven", category="oven", model_path=oven_dir)
        s.import_object(oven)
        oven.set_position([-2, 0, 0.816])
        oven.states[object_states.ToggledOn].set_value(True)

        tray_dir = os.path.join(igibson.ig_dataset_path, "objects/tray/tray_000/")
        tray_urdf = os.path.join(tray_dir, "tray_000.urdf")
        tray = URDFObject(tray_urdf, name="tray", category="tray", model_path=tray_dir, scale=np.array([0.1, 0.1, 0.1]))
        s.import_object(tray)
        tray.set_position([0, 0, 1.55])

        fridge_dir = os.path.join(igibson.ig_dataset_path, "objects/fridge/12252/")
        fridge_urdf = os.path.join(fridge_dir, "12252.urdf")
        fridge = URDFObject(
            fridge_urdf,
            name="fridge",
            category="fridge",
            model_path=fridge_dir,
            abilities={
                "coldSource": {
                    "temperature": -100.0,
                    "requires_inside": True,
                }
            },
        )
        s.import_object(fridge)
        fridge.set_position_orientation([-1, -3, 0.75], [1, 0, 0, 0])

        apple_dir = os.path.join(igibson.ig_dataset_path, "objects/apple/00_0/")
        apple_urdf = os.path.join(apple_dir, "00_0.urdf")

        apples = []
        for i in range(5):
            apple = URDFObject(apple_urdf, name="apple", category="apple", model_path=apple_dir)
            s.import_object(apple)
            apple.set_position([0, i * 0.05, 1.65])
            apples.append(apple)

        s.step()
        for apple in apples:
            apple.states[object_states.Temperature].set_value(-50)

        # Run simulation for 1000 steps
        while True:
            s.step()
            print(
                "Apple Temperature: %.2f. Frozen: %r. Cooked: %r. Burnt: %r."
                % (
                    apple.states[object_states.Temperature].get_value(),
                    apple.states[object_states.Frozen].get_value(),
                    apple.states[object_states.Cooked].get_value(),
                    apple.states[object_states.Burnt].get_value(),
                )
            )
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
