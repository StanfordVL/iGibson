import os

import gibson2
import numpy as np
from gibson2.object_states import Temperature

from gibson2.object_states.factory import prepare_object_states
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

        stove_dir = os.path.join(
            gibson2.ig_dataset_path, 'objects/stove/101930/')
        stove_urdf = os.path.join(stove_dir, "101930.urdf")
        saucepan_dir = os.path.join(
            gibson2.ig_dataset_path, 'objects/saucepan/38_2/')
        saucepan_urdf = os.path.join(saucepan_dir, "38_2.urdf")

        stove = URDFObject(stove_urdf, name="stove", model_path=stove_dir)
        prepare_object_states(stove, {"heatSource": {}})
        s.import_object(stove)
        stove.set_position([0, 0, 0.76])

        saucepan = URDFObject(saucepan_urdf, name="saucepan", model_path=saucepan_dir, scale=np.array([0.5, 0.5, 0.5]))
        prepare_object_states(saucepan, {"cookable": {}})
        s.import_object(saucepan)
        saucepan.set_position([-0.2, -0.2, 1.7])

        # Run simulation for 1000 steps
        while True:
            s.step()
            print("Saucepan Temperature: ", saucepan.states[Temperature].get_value())
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
