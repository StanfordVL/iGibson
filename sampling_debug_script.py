import os

import numpy as np

import igibson
from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
import pybullet as p

def main():
    simulator = Simulator(mode="pbgui")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    scene = EmptyScene()
    simulator.import_scene(scene)
    shelf_dir = os.path.join(igibson.ig_dataset_path, "objects/shelf/1170df5b9512c1d92f6bce2b7e6c12b7/")
    shelf_filename = os.path.join(shelf_dir, "1170df5b9512c1d92f6bce2b7e6c12b7.urdf")

    cracker_box_dir = os.path.join(igibson.ig_dataset_path, "objects/cracker_box/cracker_box_000/")
    cracker_box_filename = os.path.join(cracker_box_dir, "cracker_box_000.urdf")

    shelf = URDFObject(
        filename=shelf_filename, category="shelf", model_path=shelf_dir, bounding_box=np.array([1.0, 0.4, 2.0])
    )
    simulator.import_object(shelf)
    shelf.set_position([2, 2, 0.6])
    # shelf.set_orientation([0,0,0,1])
    p.changeDynamics(shelf.get_body_id(), linkIndex=-1,  mass=0)
    shelf.set_orientation((0.1, 0.2, 0.3, 0.8176056393931193))
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
        # cracker_box.states[object_states.Inside].set_value(shelf, True, use_ray_casting_method=True)
        cracker_box.states[object_states.OnTop].set_value(shelf, True, use_ray_casting_method=True)

        print("Box %d placed." % i)

        for _ in range(100):
            simulator.step()

if __name__ == "__main__":
    main()
