import os
import numpy as np
import igibson
from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets
from IPython import embed
import pybullet as p
if __name__ == "__main__":
    simulator = Simulator(mode="pbgui")
    scene = EmptyScene()
    simulator.import_scene(scene)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # embed()
    shelf_dir = os.path.join(igibson.ig_dataset_path, "objects/shelf/1170df5b9512c1d92f6bce2b7e6c12b7/")
    shelf_filename = os.path.join(shelf_dir, "1170df5b9512c1d92f6bce2b7e6c12b7.urdf")
    chocolate_box_dir = os.path.join(igibson.ig_dataset_path, "objects/chocolate_box/chocolate_box_000/")
    chocolate_box_filename = os.path.join(chocolate_box_dir, "chocolate_box_000.urdf")
    shelf = URDFObject(
        filename=shelf_filename, category="shelf", model_path=shelf_dir, bounding_box=np.array([1.0, 0.4, 2.0])
    )
    simulator.import_object(shelf)
    # shelf.set_position([2, 2, 1])
    shelf.set_position([0, 0, 1])
    # shelf.set_position([0, 0, 0.6])
    # shelf.set_orientation([0.576, 0, 0, 0.81])
    # p.createConstraint(shelf.get_body_id(), -1, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
    for _ in range(100):
        simulator.step()
    print("Shelf placed")
    # num = 1
    num = 10
    for i in range(num):
        chocolate_box = URDFObject(
            filename=chocolate_box_filename,
            category="chocolate_box",
            model_path=chocolate_box_dir,
            bounding_box=np.array([0.2, 0.05, 0.3]),
        )
        simulator.import_object(chocolate_box)
        chocolate_box.states[object_states.Inside].set_value(shelf, True, use_ray_casting_method=True)
        print("Box %d placed." % i)
        for _ in range(100):
            simulator.step()