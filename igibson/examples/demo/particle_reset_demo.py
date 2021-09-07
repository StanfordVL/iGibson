import os

import numpy as np
import pybullet as p

from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_model_path


def main():
    s = Simulator(mode="gui", image_width=1280, image_height=720)

    scene = EmptyScene()
    s.import_scene(scene)

    model_path = os.path.join(get_ig_model_path("sink", "sink_1"), "sink_1.urdf")

    sink = URDFObject(
        filename=model_path,
        category="sink",
        name="sink_1",
        scale=np.array([0.8, 0.8, 0.8]),
        abilities={"toggleable": {}, "waterSource": {}},
    )

    s.import_object(sink)
    sink.set_position([1, 1, 0.8])
    sink.states[object_states.ToggledOn].set_value(True)

    model_path = get_ig_model_path("vacuum", "vacuum_000")
    model_filename = os.path.join(model_path, "vacuum_000.urdf")
    max_bbox = [0.1, 0.1, 0.1]
    avg = {"size": max_bbox, "density": 67.0}
    brush = URDFObject(
        filename=model_filename, category="vacuum", name="vacuum", avg_obj_dims=avg, fit_avg_dim_volume=True
    )
    s.import_object(brush)
    brush.set_position([0, -2, 0.4])

    model_path = os.path.join(get_ig_model_path("breakfast_table", "19203"), "19203.urdf")
    desk = URDFObject(
        filename=model_path,
        category="table",
        name="19898",
        scale=np.array([0.8, 0.8, 0.8]),
        abilities={"dustyable": {}},
    )

    print(desk.states.keys())
    s.import_object(desk)
    desk.set_position([1, -2, 0.4])
    desk.states[object_states.Dusty].set_value(True)

    # Dump the initial state.
    state_dump = p.saveState()
    dump = desk.dump_state()
    print(dump)

    # Main simulation loop.
    try:
        # Keep stepping until we reach a clean state.
        while desk.states[object_states.Dusty].get_value():
            s.step()
            if not desk.states[object_states.Dusty].get_value():
                print("Cleaned.")
            print("Dirty: ", desk.states[object_states.Dusty].get_value())

        # Reset to the initial state
        p.restoreState(state_dump)
        p.removeState(state_dump)
        desk.load_state(dump)
        desk.force_wakeup()

        # Continue simulation
        while True:
            s.step()
            print("Dirty: ", desk.states[object_states.Dusty].get_value())
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
