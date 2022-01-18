import os

import numpy as np
import pybullet as p

from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_model_path


def main():
    s = Simulator(mode="gui", image_width=1280, image_height=720)
    scene = EmptyScene()
    s.import_scene(scene)

    model_path = os.path.join(get_ig_model_path("breakfast_table", "19203"), "19203.urdf")
    desk = URDFObject(
        filename=model_path, category="table", name="19898", scale=np.array([0.8, 0.8, 0.8]), abilities={}
    )
    s.import_object(desk)
    desk.set_position([0, 0, 0.4])

    model_path = os.path.join(get_ig_model_path("apple", "00_0"), "00_0.urdf")
    simulator_obj = URDFObject(
        model_path, name="00_0", category="apple", scale=np.array([1.0, 1.0, 1.0]), initial_pos=[100, 100, -100]
    )

    whole_object = simulator_obj
    object_parts = []

    for i, part in enumerate(simulator_obj.metadata["object_parts"]):
        category = part["category"]
        model = part["model"]
        # Scale the offset accordingly
        part_pos = part["pos"] * whole_object.scale
        part_orn = part["orn"]
        model_path = get_ig_model_path(category, model)
        filename = os.path.join(model_path, model + ".urdf")
        obj_name = whole_object.name + "_part_{}".format(i)
        simulator_obj_part = URDFObject(
            filename,
            name=obj_name,
            category=category,
            model_path=model_path,
            scale=whole_object.scale,
            initial_pos=[100 + i, 100, -100],
        )
        object_parts.append((simulator_obj_part, (part_pos, part_orn)))

    grouped_obj_parts = ObjectGrouper(object_parts)
    apple = ObjectMultiplexer(whole_object.name + "_multiplexer", [whole_object, grouped_obj_parts], 0)

    s.import_object(apple)
    apple.set_position([0, 0, 0.72])

    # Let the apple fall
    for _ in range(100):
        s.step()

    # Dump the initial state.
    state_dump = p.saveState()
    dump = apple.dump_state()
    print(dump)

    # Slice the apple and set the object parts away
    apple.states[object_states.Sliced].set_value(True)
    assert isinstance(apple.current_selection(), ObjectGrouper)
    for obj_part in apple.objects:
        obj_part.set_position([0, 0, 1])

    p.restoreState(state_dump)
    p.removeState(state_dump)
    # The apple should become whole again
    apple.load_state(dump)

    while True:
        s.step()


if __name__ == "__main__":
    main()
