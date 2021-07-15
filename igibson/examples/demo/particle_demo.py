import os

import numpy as np

from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_model_path


def main():
    s = Simulator(mode="iggui", image_width=1280, image_height=720)

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

    model_path = get_ig_model_path("scrub_brush", "scrub_brush_000")
    model_filename = os.path.join(model_path, "scrub_brush_000.urdf")
    max_bbox = [0.1, 0.1, 0.1]
    avg = {"size": max_bbox, "density": 67.0}
    brush = URDFObject(
        filename=model_filename,
        category="scrub_brush",
        name="scrub_brush",
        avg_obj_dims=avg,
        fit_avg_dim_volume=True,
        model_path=model_path,
    )
    s.import_object(brush)
    brush.set_position([0, -2, 0.4])

    model_path = os.path.join(get_ig_model_path("breakfast_table", "19203"), "19203.urdf")
    desk = URDFObject(
        filename=model_path,
        category="table",
        name="19898",
        scale=np.array([0.8, 0.8, 0.8]),
        abilities={"dustyable": {}, "stainable": {}},
    )

    print(desk.states.keys())
    s.import_object(desk)
    desk.set_position([1, -2, 0.4])
    assert desk.states[object_states.Dusty].set_value(True)
    assert desk.states[object_states.Stained].set_value(True)

    model_path = os.path.join(get_ig_model_path("bowl", "68_0"), "68_0.urdf")
    bowl = URDFObject(filename=model_path, category="bowl", name="bowl", abilities={"dustyable": {}, "stainable": {}})
    s.import_object(bowl)
    assert bowl.states[object_states.OnTop].set_value(desk, True, use_ray_casting_method=True)
    assert bowl.states[object_states.Dusty].set_value(True)
    assert bowl.states[object_states.Stained].set_value(True)
    # Main simulation loop
    try:
        while True:
            s.step()
            print("Dirty: ", desk.states[object_states.Dusty].get_value())
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
