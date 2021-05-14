from gibson2 import object_states
from gibson2.object_states.factory import prepare_object_states
from gibson2.objects.ycb_object import YCBObject
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator


def main():
    s = Simulator(mode='gui', device_idx=0)
    scene = InteractiveIndoorScene(
        'Rs_int',
        texture_randomization=False,
        object_randomization=False,
        merge_fixed_links=False
    )
    s.import_ig_scene(scene)

    # Set everything that can go dirty.
    stateful_objects = set(
        scene.get_objects_with_state(object_states.Dusty) + scene.get_objects_with_state(object_states.Stained) +
        scene.get_objects_with_state(object_states.WaterSource) + scene.get_objects_with_state(object_states.Open))
    for obj in stateful_objects:
        if object_states.Dusty in obj.states:
            obj.states[object_states.Dusty].set_value(True)

        if object_states.Stained in obj.states:
            obj.states[object_states.Stained].set_value(True)

        if object_states.Open in obj.states:
            obj.states[object_states.Open].set_value(True)

        if object_states.WaterSource in obj.states:
            obj.states[object_states.ToggledOn].set_value(True)

    # Take some steps so water drops appear
    for i in range(100):
        s.step()

    scene.save_modified_urdf("cleaning_demo")

    # Let the user view the frozen scene in the UI for purposes of comparison.
    try:
        while True:
            pass
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
