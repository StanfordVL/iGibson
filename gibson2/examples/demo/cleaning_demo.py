from gibson2 import object_states
from gibson2.object_states.factory import prepare_object_states
from gibson2.objects.ycb_object import YCBObject
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator


def main():
    s = Simulator(mode='gui', image_width=512,
                  image_height=512, device_idx=0)
    scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=False, object_randomization=False)
    s.import_ig_scene(scene)

    block = YCBObject(name='036_wood_block')
    block.abilities = ["soakable", "cleaning_tool"]
    prepare_object_states(block, abilities={"soakable": {}, "cleaning_tool": {}})
    s.import_object(block)
    block.set_position([1, 1, 1.8])

    # Set everything that can go dirty.
    stateful_objects = set(
        scene.get_objects_with_state(object_states.Dusty) + scene.get_objects_with_state(object_states.Stained) +
        scene.get_objects_with_state(object_states.WaterSource))
    for obj in stateful_objects:
        if object_states.Dusty in obj.states:
            obj.states[object_states.Dusty].set_value(True)

        if object_states.Stained in obj.states:
            obj.states[object_states.Stained].set_value(True)

        if object_states.WaterSource in obj.states and object_states.ToggledOn in obj.states:
            obj.states[object_states.ToggledOn].set_value(True)

    try:
        while True:
            s.step()
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
