import random


def object_boolean_state_randomizer(target_state):
    def boolean_state_randomizer(env):
        scene = env.simulator.scene
        objects = scene.get_objects_with_state(target_state)
        obj = random.choice(objects)
        obj.states[target_state].set_value(new_value=True)
        return [obj]

    return boolean_state_randomizer
