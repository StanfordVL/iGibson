import random
from dataclasses import dataclass
from typing import Type

from igibson.object_states.object_state_base import BaseObjectState


@dataclass
class ObjectBooleanStatePerturber:
    target_state: Type[BaseObjectState]

    def __call__(self, env):
        scene = env.simulator.scene
        objects = scene.get_objects_with_state(self.target_state)
        obj = random.choice(objects)
        obj.states[self.target_state].set_value(new_value=True)
        return [obj]
