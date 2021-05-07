import gibson2
import pybullet as p
from IPython import embed
from gibson2.object_states.adjacency import VerticalAdjacency
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.touching import Touching
from gibson2.object_states.utils import clear_cached_states, sample_kinematics


class OnTop(RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return RelativeObjectState.get_dependencies() + [Touching, VerticalAdjacency]

    def set_value(self, other, new_value, use_ray_casting_method=False):
        state_id = p.saveState()

        for _ in range(10):
            sampling_success = sample_kinematics(
                'onTop', self.obj, other, new_value,
                use_ray_casting_method=use_ray_casting_method)
            if sampling_success:
                clear_cached_states(self.obj)
                clear_cached_states(other)
                if self.get_value(other) != new_value:
                    sampling_success = False
                if gibson2.debug_sampling:
                    print('OnTop checking', sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                p.restoreState(state_id)

        p.removeState(state_id)

        return sampling_success

    def get_value(self, other, use_ray_casting_method=False):
        del use_ray_casting_method

        # Touching is the less costly of our conditions.
        # Check it first.
        if not self.obj.states[Touching].get_value(other):
            return False

        # Then check vertical adjacency - it's the second least
        # costly.
        adjacency = self.obj.states[VerticalAdjacency].get_value()
        return (
            other.get_body_id() in adjacency.negative_neighbors and
            other.get_body_id() not in adjacency.positive_neighbors)
