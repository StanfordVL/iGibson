from gibson2.object_states.vertical_adjacency import VerticalAdjacency
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState
from gibson2.object_states.utils import sample_kinematics, clear_cached_states
import gibson2
from IPython import embed
import pybullet as p


class Under(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + [VerticalAdjacency]

    def set_value(self, other, new_value):
        state_id = p.saveState()

        for _ in range(10):
            sampling_success = sample_kinematics(
                'under', self.obj, other, new_value)
            if sampling_success:
                clear_cached_states(self.obj)
                clear_cached_states(other)
                if self.get_value(other) != new_value:
                    sampling_success = False
                if gibson2.debug_sampling:
                    print('Under checking', sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                p.restoreState(state_id)

        p.removeState(state_id)

        return sampling_success

    def get_value(self, other):
        adjacency = self.obj.states[VerticalAdjacency].get_value()

        return other.get_body_id() in adjacency[1]
