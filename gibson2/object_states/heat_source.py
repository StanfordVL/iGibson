from gibson2.object_states.object_state_base import CachingEnabledObjectState
from gibson2.object_states.utils import get_aabb_center


class HeatSource(CachingEnabledObjectState):
    """
    This state indicates the heat source state of the object.

    Currently, if the object is not an active heat source, this returns None. Otherwise, it returns the position of the
    heat source element. E.g. on a stove object the coordinates of the heating element will be returned.
    """

    @staticmethod
    def get_dependencies():
        return CachingEnabledObjectState.get_dependencies() + ["aabb"]

    def _compute_value(self):
        # TODO: Implement checking support for non-openness as a prerequisite for heating on certain elements.

        # For now we do not support the heating element position - we just return the object's center position.
        aabb = self.obj.states['aabb'].get_value()
        return get_aabb_center(aabb)

    def set_value(self, new_value):
        raise NotImplementedError("Setting heat source capability is not supported.")
