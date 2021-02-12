from gibson2.external.pybullet_tools.utils import link_from_name, get_link_state
from gibson2.object_states.object_state_base import CachingEnabledObjectState
from gibson2.object_states.utils import get_aabb_center

# The name of the heating element link inside URDF files.
HEATING_ELEMENT_LINK_NAME = "heating_element"


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

        # Try to get heating element position from URDF
        try:
            body_id = self.obj.get_body_id()
            link_id = link_from_name(body_id, HEATING_ELEMENT_LINK_NAME)
            heating_element_state = get_link_state(body_id, link_id)
            return heating_element_state.linkWorldPosition
        except ValueError:
            return None

    def set_value(self, new_value):
        raise NotImplementedError("Setting heat source capability is not supported.")
