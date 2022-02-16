import logging

from igibson.object_states.factory import get_state_name, prepare_object_states
from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.object_states.utils import clear_cached_states
from igibson.objects.object_base import BaseObject

# Optionally import bddl for object taxonomy.
try:
    from bddl.object_taxonomy import ObjectTaxonomy

    OBJECT_TAXONOMY = ObjectTaxonomy()
except ImportError:
    print("BDDL could not be imported - object taxonomy / abilities will be unavailable.", file=sys.stderr)
    OBJECT_TAXONOMY = None

log = logging.getLogger(__name__)


class StatefulObject(BaseObject):
    """Objects that support object states."""

    def __init__(self, abilities=None, **kwargs):
        super(StatefulObject, self).__init__(**kwargs)

        # Load abilities from taxonomy if needed & possible
        if abilities is None:
            if OBJECT_TAXONOMY is not None:
                taxonomy_class = OBJECT_TAXONOMY.get_class_name_from_igibson_category(self.category)
                if taxonomy_class is not None:
                    abilities = OBJECT_TAXONOMY.get_abilities(taxonomy_class)
                else:
                    abilities = {}
            else:
                abilities = {}
        assert isinstance(abilities, dict), "Object abilities must be in dictionary form."

        prepare_object_states(self, abilities=abilities)

    def load(self, simulator):
        body_ids = super(StatefulObject, self).load(simulator)
        for state in self.states.values():
            state.initialize(simulator)

        return body_ids

    def dump_state(self):
        return {
            get_state_name(state_type): state_instance.dump()
            for state_type, state_instance in self.states.items()
            if issubclass(state_type, AbsoluteObjectState)
        }

    def load_state(self, dump):
        for state_type, state_instance in self.states.items():
            state_name = get_state_name(state_type)
            if issubclass(state_type, AbsoluteObjectState):
                if state_name in dump:
                    state_instance.load(dump[state_name])
                else:
                    log.debug("Missing object state [{}] in the state dump".format(state_name))

    def set_position_orientation(self, pos, orn):
        super(StatefulObject, self).set_position_orientation(pos, orn)
        clear_cached_states(self)

    def set_base_link_position_orientation(self, pos, orn):
        super(StatefulObject, self).set_base_link_position_orientation(pos, orn)
        clear_cached_states(self)

    def set_poses(self, poses):
        super(StatefulObject, self).set_poses(poses)
        clear_cached_states(self)
