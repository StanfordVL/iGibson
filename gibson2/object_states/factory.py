from gibson2.object_states.inside import Inside
from gibson2.object_states.next_to import NextTo
from gibson2.object_states.on_top import OnTop
from gibson2.object_states.pose import Pose
from gibson2.object_states.aabb import AABB
from gibson2.object_states.contact_bodies import ContactBodies
from gibson2.object_states.dummy_state import DummyState
from gibson2.object_states.temperature import Temperature
from gibson2.object_states.touching import Touching
from gibson2.object_states.under import Under

_STATE_NAME_TO_CLASS_MAPPING = {
    'pose': Pose,
    'aabb': AABB,
    'contact_bodies': ContactBodies,
    'temperature': Temperature,
    'onTop': OnTop,
    'inside': Inside,
    'nextTo': NextTo,
    'under': Under,
    'touching': Touching,
}

_ABILITY_TO_STATE_MAPPING = {
    "cookable": ["cooked"],
}

_DEFAULT_STATE_SET = {
    'onTop',
    'inside',
    'nextTo',
    'under',
    'touching'
}


def get_default_state_names():
    return set(_DEFAULT_STATE_SET)


def get_all_state_names():
    return set(_STATE_NAME_TO_CLASS_MAPPING.keys())


def get_state_names_for_ability(ability):
    return _ABILITY_TO_STATE_MAPPING[ability]


def get_object_state_instance(state_name, obj, online=True):
    if state_name not in _STATE_NAME_TO_CLASS_MAPPING:
        assert False, 'unknown state name: {}'.format(state_name)

    if not online:
        return DummyState(obj)

    state_class = _STATE_NAME_TO_CLASS_MAPPING[state_name]
    return state_class(obj)
