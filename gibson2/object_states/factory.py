import networkx as nx
from gibson2.object_states.aabb import AABB
from gibson2.object_states.burnt import Burnt
from gibson2.object_states.contact_bodies import ContactBodies
from gibson2.object_states.cooked import Cooked
from gibson2.object_states.dummy_state import DummyState
from gibson2.object_states.inside import Inside
from gibson2.object_states.max_temperature import MaxTemperature
from gibson2.object_states.next_to import NextTo
from gibson2.object_states.on_top import OnTop
from gibson2.object_states.open import Open
from gibson2.object_states.pose import Pose
from gibson2.object_states.temperature import Temperature
from gibson2.object_states.touching import Touching
from gibson2.object_states.under import Under

_STATE_NAME_TO_CLASS_MAPPING = {
    # Kinematic states
    'pose': Pose,
    'aabb': AABB,
    'contact_bodies': ContactBodies,
    'onTop': OnTop,
    'open': Open,
    'inside': Inside,
    'nextTo': NextTo,
    'under': Under,
    'touching': Touching,

    # Temperature / cooking states
    'temperature': Temperature,
    'maxTemperature': MaxTemperature,
    'burnt': Burnt,
    'cooked': Cooked,
}

_ABILITY_TO_STATE_MAPPING = {
    "cookable": ["cooked"],
    "burnable": ["burnt"],
}

_DEFAULT_STATE_SET = {
    'onTop',
    'inside',
    'nextTo',
    'under',
    'touching',
    'open',
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


def prepare_object_states(obj, abilities=[], online=True):
    state_names = list(get_default_state_names())

    for ability in abilities:
        state_names.extend(get_state_names_for_ability(ability))

    states = dict()
    for state_name in state_names:
        states[state_name] = get_object_state_instance(state_name, obj)

        # Add each state's dependencies, too. Note that only required dependencies are added.
        for dependency in states[state_name].get_dependencies():
            if dependency not in state_names:
                state_names.append(dependency)

    return states


def get_state_dependency_graph():
    """
    Produce dependency graph of supported object states.
    """
    dependencies = {
        state_name: (
                _STATE_NAME_TO_CLASS_MAPPING[state_name].get_dependencies() +
                _STATE_NAME_TO_CLASS_MAPPING[state_name].get_optional_dependencies())
        for state_name in get_all_state_names()}
    return nx.DiGraph(dependencies)


def get_states_by_dependency_order():
    """
    Produce a list of all states in topological order of dependency.
    """
    return list(reversed(list(nx.algorithms.topological_sort(get_state_dependency_graph()))))
