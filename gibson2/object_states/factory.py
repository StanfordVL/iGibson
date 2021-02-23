import networkx as nx
from gibson2.object_states import *
from gibson2.object_states.object_state_base import BaseObjectState

_ALL_STATES = frozenset([
    AABB,
    Burnt,
    ContactBodies,
    Cooked,
    DummyState,
    HeatSource,
    Inside,
    MaxTemperature,
    NextTo,
    OnTop,
    Open,
    Pose,
    Temperature,
    Touching,
    Under,
    Soaked,
    Dirty,
    Stained,
    ToggledOn,
    WaterSource,
    CleaningTool
])

_ABILITY_TO_STATE_MAPPING = {
    "cookable": [Cooked],
    "soakable": [Soaked],
    "dustable": [Dirty],
    "scrubbable": [Stained],
    "water_source": [WaterSource],
    "cleaning_tool": [CleaningTool],
    "toggleable": [ToggledOn],
    "burnable": [Burnt],
    "heatSource": [HeatSource]
}

_DEFAULT_STATE_SET = frozenset([
    OnTop,
    Inside,
    NextTo,
    Under,
    Touching,
    Open,
])


def get_default_states():
    return _DEFAULT_STATE_SET


def get_all_states():
    return _ALL_STATES


def get_states_for_ability(ability):
    return _ABILITY_TO_STATE_MAPPING[ability]


def get_object_state_instance(state_class, obj, params=None, online=True):
    """
    Create an BaseObjectState child class instance for a given object & state.

    The parameters passed in as a dictionary through params are passed as
    kwargs to the object state class constructor.

    :param state_class: The state name from the state name dictionary.
    :param obj: The object for which the state is being constructed.
    :param params: Dict of {param: value} corresponding to the state's params.
    :param online: Whether or not the instance should be generated for an online
        object. Offline mode involves using dummy objects rather than real state
        objects.
    :return: The constructed state object, an instance of a child of
        BaseObjectState.
    """
    if not issubclass(state_class, BaseObjectState):
        assert False, 'unknown state class: {}'.format(state_class)

    if not online:
        return DummyState(obj)

    if params is None:
        params = {}

    return state_class(obj, **params)


def prepare_object_states(obj, abilities=None, online=True):
    """
    Prepare the state dictionary for an object by generating the appropriate
    object state instances.

    This uses the abilities of the object and the state dependency graph to
    find & instantiate all relevant states.

    :param obj: The object to generate states for.
    :param abilities: dict in the form of {ability: {param: value}} containing
        object abilities and parameters.
    :param online: Whether or not the states should be generated for an online
        object. Offline mode involves using dummy objects rather than real state
        objects.
    """
    if abilities is None:
        abilities = {}

    state_types_and_params = [(state, {}) for state in get_default_states()]

    # Map the ability params to the states immediately imported by the abilities
    for ability, params in abilities.items():
        state_types_and_params.extend((state_type, params) for state_type in get_states_for_ability(ability))

    obj.states = dict()
    for state_type, params in state_types_and_params:
        obj.states[state_type] = get_object_state_instance(state_type, obj, params)

        # Add each state's dependencies, too. Note that only required dependencies are added.
        for dependency in obj.states[state_type].get_dependencies():
            if (dependency, {}) not in state_types_and_params:
                state_types_and_params.append((dependency, {}))


def get_state_dependency_graph():
    """
    Produce dependency graph of supported object states.
    """
    dependencies = {
        state: state.get_dependencies() + state.get_optional_dependencies()
        for state in get_all_states()}
    return nx.DiGraph(dependencies)


def get_states_by_dependency_order():
    """
    Produce a list of all states in topological order of dependency.
    """
    return list(reversed(list(nx.algorithms.topological_sort(get_state_dependency_graph()))))
