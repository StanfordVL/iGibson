from tasknet.backend_abc import TaskNetBackend
from tasknet.logic_base import UnaryAtomicPredicate, BinaryAtomicPredicate

from gibson2 import object_states


class ObjectStateUnaryPredicate(UnaryAtomicPredicate):
    STATE_CLASS = None

    def _evaluate(self, obj):
        return obj.states[self.STATE_CLASS].get_value()

    def _sample(self, obj, binary_state):
        return obj.states[self.STATE_CLASS].set_value(binary_state)


class ObjectStateBinaryPredicate(BinaryAtomicPredicate):
    STATE_CLASS = None

    def _evaluate(self, obj1, obj2):
        return obj1.states[self.STATE_CLASS].get_value(obj2)

    def _sample(self, obj1, obj2, binary_state):
        return obj1.states[self.STATE_CLASS].set_value(obj2, binary_state)


def get_unary_atomic_predicate_for_state(state_class):
    return type(state_class.__name__ + "StateUnaryPredicate", (ObjectStateUnaryPredicate,),
                {'STATE_CLASS': state_class})


def get_binary_atomic_predicate_for_state(state_class):
    return type(state_class.__name__ + "StateBinaryPredicate", (ObjectStateBinaryPredicate,),
                {'STATE_CLASS': state_class})


# TODO: Add remaining predicates.
SUPPORTED_PREDICATES = {
    'inside': get_binary_atomic_predicate_for_state(object_states.Inside),
    'nextto': get_binary_atomic_predicate_for_state(object_states.NextTo),
    'ontop': get_binary_atomic_predicate_for_state(object_states.OnTop),
    'under': get_binary_atomic_predicate_for_state(object_states.Under),
    'touching': get_binary_atomic_predicate_for_state(object_states.Touching),
    'onfloor': get_binary_atomic_predicate_for_state(object_states.OnFloor),
}


class IGibsonTaskNetBackend(TaskNetBackend):
    def get_predicate_class(self, predicate_name):
        return SUPPORTED_PREDICATES[predicate_name]
