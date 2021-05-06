from tasknet.backend_abc import TaskNetBackend
from tasknet.logic_base import UnaryAtomicPredicate, BinaryAtomicPredicate

from gibson2 import object_states


class ObjectStateUnaryPredicate(UnaryAtomicPredicate):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, obj, **kwargs):
        return obj.states[self.STATE_CLASS].get_value(**kwargs)

    def _sample(self, obj, binary_state, **kwargs):
        return obj.states[self.STATE_CLASS].set_value(binary_state, **kwargs)


class ObjectStateBinaryPredicate(BinaryAtomicPredicate):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, obj1, obj2, **kwargs):
        return obj1.states[self.STATE_CLASS].get_value(obj2, **kwargs)

    def _sample(self, obj1, obj2, binary_state, **kwargs):
        return obj1.states[self.STATE_CLASS].set_value(obj2, binary_state, **kwargs)


def get_unary_atomic_predicate_for_state(state_class, state_name):
    return type(state_class.__name__ + "StateUnaryPredicate", (ObjectStateUnaryPredicate,),
                {'STATE_CLASS': state_class, 'STATE_NAME': state_name})


def get_binary_atomic_predicate_for_state(state_class, state_name):
    return type(state_class.__name__ + "StateBinaryPredicate", (ObjectStateBinaryPredicate,),
                {'STATE_CLASS': state_class, 'STATE_NAME': state_name})


# TODO: Add remaining predicates.
SUPPORTED_PREDICATES = {
    'inside': get_binary_atomic_predicate_for_state(object_states.Inside, 'inside'),
    'nextto': get_binary_atomic_predicate_for_state(object_states.NextTo, 'nextto'),
    'ontop': get_binary_atomic_predicate_for_state(object_states.OnTop, 'ontop'),
    'under': get_binary_atomic_predicate_for_state(object_states.Under, 'under'),
    'touching': get_binary_atomic_predicate_for_state(object_states.Touching, 'touching'),
    'onfloor': get_binary_atomic_predicate_for_state(object_states.OnFloor, 'onfloor'),
    'sliced': get_unary_atomic_predicate_for_state(object_states.Sliced, 'sliced'),
}


class IGibsonTaskNetBackend(TaskNetBackend):
    def get_predicate_class(self, predicate_name):
        return SUPPORTED_PREDICATES[predicate_name]
