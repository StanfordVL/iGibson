from gibson2.object_states.factory import ALL_STATES
from gibson2.object_states.object_state_base import BooleanState
from gibson2.objects.object_base import Object
from gibson2.objects.stateful_object import StatefulObject
from IPython import embed


class ObjectGrouper(StatefulObject):
    """A multi-object wrapper that groups multiple objects and applies operations to all of them in parallel."""

    class StateAggregator(object):
        """A fake state that aggregates state between ObjectGrouper objects and propagates updates."""

        def __init__(self, state_type, object_grouper):
            self.state_type = state_type
            self.object_grouper = object_grouper

        def get_value(self):
            if not isinstance(self.state_type, BooleanState):
                raise ValueError(
                    "Aggregator can only aggregate boolean states.")

            return all(obj.states[self.state_type].get_value() for obj in self.object_grouper.objects)

        def set_value(self):
            raise ValueError("Cannot set state on StateAggregator.")

        def update(self, simulator):
            for obj in self.object_grouper.objects:
                obj.states[self.state_type].update(simulator)

    def __init__(self, objects):
        super(StatefulObject, self).__init__()

        assert objects and all(isinstance(obj, Object) for obj in objects)
        self.objects = objects

        self.states = {
            state_type: ObjectGrouper.StateAggregator(state_type, self)
            for state_type in ALL_STATES}

    def __getattr__(self, item):
        # Check if the attr is the same for everything
        attrs = [getattr(obj, item) for obj in self.objects]

        # If the attribute is a method, let's return a wrapper function that calls the method
        # on each object.
        if callable(attrs[0]):
            def grouped_function(*args, **kwargs):
                rets = [getattr(obj, item)(*args, **kwargs)
                        for obj in self.objects]
                if rets.count(rets[0]) != len(rets):
                    raise ValueError(
                        "Methods on grouped objects had different results.")

                return rets[0]

            return grouped_function

        # Otherwise, check that it's the same for everyone and then just return the value.
        if attrs.count(attrs[0]) != len(attrs):
            raise ValueError(
                "Grouped objects had different values for this attribute.")

        return attrs[0]

    def _load(self):
        body_ids = []
        for obj in self.objects:
            body_ids += obj._load()
        return body_ids


class ObjectMultiplexer(StatefulObject):
    """A multi-object wrapper that acts as a proxy for the selected one between the set of objects it contains."""

    def __init__(self, multiplexed_objects, current_index):
        super(StatefulObject, self).__init__()

        assert multiplexed_objects and all(isinstance(
            obj, Object) for obj in multiplexed_objects)
        assert 0 <= current_index < len(multiplexed_objects)

        self._multiplexed_objects = multiplexed_objects
        self.current_index = current_index

        # Combine the abilities in a parameterless manner.
        ability_set = set()
        for obj in self._multiplexed_objects:
            ability_set.update(obj.abilities.keys())

        # TODO: Think about whether this makes sense.
        self.abilities = {ability: {} for ability in ability_set}

    def set_selection(self, idx):
        assert 0 <= idx < len(self.multiplexed_objects)
        self.current_index = idx

    def current_selection(self):
        return self._multiplexed_objects[self.current_index]

    def __getattr__(self, item):
        return getattr(self.current_selection(), item)

    def _load(self):
        body_ids = []
        for obj in self._multiplexed_objects:
            body_ids += obj._load()
        return body_ids
