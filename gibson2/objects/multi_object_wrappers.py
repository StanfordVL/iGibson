from gibson2.object_states.factory import ALL_STATES
from gibson2.object_states.object_state_base import BooleanState
from gibson2.objects.object_base import Object
from gibson2.objects.stateful_object import StatefulObject
from gibson2.object_states.object_state_base import AbsoluteObjectState
from IPython import embed
import pybullet as p


class ObjectGrouper(StatefulObject):
    """A multi-object wrapper that groups multiple objects and applies operations to all of them in parallel."""

    class BaseStateAggregator(object):
        """A fake state that aggregates state between ObjectGrouper objects and propagates updates."""

        def __init__(self, state_type, object_grouper):
            self.state_type = state_type
            self.object_grouper = object_grouper

        def update(self, simulator):
            for obj in self.object_grouper.objects:
                obj.states[self.state_type].update(simulator)

    class AbsoluteStateAggregator(BaseStateAggregator):
        def get_value(self):
            if not issubclass(self.state_type, BooleanState):
                raise ValueError(
                    "Aggregator can only aggregate boolean states.")

            return all(obj.states[self.state_type].get_value() for obj in self.object_grouper.objects)

        def set_value(self, new_value):
            for obj in self.object_grouper.objects:
                obj.states[self.state_type].set_value(new_value)

    class RelativeStateAggregator(BaseStateAggregator):
        def get_value(self, other):
            if not issubclass(self.state_type, BooleanState):
                raise ValueError(
                    "Aggregator can only aggregate boolean states.")

            return all(obj.states[self.state_type].get_value(other) for obj in self.object_grouper.objects)

        def set_value(self, other, new_value):
            for obj in self.object_grouper.objects:
                obj.states[self.state_type].set_value(other, new_value)

    def __init__(self, objects_with_pose_offsets):
        super(StatefulObject, self).__init__()

        objects = [obj for obj, _ in objects_with_pose_offsets]
        pose_offsets = [trans for _, trans in objects_with_pose_offsets]
        assert objects and all(isinstance(obj, Object) for obj in objects)
        self.objects = objects
        self.pose_offsets = pose_offsets

        self.states = {
            state_type:
            ObjectGrouper.AbsoluteStateAggregator(state_type, self)
            if issubclass(state_type, AbsoluteObjectState)
            else
            ObjectGrouper.RelativeStateAggregator(state_type, self)
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

    def get_position(self):
        raise ValueError("Cannot get_position on ObjectGrouper")

    def get_orientation(self):
        raise ValueError("Cannot get_orientation on ObjectGrouper")

    def get_position_orientation(self):
        raise ValueError("Cannot get_position_orientation on ObjectGrouper")

    def set_position(self, pos):
        raise ValueError(
            "Cannot set_position on ObjectGrouper")

    def set_orientation(self, orn):
        raise ValueError(
            "Cannot set_orientation on ObjectGrouper")

    def set_position_orientation(self, pos, orn):
        for obj, (part_pos, part_orn) in zip(self.objects, self.pose_offsets):
            new_pos, new_orn = p.multiplyTransforms(
                pos, orn, part_pos, part_orn)
            obj.set_position_orientation(new_pos, new_orn)

    def set_base_link_position_orientation(self, pos, orn):
        raise ValueError(
            "Cannot set_base_link_position_orientation on ObjectGrouper")

    def rotate_by(self, x=0, y=0, z=0):
        raise ValueError("Cannot rotate_by on ObjectGrouper")


class ObjectMultiplexer(StatefulObject):
    """A multi-object wrapper that acts as a proxy for the selected one between the set of objects it contains."""

    def __init__(self, multiplexed_objects, current_index):
        super(StatefulObject, self).__init__()

        assert multiplexed_objects and all(isinstance(
            obj, Object) for obj in multiplexed_objects)
        assert 0 <= current_index < len(multiplexed_objects)

        for obj in multiplexed_objects:
            obj.multiplexer = self

        self._multiplexed_objects = multiplexed_objects
        self.current_index = current_index

        # TODO: why is this needed?
        # Combine the abilities in a parameterless manner.
        ability_set = set()
        for obj in self._multiplexed_objects:
            ability_set.update(obj.abilities.keys())

        # TODO: Think about whether this makes sense.
        self.abilities = {ability: {} for ability in ability_set}

        # This will help route obj.states to one of the multiplexed_objects
        del self.states

    def set_selection(self, idx):
        assert 0 <= idx < len(self._multiplexed_objects)
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

    def get_position(self):
        return self.current_selection().get_position()

    def get_orientation(self):
        return self.current_selection().get_orientation()

    def get_position_orientation(self):
        return self.current_selection().get_position_orientation()

    def set_position(self, pos):
        return self.current_selection().set_position(pos)

    def set_orientation(self, orn):
        return self.current_selection().set_orientation(orn)

    def set_position_orientation(self, pos, orn):
        return self.current_selection().set_position_orientation(pos, orn)

    def set_base_link_position_orientation(self, pos, orn):
        return self.current_selection().set_base_link_position_orientation(pos, orn)

    def rotate_by(self, x=0, y=0, z=0):
        return self.current_selection().rotate_by(x, y, z)
