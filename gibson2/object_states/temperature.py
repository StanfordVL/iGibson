from gibson2.object_states.object_state_base import AbsoluteObjectState

# TODO: Consider sourcing default temperature from scene
DEFAULT_TEMPERATURE = 23  # degrees Celsius


class Temperature(AbsoluteObjectState):

    def __init__(self, obj):
        super(Temperature, self).__init__(obj)

        self.value = DEFAULT_TEMPERATURE

    def set_value(self, new_value):
        self.value = new_value

    def update(self, simulator):
        # TODO: Implement temperature updates.
        pass
