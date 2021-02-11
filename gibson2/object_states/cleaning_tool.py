from gibson2.object_states.object_state_base import AbsoluteObjectState

class CleaningTool(AbsoluteObjectState):

    def __init__(self, obj):
        super(CleaningTool, self).__init__(obj)

    def update(self, simulator):
        pass

    def set_value(self, new_value):
        pass

    def get_value(self):
        pass

    @staticmethod
    def get_optional_dependencies():
        return ["toggled_open"]