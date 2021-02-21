from gibson2.object_states.object_state_base import BaseObjectState


class TextureChangeMixin(BaseObjectState):
    """
    TBA
    """

    def __init__(self, obj):
        super(TextureChangeMixin, self).__init__(obj)
        self.material_callback = None

    def update(self, simulator):
        super(TextureChangeMixin, self).update(simulator)
        if self.material_callback is not None:
            self.material_callback(self.get_value())
