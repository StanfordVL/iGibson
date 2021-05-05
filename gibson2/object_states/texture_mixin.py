from gibson2.object_states.object_state_base import BaseObjectState


class TextureChangeMixin(BaseObjectState):
    def __init__(self):
        super(TextureChangeMixin, self).__init__()
        self.material = None

    def update(self, simulator):
        super(TextureChangeMixin, self).update(simulator)
        if self.material is not None:
            self.material.change_material(self.get_value())
