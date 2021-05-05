from gibson2.object_states.object_state_base import BaseObjectState


class TextureChangeState(BaseObjectState):
    def __init__(self, obj):
        super(TextureChangeState, self).__init__(obj)
        self.material = None

    def update(self, simulator):
        super(TextureChangeState, self).update(simulator)

        if self.material is not None:
            self.material.change_material(self.get_value())
