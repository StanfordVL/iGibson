class TextureChangeStateMixin(object):
    def __init__(self):
        super(TextureChangeStateMixin, self).__init__()
        self.material = None

    def update_texture(self):
        # Assume only state evaluated True will need non-default texture
        if self.material is not None and self.get_value():
            self.material.request_texture_change(
                self.__class__,
            )
