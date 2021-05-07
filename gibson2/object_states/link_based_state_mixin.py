from gibson2.external.pybullet_tools.utils import link_from_name, get_link_state


class LinkBasedStateMixin(object):
    def __init__(self):
        super(LinkBasedStateMixin, self).__init__()

        self.body_id = None
        self.link_id = None

    @staticmethod
    def get_state_link_name():
        raise ValueError(
            "LinkBasedState child should specify link name by overriding get_state_link_name.")

    def initialize_link_mixin(self):
        # If we haven't tried to load the link before
        if self.body_id is None:
            # Get the body id
            self.body_id = self.obj.get_body_id()

            try:
                self.link_id = link_from_name(
                    self.body_id, self.get_state_link_name())
            except ValueError:
                pass

    def get_link_position(self):
        # The necessary link is not found
        if self.link_id is None:
            return

        return get_link_state(self.body_id, self.link_id).linkWorldPosition
