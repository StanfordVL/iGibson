from gibson2.external.pybullet_tools.utils import link_from_name, get_link_state


class LinkBasedStateMixin(object):
    def __init__(self):
        super(LinkBasedStateMixin, self).__init__()

        # This variable indicates that the object does not have the necessary link.
        self.link_missing = False
        self.link_id = None
        self.body_id = None

    @staticmethod
    def get_state_link_name():
        raise ValueError("LinkBasedState child should specify link name by overriding get_state_link_name.")

    def _load_link(self):
        # If we need the link info, get it now.
        if self.link_id is None or self.body_id is None:
            # Get the body id
            self.body_id = self.obj.get_body_id()

            try:
                self.link_id = link_from_name(self.body_id, self.get_state_link_name())
            except ValueError:
                self.link_missing = True

    def get_link_position(self):
        # Stop if we already tried to find the link & couldn't.
        if self.link_missing:
            return

        # Load the link if necessary.
        self._load_link()

        return get_link_state(self.body_id, self.link_id).linkWorldPosition
