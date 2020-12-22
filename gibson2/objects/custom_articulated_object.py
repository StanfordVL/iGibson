from gibson2.objects.articulated_object import ArticulatedObject


class CustomArticulatedObject(ArticulatedObject):
    """
    A custom class that extends the articulated object by assigning this object a name, keeping track of its class id,
    and providing sampling functions for automatically sampling positions / orientations for this object

    Args:
        name (str): Name to assign this object -- should be unique

        filename (str): fpath to the urdf associated with this object

        scale (float): relative scale of the object when loading

        class_id (int): integer to assign to this object when using semantic segmentation

        pos_sampler (None or function): function that should take no args and return a 3-tuple for the
            global (x,y,z) cartesian position values for the object. None results in a default pos being
            generated, as handled per env

        ori_sampler (None or function): function that should take no args and return a 4-tuple for the
            global (x,y,z,w) quaternion for the object. None results in a default ori being generated, as handled
            per env
    """
    def __init__(
        self,
        name,
        class_id,
        pos_sampler,
        ori_sampler,
        filename,
        scale=1,
    ):
        # Run super init first
        super().__init__(filename=filename, scale=scale)

        # Store other internal vars
        self.name = name
        self.class_id = class_id
        self.pos_sampler = pos_sampler
        self.ori_sampler = ori_sampler

    def sample_pose(self):
        """
        Samples a new pose for this object.

        Returns:
            2-tuple:
                3-array: (x,y,z) cartesian global pos for this object
                4-array: (x,y,z,w) quaternion global orientation for this object
        """
        assert self.pos_sampler is not None and self.ori_sampler is not None, "Samplers still need to be added!"
        return self.pos_sampler(), self.ori_sampler()

    def update_pos_sampler(self, pos_sampler):
        """
        Updates the internal position sampler

        Args:
            pos_sampler (function): function that should take no args and return a 3-tuple for the
                global (x,y,z) cartesian position values for the object.
        """
        self.pos_sampler = pos_sampler

    def update_ori_sampler(self, ori_sampler):
        """
        Updates the internal orientation sampler

        Args:
            ori_sampler (None or function): function that should take no args and return a 4-tuple for the
                global (x,y,z,w) quaternion for the object.
        """
        self.ori_sampler = ori_sampler