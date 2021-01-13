from gibson2.object_states.pose import Pose
from gibson2.object_states.aabb import AABB
from gibson2.object_states.contact_bodies import ContactBodies


def get_object_state_instance(state_name, obj, online=True):
    if state_name == 'pose':
        return Pose(obj, online)
    elif state_name == 'aabb':
        return AABB(obj, online)
    elif state_name == 'contact_bodies':
        return ContactBodies(obj, online)
    else:
        assert False, 'unknown state name: {}'.format(state_name)
