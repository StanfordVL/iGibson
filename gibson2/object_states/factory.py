from gibson2.object_states.pose import Pose
from gibson2.object_states.aabb import AABB
from gibson2.object_states.contact_bodies import ContactBodies
from gibson2.object_states.dummy_state import DummyState
from gibson2.object_states.temperature import Temperature

STATE_NAME_TO_CLASS_MAPPING = {
    'pose': Pose,
    'aabb': AABB,
    'contact_bodies': ContactBodies,
    'temperature': Temperature,
}


def get_object_state_instance(state_name, obj, online=True):
    if state_name not in STATE_NAME_TO_CLASS_MAPPING:
        assert False, 'unknown state name: {}'.format(state_name)

    if not online:
        return DummyState(obj)

    state_class = STATE_NAME_TO_CLASS_MAPPING[state_name]
    return state_class(obj)
