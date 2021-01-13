
from gibson2.object_properties.object_property_base import BaseObjectProperty


class Kinematics(BaseObjectProperty):

    @staticmethod
    def get_relevant_states():
        return {'pose', 'aabb', 'contact_bodies'}
