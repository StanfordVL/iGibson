from gibson2.object_properties.on_top import OnTop
from gibson2.object_properties.inside import Inside
from gibson2.object_properties.next_to import NextTo
from gibson2.object_properties.under import Under
from gibson2.object_properties.touching import Touching


def get_all_object_properties():
    return set({'onTop', 'inside', 'nextTo', 'under', 'touching'})


def get_object_property_class(property_name):
    if property_name == 'onTop':
        return OnTop
    elif property_name == 'inside':
        return Inside
    elif property_name == 'nextTo':
        return NextTo
    elif property_name == 'under':
        return Under
    elif property_name == 'touching':
        return Touching
    else:
        assert False, 'unknown property name: {}'.format(property_name)
