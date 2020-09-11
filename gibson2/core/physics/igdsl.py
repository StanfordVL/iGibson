import pybullet as p


# TODO get scene, or at least what's in it? 


########## OBJECTS ##########

class BaseObject(object):

    def __init__(object):
        self._position = None
        self._orientation = None
    
    def load(self):
        '''
        load into pybullet 
        '''
        pass

    def get_position(self):
        pass

    def get_orientation(self):
        pass

    def get_position_orientation(self):
        pass

    def set_position(self, pos):
        pass

    def set_orientation(self, orn):
        pass

    def set_position_orientation(self, pos, orn):
        pass

    def get_category(self):
        pass

    @property
    def size(self):
        '''
        get object size
        '''
        pass
    
    def aabb(self):
        '''
        get axis aligned bounding box
        '''
        # return xmin, xmax, ymin, ymax, zmin, zmax
        pass

class Food(BaseObject):     # split subclasses into files? 
    self.properties['out_of_date'] = Perishable(self, date)       # TODO implement self.properties

class Fruit(Food):      # cookable? 
    pass

class Meat(Food, Cookable):
    def __init__(self):
        self.properties['cookable'] = Cookable(self)

class Apple(Fruit):
    def __init__(self, othr_objs, cookable_threshold):
        super() # TODO complete 
        self.properties['cookable'].set_threshold(50)
        self.properties['burnable'] # TODO complete
        self.properties['onTop'] = onTop(self)
        self.temp = 60
        self.properties['burnable'].is_true(self.temp)      # TODO here?

    def update_obj_state():
        self.temp += 20

    def step_state(self):
        for prop in self.properties:
            prop.step(self)

# TODO see dessign doc for use 

class Oven(BaseObject, Toggleable, Openable):   # TODO remove properties as superclasses
    def __init__(self):
        super().__init__()
    
    def step_state():
        if p.getJointState(self.body_id, self._toggle_axis)[0] < self._toggle_threshold:
            self.is_on = True
        else:
            self.is_on = False

        if p.getJointState(self.body_id, self._open_axis)[0] < self.    

 
########## OBJECT STATES (BINARY AND UNARY) ##########

class O2ORelationship(object):
    '''
    Handles binary object-to-object relationships
    '''
    def __init__(self):
        pass

    def __call__(self, object_a, object_b):
        pass


class AonB(O2ORelationship):
    def __call__(self, object_a, object_b):
        # NOTE: defined by aabb of object_a and object_b
        pass

class AinB(O2ORelationship):
    def __call__(self, object_a, object_b):
        # NOTE: defined by aabb of object_a and object_b
        pass

class ObjectState(object):
    '''
    Handles unary object states
    '''

class Toggleable(ObjectState):
    def __init__(self):
        self.is_on = False
        self._toggle_axis = o
        self._toggle_threshold = 0

    def set_axis(self, axis):
        self._toggle_axis = axis        # TODO turn into property?
    
    def set_threshold(self, threshold):
        self._toggle_threshold = threshold 

class Openable(ObjectState)
    def __init__(self):
        self.is_open = False

class Moveable(ObjectState):
    pass

class Dirty(ObjectState):
    def __init__(self):
        self.is_dirty = False

class OnTop(ObjectState):
    def __init__(self):
        pass

    def is_true(self, another_obj):
        return p.getPosition(self)[2] > p.getPosition(another_obj)[2]

class Cookable(ObjectState):
    def __init__(self):
        self.temp = 0
        self.temp_threshold = 0

    def set_threshold(self, threshold):
        self.temp_threshold = threshold

    def is_true():
        return self.temp > self.temp_threshold



# TODO: Task, Env, Scene 
























