

class ObjectProperty(object):
    '''
    Handles unary object states
    '''
    def __init__(self):
        pass
        # TODO
    

    def is_true(self, compare_object, compare=None):
        pass 


########## SPECIFIC PROPERTIES ##########

class Classifiable(ObjectProperty):
    def __init__(self): 
        ObjectProperty.__init__(self)
    
    def is_true(self, compare_object, compare=None):
        return isinstance(compare_object, eval(compare))


class Toggleable(ObjectProperty):
    def __init__(self, toggleable_th=0):
        ObjectProperty.__init__(self)
        self._toggleable_th = toggleable_th

    def set_threshold(self, toggleable_th):
        self._toggleable_th = toggleable_th

    def is_true(self, compare_object, compare=None):
        return compare_object.get_joint_state(compare_object._toggleable_joint) > self._toggleable_th


class Openable(ObjectProperty):
    def __init__(self, openable_th=0):
        self._openable_th = openable_th
    
    def set_threshold(self, openable_th):
        self._openable_th = openable_th

    def is_true(self, compare_object, compare=None):
        return object.get_joint_state(compare_object._openable_joint) > self._openable_th   # TODO toggleable_joint?


class Moveable(ObjectProperty):
    pass


class Cookable(ObjectProperty):
    def __init__(self, temp_th=100):
        self._temp_th = temp_th

    def set_threshold(self, temp_th):
        self._temp_th = temp_th
   
    def is_true(self, compare_object, compare=None):
        return compare_object > self._temp_th

   
class Perishable(ObjectProperty):
    def __init__(self, expiration_date=date.today()):
        self._expiration_date = expiration_date
    
    def set_threshold(self, expiration_date):
        self._expiration_date = expiration_date

    def is_true(self, compare_object, compare=None):
        return compare_object._expiration_date > self._expiration_date


class Burnable(ObjectProperty):
    def __init__(self, temp_th=100):
        self._temp_th = temp_th
    
    def set_threshold(self, temp_th):
        self._temp_th = temp_th
    
    def is_true(self, compare_object, compare=None):
        return compare_object > self._temp_th       # though it should stay burnt even if it cools off, same with cookables
































