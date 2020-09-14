import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


class Scene:
    def __init__(self):
        self.is_interactive = False

    def load(self):
        raise NotImplementedError()
