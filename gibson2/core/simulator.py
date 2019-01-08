import pybullet as p
from gibson2.core.physics.scene import StadiumScene, BuildingScene
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import *
from gibson2.core.physics.interactive_objects import *
from gibson2.core.viewer import Viewer

class Simulator:
    def __init__(self, gravity=9.8, timestep=1 / 240.0):

        # physics simulator
        self.gravity = gravity
        self.timestep = timestep
        self.cid = p.connect(p.GUI)
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -self.gravity)
        self.objects = []

        # renderer
        self.renderer = MeshRenderer(width=300, height=300)
        self.renderer.set_fov(90)
        self.visual_objects = {}

        self.add_viewer()

    def add_viewer(self):
        self.viewer = Viewer()
        self.viewer.renderer = self.renderer

    def import_scene(self, scene):
        new_objects = scene.load()
        for item in new_objects:
            self.objects.append(item)
        for new_object in new_objects:
            for shape in p.getVisualShapeData(new_object):
                id, _, type, _, filename  = shape[:5]
                if type == p.GEOM_MESH:
                    filename = filename.decode('utf-8')
                    if not filename in self.visual_objects.keys():
                        self.renderer.load_object(filename)
                        self.visual_objects[filename] = len(self.renderer.visual_objects)-1
                        self.renderer.add_instance(len(self.renderer.visual_objects)-1, new_object)
                    else:
                        self.renderer.add_instance(self.visual_objects[filename], new_object)

    def import_object(self, object):
        new_object = object.load()
        self.objects.append(new_object)
        for shape in p.getVisualShapeData(new_object):
            id, _, type, _, filename = shape[:5]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                print(filename, self.visual_objects)
                if not filename in self.visual_objects.keys():
                    self.renderer.load_object(filename)
                    self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                    self.renderer.add_instance(len(self.renderer.visual_objects) - 1, new_object, dynamic=True)
                else:
                    self.renderer.add_instance(self.visual_objects[filename], new_object, dynamic=True)

    def import_robot(self, robot):
        ids = robot.load()
        visual_objects = []
        link_ids = []
        for shape in p.getVisualShapeData(ids[0]):
            id, link_id, type, _, filename = shape[:5]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                if not filename in self.visual_objects.keys():
                    print(filename)
                    self.renderer.load_object(filename)
                    visual_objects.append(len(self.renderer.visual_objects) - 1)
                    self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                else:
                    visual_objects.append(self.visual_objects[filename])
                link_ids.append(link_id)
        self.renderer.add_instance_group(visual_objects, link_ids, ids[0], dynamic=True)
        return ids

    def step(self):
        p.stepSimulation()
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.update_position(instance)
        self.viewer.update()


    @staticmethod
    def update_position(instance):
        if isinstance(instance, Instance):
            pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)
            instance.set_position(pos)
            instance.set_rotation([orn[-1], orn[0], orn[1], orn[2]])
        elif isinstance(instance, InstanceGroup):
            for link_id in instance.link_ids:
                if link_id == -1:
                    pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)
                else:
                    _, _, _, _, pos, orn = p.getLinkState(instance.pybullet_uuid, link_id)

                print(instance.pybullet_uuid, link_id, pos, orn)
    def isconnected(self):
        return p.getConnectionInfo(self.cid)['isConnected']

    def disconnect(self):
        p.disconnect(self.cid)
        self.renderer.release()

if __name__ == '__main__':
    s = Simulator()
    scene = StadiumScene()
    s.import_scene(scene)
    obj = YCBObject('006_mustard_bottle')

    for i in range(10):
        s.import_object(obj)

    obj = YCBObject('002_master_chef_can')
    for i in range(10):
        s.import_object(obj)

    while s.isconnected():
        s.step()
