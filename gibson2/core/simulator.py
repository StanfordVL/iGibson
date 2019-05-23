from gibson2.core.physics.scene import StadiumScene
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import *
from gibson2.core.physics.interactive_objects import *
from gibson2.core.render.viewer import Viewer


class Simulator:
    def __init__(self,
                 gravity=9.8,
                 timestep=1 / 240.0,
                 use_fisheye=False,
                 mode='gui',
                 resolution=256,
                 device_idx=0):

        # physics simulator
        self.gravity = gravity
        self.timestep = timestep
        self.mode = mode
        # renderer
        self.resolution = resolution
        self.device_idx = device_idx
        self.use_fisheye = use_fisheye

        if self.mode == 'gui':
            self.viewer = Viewer()

        self.load()

    def set_timestep(self, timestep):
        self.timestep = timestep
        p.setTimeStep(self.timestep)

    def add_viewer(self):
        self.viewer = Viewer()
        self.viewer.renderer = self.renderer

    def reload(self):
        self.renderer.release()
        p.disconnect(self.cid)
        self.load()

    def load(self):
        if self.mode == 'gui':
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -self.gravity)

        self.renderer = MeshRenderer(width=self.resolution,
                                     height=self.resolution,
                                     device_idx=self.device_idx,
                                     use_fisheye=self.use_fisheye)
        self.renderer.set_fov(90)

        if self.mode == 'gui':
            self.viewer.renderer = self.renderer

        self.visual_objects = {}
        self.robots = []
        self.scene = None
        self.objects = []

    def import_scene(self, scene, texture_scale=1.0):
        new_objects = scene.load()
        for item in new_objects:
            self.objects.append(item)
        for new_object in new_objects:
            for shape in p.getVisualShapeData(new_object):
                id, _, type, _, filename = shape[:5]
                if type == p.GEOM_MESH:
                    filename = filename.decode('utf-8')
                    if not filename in self.visual_objects.keys():
                        self.renderer.load_object(filename, texture_scale=texture_scale)
                        self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                        self.renderer.add_instance(
                            len(self.renderer.visual_objects) - 1, new_object)

                    else:
                        self.renderer.add_instance(self.visual_objects[filename], new_object)
        self.scene = scene
        return new_objects

    def import_object(self, object):
        new_object = object.load()
        self.objects.append(new_object)
        for shape in p.getVisualShapeData(new_object):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                print(filename, self.visual_objects)
                if not filename in self.visual_objects.keys():
                    self.renderer.load_object(filename)
                    self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                    self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                               new_object,
                                               dynamic=True)
                else:
                    self.renderer.add_instance(self.visual_objects[filename],
                                               pybullet_uuid=new_object,
                                               dynamic=True)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                self.renderer.load_object(
                    filename,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                           pybullet_uuid=new_object,
                                           dynamic=True)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                           pybullet_uuid=new_object,
                                           dynamic=True)
            elif type == p.GEOM_BOX:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=[dimensions[0], dimensions[1], dimensions[2]])
                self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                           pybullet_uuid=new_object,
                                           dynamic=True)

        return new_object

    def import_robot(self, robot):
        ids = robot.load()
        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []
        self.robots.append(robot)

        for shape in p.getVisualShapeData(ids[0]):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                if not filename in self.visual_objects.keys():
                    print(filename, rel_pos, rel_orn, color, dimensions)
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))

                    visual_objects.append(len(self.renderer.visual_objects) - 1)
                    self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                else:
                    visual_objects.append(self.visual_objects[filename])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])

                visual_objects.append(len(self.renderer.visual_objects) - 1)
                #self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                link_ids.append(link_id)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])

                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=[dimensions[0], dimensions[1], dimensions[2]])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)

            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
            poses_rot.append(np.ascontiguousarray(quat2rotmat([orn[-1], orn[0], orn[1], orn[2]])))
            poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        self.renderer.add_robot(object_ids=visual_objects,
                                link_ids=link_ids,
                                pybullet_uuid=ids[0],
                                poses_rot=poses_rot,
                                poses_trans=poses_trans,
                                dynamic=True,
                                robot=robot)

        return ids

    def import_interactive_object(self, obj):
        ids = obj.load()
        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []

        for shape in p.getVisualShapeData(ids):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                if not filename in self.visual_objects.keys():
                    print(filename, rel_pos, rel_orn, color, dimensions)
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))

                    visual_objects.append(len(self.renderer.visual_objects) - 1)
                    self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                else:
                    visual_objects.append(self.visual_objects[filename])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                # self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                link_ids.append(link_id)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=[dimensions[0], dimensions[1], dimensions[2]])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)

            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
            poses_rot.append(np.ascontiguousarray(quat2rotmat([orn[-1], orn[0], orn[1], orn[2]])))
            poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        self.renderer.add_instance_group(object_ids=visual_objects,
                                         link_ids=link_ids,
                                         pybullet_uuid=ids,
                                         poses_rot=poses_rot,
                                         poses_trans=poses_trans,
                                         dynamic=True,
                                         robot=None)

        return ids

    def step(self):
        p.stepSimulation()
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.update_position(instance)
        if self.mode == 'gui' and not self.viewer is None:
            self.viewer.update()

    @staticmethod
    def update_position(instance):
        if isinstance(instance, Instance):
            pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)
            instance.set_position(pos)
            instance.set_rotation([orn[-1], orn[0], orn[1], orn[2]])
        elif isinstance(instance, InstanceGroup):
            poses_rot = []
            poses_trans = []

            for link_id in instance.link_ids:
                if link_id == -1:
                    pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)
                else:
                    _, _, _, _, pos, orn = p.getLinkState(instance.pybullet_uuid, link_id)
                poses_rot.append(
                    np.ascontiguousarray(quat2rotmat([orn[-1], orn[0], orn[1], orn[2]])))
                poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))
                #print(instance.pybullet_uuid, link_id, pos, orn)

            instance.poses_rot = poses_rot
            instance.poses_trans = poses_trans

    def isconnected(self):
        return p.getConnectionInfo(self.cid)['isConnected']

    def disconnect(self):
        if self.isconnected():
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
