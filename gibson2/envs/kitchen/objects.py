from copy import deepcopy
import numpy as np

import gibson2.external.pybullet_tools.transformations as T
import pybullet as p
from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, Object, VisualMarker
import gibson2.external.pybullet_tools.utils as PBU
import gibson2.envs.kitchen.env_utils as EU


class Faucet(Object):
    def __init__(
            self,
            num_beads=20,
            dispense_freq=1,
            dispense_height=0.3,
            base_color=(0.75, 0.75, 0.75, 1),
            beads_color=(0, 0, 1, 1),
            base_size=(0.15, 0.15, 0.01),
            beads_size=0.015,
            exclude_ids=(),
    ):
        self._dispense_freq = dispense_freq
        self._beads = []
        self._next_bead_index = 0
        self._n_step_since = 0
        self._base_color = base_color
        self._base_size = base_size
        self._beads_color = beads_color
        self._beads_size = beads_size
        self._num_beads = num_beads
        self._dispense_position = np.array([0, 0, dispense_height])
        self._exclude_ids = exclude_ids
        super(Faucet, self).__init__()

    @property
    def beads(self):
        return deepcopy(self._beads)

    def load(self):
        self.body_id = PBU.create_box(*self._base_size, mass=100, color=self._base_color)
        self._beads = [PBU.create_sphere(
            self._beads_size, mass=PBU.STATIC_MASS, color=self._beads_color
        ) for _ in range(self._num_beads)]
        self.loaded = True

    def reset(self):
        self._next_bead_index = 0
        for i, b in enumerate(self._beads):
            p.resetBasePositionAndOrientation(b, self.get_position() + np.array([0, 0, 10 + b * 0.1]), PBU.unit_quat())
            p.changeDynamics(b, -1, mass=PBU.STATIC_MASS)

        self._n_step_since = 0

    def _try_dispense(self, task_objs):
        if self._next_bead_index == self._num_beads:
            return False
        bid = self._beads[self._next_bead_index]
        prev_pose = PBU.get_pose(bid)
        PBU.set_pose(bid, (self.get_position() + self._dispense_position, PBU.unit_quat()))
        for oid in [o.body_id for o in task_objs] + self._beads:
            if oid != bid and PBU.body_collision(oid, bid):
                PBU.set_pose(bid, prev_pose)
                return False
        p.changeDynamics(bid, -1, mass=0.3)
        self._next_bead_index += 1
        return True

    def step(self, task_objs, gripper=None):
        should_dispense = False
        for o in task_objs:
            if o.body_id == self.body_id or o.body_id in self._exclude_ids:
                continue
            center_place = PBU.is_center_stable(o.body_id, self.body_id, above_epsilon=0.01, below_epsilon=0.02)
            in_contact = PBU.body_collision(self.body_id, o.body_id)
            should_dispense = should_dispense or (center_place and in_contact)
        if should_dispense and self._n_step_since >= self._dispense_freq:
            self._try_dispense(task_objs)
            self._n_step_since = 0
        else:
            self._n_step_since += 1


class CoffeeGrinder(Object):
    def __init__(self, mass=10.0):
        super(CoffeeGrinder, self).__init__()
        self._mass = mass

    def load(self):
        # collision_id1, visual_id1 = PBU.create_shape(
        #     PBU.get_cylinder_geometry(0.1, 0.01), color=(0.5, 0.5, 0.5, 1))
        collision_id1, visual_id1 = PBU.create_shape(
            PBU.get_box_geometry(0.2, 0.2, 0.01), color=(0.5, 0.5, 0.5, 1))
        collision_id2, visual_id2 = PBU.create_shape(
            PBU.get_cylinder_geometry(0.015, 0.3), color=(0.3, 0.3, 0.3, 1))
        collision_id3, visual_id3 = PBU.create_shape(
            PBU.get_box_geometry(0.05, 0.02, 0.03), color=(0.5, 0.5, 0.5, 1))
        collision_id4, visual_id4 = PBU.create_shape(
            PBU.get_cylinder_geometry(0.05, 0.12), color=(0.3, 0.3, 0.3, 1))
        collision_id5, visual_id5 = PBU.create_shape(
            PBU.get_cylinder_geometry(0.05, 0.04), color=(0.2, 0.2, 0.2, 0.7))

        link_masses = [self._mass * 0.7, self._mass * 0.1, self._mass * 0.1, self._mass * 0.1]

        self.body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_id1,
            baseVisualShapeIndex=visual_id1,
            basePosition=PBU.unit_point(),
            baseOrientation=PBU.unit_quat(),
            baseInertialFramePosition=PBU.unit_point(),
            baseInertialFrameOrientation=PBU.unit_quat(),
            linkMasses=link_masses,
            linkCollisionShapeIndices=[collision_id2, collision_id3, collision_id4, collision_id5],
            linkVisualShapeIndices=[visual_id2, visual_id3, visual_id4, visual_id5],
            linkPositions=[(0.09, 0, 0.15), (0.07, 0, 0.27), (0, 0, 0.27), (0, 0, 0.35)],
            linkOrientations=[T.quaternion_from_euler(0, 0, np.pi / 2), PBU.unit_quat(), PBU.unit_quat(), PBU.unit_quat()],
            linkInertialFramePositions=[PBU.unit_point(), PBU.unit_point(), PBU.unit_point(), PBU.unit_point()],
            linkInertialFrameOrientations=[PBU.unit_quat(), PBU.unit_quat(), PBU.unit_quat(), PBU.unit_quat()],
            linkParentIndices=[0, 0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        )
        self.loaded = True


class TeaDispenser(Object):
    def __init__(self, mass=10.0):
        super(TeaDispenser, self).__init__()
        self._mass = mass

    def load(self):
        collision_id1, visual_id1 = PBU.create_shape(
            PBU.get_cylinder_geometry(0.1, 0.01), color=(0.5, 0.5, 0.5, 1))
        collision_id2, visual_id2 = PBU.create_shape(
            PBU.get_cylinder_geometry(0.04, 0.36), color=(0.3, 0.3, 0.3, 1))
        collision_id3, visual_id3 = PBU.create_shape(
            PBU.get_box_geometry(0.08, 0.02, 0.03), color=(0.5, 0.5, 0.5, 1))
        collision_id4, visual_id4 = PBU.create_shape(
            PBU.get_cylinder_geometry(0.02, 0.05), color=(0.7, 0.7, 0.7, 1))

        link_masses = [self._mass * 0.5, self._mass * 0.1, self._mass * 0.1]

        self.body_id = p.createMultiBody(
            baseMass=0.3 * self._mass,
            baseCollisionShapeIndex=collision_id1,
            baseVisualShapeIndex=visual_id1,
            basePosition=PBU.unit_point(),
            baseOrientation=PBU.unit_quat(),
            baseInertialFramePosition=PBU.unit_point(),
            baseInertialFrameOrientation=PBU.unit_quat(),
            linkMasses=link_masses,
            linkCollisionShapeIndices=[collision_id2, collision_id3, collision_id4],
            linkVisualShapeIndices=[visual_id2, visual_id3, visual_id4],
            linkPositions=[(0.14, 0, 0.175), (0.05, 0, 0.27), (0, 0, 0.27)],
            linkOrientations=[T.quaternion_from_euler(0, 0, np.pi / 2), PBU.unit_quat(), PBU.unit_quat()],
            linkInertialFramePositions=[PBU.unit_point(), PBU.unit_point(), PBU.unit_point()],
            linkInertialFrameOrientations=[PBU.unit_quat(), PBU.unit_quat(), PBU.unit_quat()],
            linkParentIndices=[0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        )
        self.loaded = True


class CoffeeMachine(Faucet):
    def __init__(
            self,
            filename,
            beans_set,
            num_beans_trigger=5,
            num_coffee_beads=20,
            dispense_freq=1,
            dispense_position=(0, 0, 0),
            platform_position=(0, 0, 0),
            beads_color=(1, 1, 1, 1),
            beads_size=0.01,
            button_pose=((0, 0, 0), PBU.unit_quat()),
            scale=1.
    ):

        super(CoffeeMachine, self).__init__(
            num_beads=num_coffee_beads,
            dispense_freq=dispense_freq,
            beads_color=beads_color,
            beads_size=beads_size
        )
        self._file_path = filename
        self._beans_set = beans_set
        self._num_beans_trigger = num_beans_trigger
        self._dispense_position = np.array(dispense_position)
        self._platform_position = np.array(platform_position)
        self._scale = scale
        self._button_pose = button_pose
        self._should_dispense = False
        self.platform = None
        self.button = None

    def reset(self):
        super(CoffeeMachine, self).reset()
        self._sync_parts()
        self._should_dispense = False

    def _sync_parts(self):
        p.resetBasePositionAndOrientation(
            self.platform.body_id, self._platform_position + np.array(self.get_position()), PBU.unit_quat())
        PBU.multiply(self._button_pose, self.get_position_orientation())
        p.resetBasePositionAndOrientation(
            self.button.body_id, *PBU.multiply(self.get_position_orientation(), self._button_pose)
        )

    def load(self):
        self.body_id = p.loadURDF(
            self._file_path, globalScaling=self._scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.platform = Box(size=[0.12, 0.12, 0.01], color=(0.8, 0.8, 0.8, 0.1))
        self.platform.load()
        # self.button = VisualMarker(visual_shape=p.GEOM_CYLINDER, radius=0.03, length=0.008, rgba_color=(0, 1, 0, 0))
        self.button = Cylinder(radius=0.03, height=0.008, color=(0, 1, 0, 0), mass=PBU.STATIC_MASS)
        self.button.load()
        self._beads = [PBU.create_sphere(
            self._beads_size, mass=PBU.STATIC_MASS, color=self._beads_color
        ) for _ in range(self._num_beads)]
        self.loaded = True

    def step(self, task_objs, gripper=None):
        beans = EU.objects_center_in_container(self._beans_set, self.body_id)

        # needs to be in the funnel
        beans = [bid for bid in beans if PBU.get_pose(bid)[0][2] > self.get_position()[2] - 0.05]

        # start_machine = PBU.body_collision(gripper.body_id, self.button.body_id)
        start_machine = False
        for o in task_objs:
            if o.body_id == self.platform.body_id:
                continue
            center_place = PBU.is_center_stable(o.body_id, self.platform.body_id, above_epsilon=0.01, below_epsilon=0.02)
            collision = PBU.body_collision(o.body_id, self.platform.body_id)
            start_machine = start_machine or (collision and center_place)

        # start dispensing when button is pressed
        self._should_dispense = self._should_dispense or start_machine
        # stop when no more beans in the machine
        self._should_dispense = self._should_dispense and len(beans) > 0

        if self._should_dispense and self._n_step_since >= self._dispense_freq:
            if self._try_dispense(task_objs):
                p.resetBasePositionAndOrientation(
                    beans[0], np.array([0, 0, 10 + self._next_bead_index * 0.1]), PBU.unit_quat())
                p.changeDynamics(beans[0], -1, mass=PBU.STATIC_MASS)
                self._n_step_since = 0
        else:
            self._n_step_since += 1


class Box(Object):
    def __init__(self, color=(0, 1, 0, 1), size=(0.15, 0.15, 0.01), mass=100):
        super(Box, self).__init__()
        self._color = color
        self._size = size
        self._mass = mass

    def load(self):
        self.body_id = PBU.create_box(*self._size, mass=self._mass, color=self._color)
        self.loaded = True


class Cylinder(Object):
    def __init__(self, color=(0, 1, 0, 1), radius=0.1, height=0.1, mass=100):
        super(Cylinder, self).__init__()
        self._color = color
        self._radius = radius
        self._height = height
        self._mass = mass

    def load(self):
        self.body_id = PBU.create_cylinder(radius=self._radius, height=self._height, mass=self._mass, color=self._color)
        self.loaded = True


class MessyPlate(Box):
    def __init__(self, color=(0, 1, 0, 1), size=(0.15, 0.15, 0.01), mass=100, num_stuff=5, stuff_size=(0.01, 0.01, 0.01)):
        super(MessyPlate, self).__init__(color=color, size=size, mass=mass)
        self.num_stuff = num_stuff
        self.stuff_size = stuff_size
        self._stuff = []

    @property
    def stuff(self):
        return deepcopy(self._stuff)

    def load(self):
        super(MessyPlate, self).load()
        for i in range(self.num_stuff):
            color = np.random.random(4)
            color[3] = 1
            self._stuff.append(PBU.create_box(*self.stuff_size, mass=0.01, color=color))

    def reset(self):
        for bid in self._stuff:
            PBU.sample_placement(top_body=bid, bottom_body=self.body_id)


class Tube(Object):
    def __init__(self, color=(0, 1, 0, 1), size=(0.5, 0.1, 0.1), width=0.01, mass=1.):
        super(Tube, self).__init__()
        self._color = color
        self._size = size
        self._width = width
        self._mass = mass

    def load(self):
        self.loaded = True
        l, w, h = self._size

        bottom_col, bottom_vir = PBU.create_shape(
            PBU.get_box_geometry(l, w, self._width), color=self._color)

        left_col, left_vir = PBU.create_shape(
            PBU.get_box_geometry(l, self._width, h), color=self._color)

        right_col, right_vir = PBU.create_shape(
            PBU.get_box_geometry(l, self._width, h), color=self._color)

        top_col, top_vir = PBU.create_shape(
            PBU.get_box_geometry(l, w, self._width), color=self._color)

        masses = [self._mass / 4] * 3
        col_indices = (left_col, right_col, top_col)
        vir_indices = (left_vir, right_vir, top_vir)
        positions = [
            (0, -w / 2, h / 2),
            (0, w / 2, h / 2),
            (0, 0, h),
        ]
        orns = [PBU.unit_quat()] * 3

        self.body_id = p.createMultiBody(
            baseMass=self._mass / 4,
            baseCollisionShapeIndex=bottom_col,
            baseVisualShapeIndex=bottom_vir,
            basePosition=PBU.unit_point(),
            baseOrientation=PBU.unit_quat(),
            baseInertialFramePosition=PBU.unit_point(),
            baseInertialFrameOrientation=PBU.unit_quat(),
            linkMasses=masses,
            linkCollisionShapeIndices=col_indices,
            linkVisualShapeIndices=vir_indices,
            linkPositions=positions,
            linkOrientations=orns,
            linkInertialFramePositions=[PBU.unit_point()] * 3,
            linkInertialFrameOrientations=orns,
            linkParentIndices=[0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[PBU.unit_point()] * 3
        )


class Hook(Object):
    def __init__(self, width, length1, length2, color=(0, 1, 0, 1)):
        super(Hook, self).__init__()
        self._width = width
        self._length1 = length1
        self._length2 = length2
        self._color = color

    def load(self):
        self.loaded = True

        collision_id1, visual_id1 = PBU.create_shape(
            PBU.get_box_geometry(self._length1, self._width, self._width), color=self._color)
        collision_id2, visual_id2 = PBU.create_shape(
            PBU.get_box_geometry(self._length2, self._width, self._width), color=self._color)
        self.body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_id1,
            baseVisualShapeIndex=visual_id1,
            basePosition=PBU.unit_point(),
            baseOrientation=PBU.unit_quat(),
            baseInertialFramePosition=PBU.unit_point(),
            baseInertialFrameOrientation=PBU.unit_quat(),
            linkMasses=(0.5,),
            linkCollisionShapeIndices=[collision_id2],
            linkVisualShapeIndices=[visual_id2],
            linkPositions=[(-self._length1 / 2 + self._width / 2, -self._length2 / 2 + self._width / 2, 0)],
            linkOrientations=[T.quaternion_from_euler(0, 0, np.pi / 2)],
            linkInertialFramePositions=[(0, 0, 0)],
            linkInertialFrameOrientations=[PBU.unit_quat()],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]]
        )