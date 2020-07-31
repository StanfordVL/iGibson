from copy import deepcopy
import numpy as np

import pybullet as p
from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, Object
import gibson2.external.pybullet_tools.utils as PBU


class Faucet(Object):
    def __init__(
            self,
            num_beads=20,
            dispense_freq=1,
            dispense_height=0.3,
            base_color=(0.75, 0.75, 0.75, 1),
            beads_color=(0, 0, 1, 1),
            beads_size=0.015
    ):
        self._dispense_freq = dispense_freq
        self._dispense_height = dispense_height
        self._beads = []
        self._next_bead_index = 0
        self._n_step_since = 0
        self._base_color = base_color
        self._beads_color = beads_color
        self._beads_size = beads_size
        self._num_beads = num_beads
        super(Faucet, self).__init__()

    @property
    def beads(self):
        return deepcopy(self._beads)

    def load(self):
        self.body_id = PBU.create_box(0.15, 0.15, 0.01, mass=100, color=self._base_color)
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
            return
        bid = self._beads[self._next_bead_index]
        prev_pose = PBU.get_pose(bid)
        PBU.set_pose(bid, (self.get_position() + np.array([0, 0, self._dispense_height]), PBU.unit_quat()))
        for oid in [o.body_id for o in task_objs] + self._beads:
            if oid != bid and PBU.body_collision(oid, bid):
                PBU.set_pose(bid, prev_pose)
                return
        p.changeDynamics(bid, -1, mass=0.3)
        self._next_bead_index += 1

    def step(self, task_objs):
        should_dispense = False
        for o in task_objs:
            if o.body_id == self.body_id:
                continue
            center_place = PBU.is_center_stable(o.body_id, self.body_id, above_epsilon=0.01, below_epsilon=0.02)
            in_contact = PBU.body_collision(self.body_id, o.body_id)
            should_dispense = should_dispense or (center_place and in_contact)
        if should_dispense and self._n_step_since >= self._dispense_freq:
            self._try_dispense(task_objs)
            self._n_step_since = 0
        else:
            self._n_step_since += 1


class Platform(Object):
    def __init__(self, color=(0, 1, 0, 1), size=(0.15, 0.15, 0.01)):
        super(Platform, self).__init__()
        self._color = color
        self._size = size

    def load(self):
        self.body_id = PBU.create_box(*self._size, mass=100, color=self._color)
        self.loaded = True
