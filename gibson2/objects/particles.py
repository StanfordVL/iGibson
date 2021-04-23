from collections import deque

import numpy as np
import pybullet as p
from gibson2.objects.object_base import Object
from gibson2.utils import sampling_utils

_STASH_POSITION = [0, 0, -100]

# This parameters are used when sampling dirt particles.
# See gibson2/utils/sampling_utils.py for how they are used.
_DIRT_SAMPLING_BOTTOM_SIDE_PROBABILITY = 0.1
_DIRT_SAMPLING_AXIS_PROBABILITIES = [0.25, 0.25, 0.5]
_DIRT_SAMPLING_BIMODAL_MEAN_FRACTION = 0.9
_DIRT_SAMPLING_BIMODAL_STDEV_FRACTION = 0.2
_DIRT_RAY_CASTING_PARALLEL_RAY_SOURCE_OFFSET = 0.05

_WATER_SOURCE_PERIOD = 0.3  # new water every this many seconds.


class Particle(Object):
    """
    A particle object, used to simulate water stream and dust/stain
    """

    def __init__(self, pos=(0, 0, 0), dim=0.1, visual_only=False, mass=0.1, color=(1, 1, 1, 1), base_shape="sphere"):
        super(Particle, self).__init__()
        self.base_pos = pos
        self.dimension = [dim, dim, dim]
        self.visual_only = visual_only
        self.mass = mass
        self.color = color
        self.base_shape = base_shape

    def _load(self):
        """
        Load the object into pybullet
        """
        base_orientation = [0, 0, 0, 1]

        if self.base_shape == "box":
            colBoxId = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=self.dimension)
            visualShapeId = p.createVisualShape(
                p.GEOM_BOX, halfExtents=self.dimension, rgbaColor=self.color)
        elif self.base_shape == 'sphere':
            colBoxId = p.createCollisionShape(
                p.GEOM_SPHERE, radius=self.dimension[0])
            visualShapeId = p.createVisualShape(
                p.GEOM_SPHERE, radius=self.dimension[0], rgbaColor=self.color)

        if self.visual_only:
            body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visualShapeId)
        else:
            body_id = p.createMultiBody(baseMass=self.mass,
                                        baseCollisionShapeIndex=colBoxId,
                                        baseVisualShapeIndex=visualShapeId)

        p.resetBasePositionAndOrientation(
            body_id, np.array(self.base_pos), base_orientation)

        self.force_sleep(body_id)

        return body_id

    def force_sleep(self, body_id=None):
        if body_id is None:
            body_id = self.body_id

        activationState = p.ACTIVATION_STATE_ENABLE_SLEEPING + p.ACTIVATION_STATE_SLEEP + p.ACTIVATION_STATE_DISABLE_WAKEUP
        p.changeDynamics(body_id, -1, activationState=activationState)

    def force_wakeup(self):
        activationState = p.ACTIVATION_STATE_ENABLE_SLEEPING + p.ACTIVATION_STATE_WAKE_UP
        p.changeDynamics(self.body_id, -1, activationState=activationState)


class ParticleSystem(object):
    def __init__(self, num=20, visual_only=False, **kwargs):
        self._active_particles = []
        self._stashed_particles = deque()

        for i in range(num):
            self._stashed_particles.append(Particle(pos=_STASH_POSITION, visual_only=visual_only, **kwargs))

        self.visual_only = visual_only

    def update(self, simulator):
        pass

    def get_num(self):
        return len(self._stashed_particles) + len(self._active_particles)

    def get_num_stashed(self):
        return len(self._stashed_particles)

    def get_num_active(self):
        return len(self._active_particles)

    def get_stashed_particles(self):
        return list(self._stashed_particles)

    def get_active_particles(self):
        return list(self._active_particles)

    def get_particles(self):
        return self.get_active_particles() + self.get_stashed_particles()

    def stash_particle(self, particle):
        assert particle in self._active_particles
        self._active_particles.remove(particle)
        self._stashed_particles.append(particle)

        particle.set_position(_STASH_POSITION)
        particle.force_sleep()

    def unstash_particle(self, position, orientation):
        # This assumes that the stashed particle has been moved to the appropriate position already.
        particle = self._stashed_particles.popleft()
        particle.set_position_orientation(position, orientation)
        particle.force_wakeup()

        self._active_particles.append(particle)

        return particle


class AttachedParticleSystem(ParticleSystem):
    def __init__(self, parent_obj, **kwargs):
        super(AttachedParticleSystem, self).__init__(**kwargs)

        self._parent_obj = parent_obj
        self._attachment_offsets = {}  # in the format of {particle: offset}

    def unstash_particle(self, position, orientation):
        particle = super(AttachedParticleSystem, self).unstash_particle(position, orientation)

        # Compute the offset for this particle.
        base_pos, base_orn = p.invertTransform(self._parent_obj.get_position(), self._parent_obj.get_orientation())
        offsets = p.multiplyTransforms(base_pos, base_orn, position, orientation)
        self._attachment_offsets[particle] = offsets

        return particle

    def stash_particle(self, particle):
        super(AttachedParticleSystem, self).stash_particle(particle)
        del self._attachment_offsets[particle]

    def update(self, simulator):
        super(AttachedParticleSystem, self).update(simulator)

        # Move every particle to their known parent object offsets.
        # TODO: Find the surface link so that we can attach to the correct link rather than main body.
        base_pos, base_orn = self._parent_obj.get_position(), self._parent_obj.get_orientation()
        for particle in self.get_active_particles():
            pos_offset, orn_offset = self._attachment_offsets[particle]
            position, orientation = p.multiplyTransforms(base_pos, base_orn, pos_offset, orn_offset)
            particle.set_position_orientation(position, orientation)


class WaterStream(ParticleSystem):
    def __init__(self, water_source_pos, **kwargs):
        super(WaterStream, self).__init__(
            dim=0.01,
            visual_only=False,
            mass=0.1,
            color=(0, 0, 1, 1),
            **kwargs
        )

        self.steps_since_last_drop_step = float('inf')
        self.water_source_pos = water_source_pos
        self.on = False

    def set_running(self, on):
        self.on = on

    def update(self, simulator):
        # If the stream is off, return.
        if not self.on:
            return

        # If we don't have any stashed particles, return.
        if self.get_num_stashed() == 0:
            return

        # If enough time hasn't passed since last drop, return.
        if self.steps_since_last_drop_step < _WATER_SOURCE_PERIOD / simulator.render_timestep:
            self.steps_since_last_drop_step += 1
            return

        # Otherwise, create & drop the water.
        self.unstash_particle(self.water_source_pos, [0, 0, 0, 1])
        self.steps_since_last_drop_step = 0


class _Dirt(AttachedParticleSystem):
    """
    This class represents common logic between particle-based dirtyness states like
    dusty and stained. It should not be directly instantiated - use subclasses instead.
    """
    def __init__(self, parent_obj, color, **kwargs):
        super(_Dirt, self).__init__(
            parent_obj,
            dim=0.01,
            visual_only=True,
            mass=0,
            color=color,
            **kwargs
        )

    def randomize(self, obj):
        # Sample points using the raycasting sampler.
        results = sampling_utils.sample_points_on_object(
            obj, self.get_num_stashed(), _DIRT_RAY_CASTING_PARALLEL_RAY_SOURCE_OFFSET,
            _DIRT_SAMPLING_BIMODAL_MEAN_FRACTION, _DIRT_SAMPLING_BIMODAL_STDEV_FRACTION,
            _DIRT_SAMPLING_AXIS_PROBABILITIES, _DIRT_SAMPLING_BOTTOM_SIDE_PROBABILITY, refuse_downwards=True)

        # Use the sampled points to set the dirt positions.
        for position, normal, quaternion, reasons in results:
            if position is not None:
                self.unstash_particle(position, [0, 0, 0, 1])


class Dust(_Dirt):
    def __init__(self, parent_obj, **kwargs):
        super(Dust, self).__init__(parent_obj, (0, 0, 0, 1), **kwargs)


class Stain(_Dirt):
    def __init__(self, parent_obj, **kwargs):
        super(Stain, self).__init__(parent_obj, (0.4, 0, 0, 1), **kwargs)
