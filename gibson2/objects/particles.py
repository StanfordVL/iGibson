from collections import deque

import numpy as np
import pybullet as p
from scipy.stats import truncnorm
from gibson2 import object_states
from gibson2.objects.object_base import Object

_STASH_POSITION = [0, 0, -100]

_DIRT_MAX_SAMPLING_ATTEMPTS = 10
_DIRT_SAMPLING_BOTTOM_SIDE_PROBABILITY = 0.1
_DIRT_SAMPLING_AXIS_PROBABILITIES = [0.25, 0.25, 0.5]
_DIRT_SAMPLING_BIMODAL_MEAN_FRACTION = 0.9
_DIRT_SAMPLING_BIMODAL_STDEV_FRACTION = 0.2
_DIRT_RAY_CASTING_AABB_OFFSET = 0.1
_DIRT_RAY_CASTING_PARALLEL_RAY_SOURCE_OFFSET = 0.05
_DIRT_RAY_CASTING_PARALLEL_HIT_NORMAL_ANGLE_TOLERANCE = 0.2
_DIRT_SAMPLING_MAX_PERPENDICULAR_VECTOR_ATTEMPTS = 3
_DIRT_MAX_ANGLE_WITH_Z_AXIS = 3 * np.pi / 4

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

    def force_wakeup(self, body_id=None):
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


class Dirt(AttachedParticleSystem):
    def __init__(self, parent_obj, color, **kwargs):
        super(Dirt, self).__init__(
            parent_obj,
            dim=0.01,
            visual_only=True,
            mass=0,
            color=color,
            **kwargs
        )

    @staticmethod
    def sample_origin_positions(mins, maxes, count):
        """
        Sample ray casting origin positions with a given distribution.

        The way the sampling works is that for each particle, it will sample two coordinates uniformly and one
        using a bimodal truncated normal distribution. This way, the particles will mostly be close to the faces
        of the AABB (given a correctly parameterized bimodal truncated normal) and will be spread across each face,
        but there will still be a small number of particles spawned inside the object if it has an interior.

        :param mins: The minimum coordinate along each axis.
        :param maxes: The maximum coordinate along each axis.
        :param count: Number of origins to sample.
        :return:
        """
        assert len(mins.shape) == 1
        assert mins.shape == maxes.shape

        # First sample the bimodal normals.
        bimodals = []
        for axis in range(mins.shape[0]):
            bottom = (0 - _DIRT_SAMPLING_BIMODAL_MEAN_FRACTION) / _DIRT_SAMPLING_BIMODAL_STDEV_FRACTION
            top = (1 - _DIRT_SAMPLING_BIMODAL_MEAN_FRACTION) / _DIRT_SAMPLING_BIMODAL_STDEV_FRACTION
            results_1 = truncnorm.rvs(bottom, top, loc=_DIRT_SAMPLING_BIMODAL_MEAN_FRACTION,
                                      scale=_DIRT_SAMPLING_BIMODAL_STDEV_FRACTION, size=count)

            # We want the bottom side to show up much less often.
            side_selection_p = (
                [0.5, 0.5] if axis != 2 else
                [1 - _DIRT_SAMPLING_BOTTOM_SIDE_PROBABILITY, _DIRT_SAMPLING_BOTTOM_SIDE_PROBABILITY])

            # Choose which side of the axis to sample from.
            results_which_side = np.random.choice([True, False], size=count, p=side_selection_p)
            side_selection_applied_results = np.where(results_which_side, results_1, 1 - results_1)
            scaled_results = mins[axis] + (maxes[axis] - mins[axis]) * side_selection_applied_results
            bimodals.append(scaled_results)

        # Transpose so that we have (sample, axis) indexing.
        bimodals = np.array(bimodals).T

        # Now sample the uniform axes.
        uniforms = np.random.uniform(mins, maxes, (count, mins.shape[0]))

        # Finally merge them so that only one axis, randomly chosen with weights, is from the bimodal.
        dim_for_bimodal = np.random.choice([0, 1, 2], size=count, p=_DIRT_SAMPLING_AXIS_PROBABILITIES)
        uniforms[np.arange(count), dim_for_bimodal] = bimodals[np.arange(count), dim_for_bimodal]

        return uniforms

    def randomize(self, obj):
        aabb = obj.states[object_states.AABB].get_value()
        aabb_min = np.array(aabb[0])
        aabb_max = np.array(aabb[1])

        sampling_aabb_min = aabb_min - _DIRT_RAY_CASTING_AABB_OFFSET
        sampling_aabb_max = aabb_max + _DIRT_RAY_CASTING_AABB_OFFSET

        body_id = obj.get_body_id()
        pos = obj.get_position()

        for i in range(self.get_num_stashed()):
            # Sample the starting positions in advance.
            samples = self.sample_origin_positions(sampling_aabb_min, sampling_aabb_max, _DIRT_MAX_SAMPLING_ATTEMPTS)

            # Try each sampled position in the AABB.
            for start_pos in samples:
                # Extend vector across center of mass, to the AABB's face.
                towards_com = pos - start_pos

                # Extend vector until it intersects one of the AABB's faces.
                point_to_min = aabb_min - start_pos
                point_to_max = aabb_max - start_pos
                closer_point_on_each_axis = np.where(towards_com < 0, point_to_min, point_to_max)
                multiple_to_face_on_each_axis = closer_point_on_each_axis / towards_com
                multiple_to_face = np.min(multiple_to_face_on_each_axis)
                point_on_face = start_pos + towards_com * multiple_to_face

                # We will cast multiple parallel rays to check that we have a nice flat area.
                # Find two perpendicular vectors to the towards_com vector.
                for _ in range(_DIRT_SAMPLING_MAX_PERPENDICULAR_VECTOR_ATTEMPTS):
                    seed = np.random.rand(3)
                    v1 = np.cross(towards_com, seed)

                    # If the seed is parallel to the towards_com vector, retry.
                    if np.all(v1 == 0):
                        continue

                    # Find 2nd vector orthogonal to v1.
                    v2 = np.cross(towards_com, v1)

                    # Normalize both vectors.
                    v1 /= np.linalg.norm(v1)
                    v2 /= np.linalg.norm(v2)

                    break
                else:
                    # We couldn't find perpendicular vectors. Better luck next time.
                    continue

                # Use the perpendicular vectors to cast some parallel rays.
                ray_offsets = np.array([(-1, -1), (-1, 1), (1, 1), (1, -1)]) * _DIRT_RAY_CASTING_PARALLEL_RAY_SOURCE_OFFSET
                sources = [start_pos] + [start_pos + offsets[0] * v1 + offsets[1] * v2 for offsets in ray_offsets]
                destinations = [point_on_face] + [point_on_face + offsets[0] * v1 + offsets[1] * v2 for offsets in ray_offsets]

                # Time to cast the rays.
                res = p.rayTestBatch(rayFromPositions=sources, rayToPositions=destinations)

                # Check that the center ray has hit our object.
                if res[0][0] != body_id:
                    continue

                # Get the candidate dirt position.
                hit_pos = np.array(res[0][3])

                # Reject anything facing more than 45deg downwards. These are hard to clean.
                hit_normal = np.array(res[0][4])
                hit_normal /= np.linalg.norm(hit_normal)
                hit_angle_with_z = np.arccos(np.dot(hit_normal, np.array([0, 0, 1])))
                if hit_angle_with_z > _DIRT_MAX_ANGLE_WITH_Z_AXIS:
                    continue

                # Check that none of the parallel rays' hit normal differs from center ray by more than threshold.
                all_rays_hit_with_similar_normal = all(
                    ray_res[0] == body_id and
                    (np.arccos(np.dot(hit_normal, ray_res[4] / np.linalg.norm(ray_res[4]))) <
                        _DIRT_RAY_CASTING_PARALLEL_HIT_NORMAL_ANGLE_TOLERANCE)
                    for ray_res in res[1:])
                if not all_rays_hit_with_similar_normal:
                    continue

                # We've found a nice attachment point. Let's go.
                self.unstash_particle(hit_pos, [0, 0, 0, 1])
                break

class Dust(Dirt):
    def __init__(self, parent_obj, **kwargs):
        super(Dust, self).__init__(parent_obj, (0, 0, 0, 1), **kwargs)

class Stain(Dirt):
    def __init__(self, parent_obj, **kwargs):
        super(Stain, self).__init__(parent_obj, (0.4, 0, 0, 1), **kwargs)
