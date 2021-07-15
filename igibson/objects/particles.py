import os
from collections import deque

import numpy as np
import pybullet as p

import igibson
from igibson.external.pybullet_tools import utils
from igibson.external.pybullet_tools.utils import get_aabb_extent, get_link_name, link_from_name
from igibson.objects.object_base import Object
from igibson.utils import sampling_utils
from igibson.utils.constants import PyBulletSleepState, SemanticClass

_STASH_POSITION = [0, 0, -100]


class Particle(Object):
    """
    A particle object, used to simulate water stream and dust/stain
    """

    def __init__(
        self,
        size,
        pos,
        visual_only=False,
        mass=0.0,
        color=(1, 1, 1, 1),
        base_shape="sphere",
        mesh_filename=None,
        mesh_bounding_box=None,
    ):
        """
        Create a particle.

        :param size: 3-dimensional bounding box size to fit particle in. For sphere, the smallest dimension is used.
        :param pos: Initial position of particle.
        :param visual_only: True if the particle should be visual only, False if it should have collisions too.
        :param mass: Particle mass.
        :param color: RGBA particle color.
        :param base_shape: One of "cube", "sphere", "mesh". If mesh, mesh_filename also required.
        :param mesh_filename: Filename of obj file to load mesh from, if base_shape is "mesh".
        :param mesh_bounding_box: bounding box of the mesh when scale=1. Needed for scale computation.
        """
        super(Particle, self).__init__()
        self.base_pos = pos
        self.size = size
        self.visual_only = visual_only
        self.mass = mass
        self.color = color
        self.base_shape = base_shape
        self.bounding_box = np.array(self.size)
        assert len(self.size) == 3

        if self.base_shape == "mesh":
            assert mesh_filename is not None and mesh_bounding_box is not None
            self.mesh_filename = mesh_filename
            self.mesh_scale = np.array(size) / np.array(mesh_bounding_box)

    def _load(self):
        """
        Load the object into pybullet
        """
        base_orientation = [0, 0, 0, 1]

        if self.base_shape == "box":
            colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.bounding_box / 2.0)
            visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=self.bounding_box / 2.0, rgbaColor=self.color)
        elif self.base_shape == "sphere":
            colBoxId = p.createCollisionShape(p.GEOM_SPHERE, radius=self.bounding_box[0] / 2.0)
            visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=self.bounding_box[0] / 2.0, rgbaColor=self.color)
        elif self.base_shape == "mesh":
            colBoxId = p.createCollisionShape(p.GEOM_MESH, fileName=self.mesh_filename, meshScale=self.mesh_scale)
            visualShapeId = p.createVisualShape(p.GEOM_MESH, fileName=self.mesh_filename, meshScale=self.mesh_scale)
        else:
            raise ValueError("Unsupported particle base shape.")

        if self.visual_only:
            body_id = p.createMultiBody(
                baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId, flags=p.URDF_ENABLE_SLEEPING
            )
        else:
            body_id = p.createMultiBody(
                baseMass=self.mass,
                baseCollisionShapeIndex=colBoxId,
                baseVisualShapeIndex=visualShapeId,
                flags=p.URDF_ENABLE_SLEEPING,
            )

        p.resetBasePositionAndOrientation(body_id, np.array(self.base_pos), base_orientation)

        self.force_sleep(body_id)

        return body_id

    def force_sleep(self, body_id=None):
        if body_id is None:
            body_id = self.body_id

        activationState = p.ACTIVATION_STATE_SLEEP + p.ACTIVATION_STATE_DISABLE_WAKEUP
        p.changeDynamics(body_id, -1, activationState=activationState)

    def force_wakeup(self):
        activationState = p.ACTIVATION_STATE_WAKE_UP
        p.changeDynamics(self.body_id, -1, activationState=activationState)


class ParticleSystem(object):
    def __init__(
        self,
        num,
        size,
        color=(1, 1, 1, 1),
        class_id=SemanticClass.USER_ADDED_OBJS,
        use_pbr=False,
        use_pbr_mapping=False,
        shadow_caster=True,
        **kwargs
    ):
        size = np.array(size)
        if size.ndim == 2:
            assert size.shape[0] == num

        color = np.array(color)
        if color.ndim == 2:
            assert color.shape[0] == num

        self._all_particles = []
        self._active_particles = []
        self._stashed_particles = deque()
        self._particles_activated_at_any_time = set()

        self._simulator = None
        self._import_params = {
            "class_id": class_id,
            "use_pbr": use_pbr,
            "use_pbr_mapping": use_pbr_mapping,
            "shadow_caster": shadow_caster,
        }

        for i in range(num):
            # If different sizes / colors provided for each instance, pick the correct one for this instance.
            this_size = size if size.ndim == 1 else size[i]
            this_color = color if color.ndim == 1 else color[i]

            particle = Particle(this_size, _STASH_POSITION, color=this_color, **kwargs)
            self._all_particles.append(particle)
            self._stashed_particles.append(particle)

    def dump(self):
        return [
            particle.get_position_orientation() if particle in self.get_active_particles() else None
            for particle in self.get_particles()
        ]

    def reset_to_dump(self, dump):
        # First, stash all particles
        self.reset_stash()

        for i, particle_pose in enumerate(dump):
            # particle_data will be None for stashed particles
            if particle_pose is not None:
                particle = self.get_particles()[i]
                self.unstash_particle(particle_pose[0], particle_pose[1], particle)

    def initialize(self, simulator):
        # Keep a handle to the simulator for lazy loads later.
        self._simulator = simulator

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
        return self._all_particles

    def stash_particle(self, particle):
        assert particle in self._active_particles
        self._active_particles.remove(particle)
        self._stashed_particles.append(particle)

        particle.set_position(_STASH_POSITION)
        if particle.visual_only:
            # Stain and Dust need to be woken up before stashing because if
            # they are asleep, their poses will not be updated in the renderer
            particle.force_wakeup()
        else:
            # Water (awake when stash_particle is called) needs to be
            # put to sleep because they would collide in _STASH_POSITION
            # It's okay to call force_sleep() because the sleep state will only
            # be reflected after p.stepSimulation() is called. Thus, the
            # renderer should still update its pose in the curren timestep
            particle.force_sleep()

    def _load_particle(self, particle):
        body_id = self._simulator.import_object(particle, **self._import_params)
        # Put loaded particles at the stash position initially.
        particle.set_position(_STASH_POSITION)
        return body_id

    def unstash_particle(self, position, orientation, particle=None):
        # If the user wants a particular particle, give it to them. Otherwise, unstash one.
        if particle is not None:
            self._stashed_particles.remove(particle)
        else:
            particle = self._stashed_particles.popleft()

        # Lazy loading of the particle now if not already loaded
        if not particle.loaded:
            self._load_particle(particle)

        particle.set_position_orientation(position, orientation)
        particle.force_wakeup()

        self._active_particles.append(particle)
        self._particles_activated_at_any_time.add(particle)

        return particle

    def reset_stash(self):
        """Stash all particles and re-order the stash in the all_particles order for determinism."""
        for particle in self.get_active_particles():
            self.stash_particle(particle)

        self._stashed_particles.clear()
        self._stashed_particles.extend(self._all_particles)

    def get_num_particles_activated_at_any_time(self):
        """Get the number of unique particles that were active at some point in history."""
        return len(self._particles_activated_at_any_time)

    def reset_particles_activated_at_any_time(self):
        self._particles_activated_at_any_time = set()


class AttachedParticleSystem(ParticleSystem):
    def __init__(self, parent_obj, initial_dump=None, **kwargs):
        super(AttachedParticleSystem, self).__init__(**kwargs)

        self.parent_obj = parent_obj
        self._attachment_offsets = {}  # in the format of {particle: offset}
        self.initial_dump = initial_dump

    def reset_to_dump(self, dump):
        # Assert that the dump is compatible
        assert len(dump) == self.get_num()

        # First, stash all particles
        self.reset_stash()

        for i, particle_data in enumerate(dump):
            # particle_data will be None for stashed particles
            if particle_data is not None:
                particle_attached_link_name, particle_pos, particle_orn = particle_data
                # If particle_attached_link_id cannot be found, it’s because it has been merged
                # (p.URDF_MERGE_FIXED_LINKS). Since it’s a fixed link anyways and the absolute pose (not link-relative
                # pose) of the particle is dumped, we can safely assign this particle to the base link.
                particle_attached_link_id = -1
                if particle_attached_link_name is not None:
                    try:
                        particle_attached_link_id = link_from_name(
                            self.parent_obj.get_body_id(), particle_attached_link_name
                        )
                    except ValueError:
                        pass

                particle = self.get_particles()[i]
                self.unstash_particle(particle_pos, particle_orn, link_id=particle_attached_link_id, particle=particle)

    def initialize(self, simulator):
        super(AttachedParticleSystem, self).initialize(simulator)

        # Unstash particles in dump.
        if self.initial_dump:
            self.reset_to_dump(self.initial_dump)
            del self.initial_dump

    def unstash_particle(self, position, orientation, link_id=-1, **kwargs):
        particle = super(AttachedParticleSystem, self).unstash_particle(position, orientation, **kwargs)

        # Compute the offset for this particle.
        if link_id == -1:
            attachment_source_pos = self.parent_obj.get_position()
            attachment_source_orn = self.parent_obj.get_orientation()
        else:
            link_state = utils.get_link_state(self.parent_obj.get_body_id(), link_id)
            attachment_source_pos = link_state.linkWorldPosition
            attachment_source_orn = link_state.linkWorldOrientation

        base_pos, base_orn = p.invertTransform(attachment_source_pos, attachment_source_orn)
        offsets = p.multiplyTransforms(base_pos, base_orn, position, orientation)
        self._attachment_offsets[particle] = (link_id, offsets)

        return particle

    def stash_particle(self, particle):
        super(AttachedParticleSystem, self).stash_particle(particle)
        del self._attachment_offsets[particle]

    def update(self, simulator):
        super(AttachedParticleSystem, self).update(simulator)

        # Move every particle to their known parent object offsets.
        for particle in self.get_active_particles():
            link_id, (pos_offset, orn_offset) = self._attachment_offsets[particle]

            dynamics_info = p.getDynamicsInfo(self.parent_obj.get_body_id(), link_id)

            if len(dynamics_info) == 13:
                activation_state = dynamics_info[12]
            else:
                activation_state = PyBulletSleepState.AWAKE

            if activation_state != PyBulletSleepState.AWAKE:
                # If parent object is in sleep, don't update particle poses
                continue

            if link_id == -1:
                attachment_source_pos = self.parent_obj.get_position()
                attachment_source_orn = self.parent_obj.get_orientation()
            else:
                link_state = utils.get_link_state(self.parent_obj.get_body_id(), link_id)
                attachment_source_pos = link_state.linkWorldPosition
                attachment_source_orn = link_state.linkWorldOrientation

            position, orientation = p.multiplyTransforms(
                attachment_source_pos, attachment_source_orn, pos_offset, orn_offset
            )
            particle.set_position_orientation(position, orientation)
            particle.force_wakeup()

    def dump(self):
        data = []
        for particle in self.get_particles():
            if particle in self.get_stashed_particles():
                data.append(None)
            else:
                link_id, (pos_offset, orn_offset) = self._attachment_offsets[particle]

                if link_id == -1:
                    link_name = None
                    attachment_source_pos = self.parent_obj.get_position()
                    attachment_source_orn = self.parent_obj.get_orientation()
                else:
                    link_name = get_link_name(self.parent_obj.get_body_id(), link_id)
                    link_state = utils.get_link_state(self.parent_obj.get_body_id(), link_id)
                    attachment_source_pos = link_state.linkWorldPosition
                    attachment_source_orn = link_state.linkWorldOrientation

                position, orientation = p.multiplyTransforms(
                    attachment_source_pos, attachment_source_orn, pos_offset, orn_offset
                )
                data.append((link_name, position, orientation))

        return data


class WaterStream(ParticleSystem):
    _DROP_PERIOD = 0.1  # new water every this many seconds.
    _SIZE_OPTIONS = np.array(
        [
            [0.02] * 3,
            [0.018] * 3,
            [0.016] * 3,
        ]
    )
    _COLOR_OPTIONS = np.array([(0.61, 0.82, 0.86, 1), (0.5, 0.77, 0.87, 1)])

    def __init__(self, water_source_pos, num, initial_dump=None, **kwargs):
        if initial_dump is not None:
            self.sizes = np.array(initial_dump["sizes"])
            self.colors = np.array(initial_dump["colors"])
        else:
            size_idxs = np.random.choice(len(self._SIZE_OPTIONS), num, replace=True)
            self.sizes = self._SIZE_OPTIONS[size_idxs]

            color_idxs = np.random.choice(len(self._COLOR_OPTIONS), num, replace=True)
            self.colors = self._COLOR_OPTIONS[color_idxs]

        super(WaterStream, self).__init__(
            num=num,
            size=self.sizes,
            color=self.colors,
            visual_only=False,
            mass=0.00005,  # each drop is around 0.05 grams
            use_pbr=True,  # PBR needs to be on for the shiny water particles.
            **kwargs
        )

        self.steps_since_last_drop_step = float("inf")
        self.water_source_pos = water_source_pos
        self.on = False
        self.initial_dump = initial_dump

    def reset_to_dump(self, dump):
        # Assert that the dump is compatible with the particle system state.
        assert np.all(self.sizes == np.array(dump["sizes"])), "Incompatible WaterStream dump."
        assert np.all(self.colors == np.array(dump["colors"])), "Incompatible WaterStream dump."

        self.steps_since_last_drop_step = dump["steps_since_last_drop_step"]

        # Use the ParticleSystem dump reset for the particle positions.
        super(WaterStream, self).reset_to_dump(dump["particle_poses"])

    def initialize(self, simulator):
        super(WaterStream, self).initialize(simulator)

        # For a water source, we are guaranteed to eventually use each particle, so we
        # can immediately load all of them.
        for particle in self.get_particles():
            self._load_particle(particle)

        # Unstash particles in dump.
        if self.initial_dump:
            self.reset_to_dump(self.initial_dump)
            del self.initial_dump

    def _load_particle(self, particle):
        # First load the particle normally.
        body_id = super(WaterStream, self)._load_particle(particle)

        # Set renderer instance settings on the particles.
        instances = self._simulator.renderer.get_instances()
        for instance in instances:
            if instance.pybullet_uuid == body_id:
                instance.roughness = 0
                instance.metalness = 1

        return body_id

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
        if self.steps_since_last_drop_step < self._DROP_PERIOD / simulator.render_timestep:
            self.steps_since_last_drop_step += 1
            return

        # Otherwise, create & drop the water.
        self.unstash_particle(self.water_source_pos, [0, 0, 0, 1])
        self.steps_since_last_drop_step = 0

    def dump(self):
        data = {
            "sizes": [tuple(size) for size in self.sizes],
            "colors": [tuple(color) for color in self.colors],
            "steps_since_last_drop_step": self.steps_since_last_drop_step,
            "particle_poses": super(WaterStream, self).dump(),
        }

        return data


class _Dirt(AttachedParticleSystem):
    """
    This class represents common logic between particle-based dirtyness states like
    dusty and stained. It should not be directly instantiated - use subclasses instead.
    """

    # This parameters are used when sampling dirt particles.
    # See igibson/utils/sampling_utils.py for how they are used.
    _SAMPLING_AXIS_PROBABILITIES = [0.25, 0.25, 0.5]
    _SAMPLING_AABB_OFFSET = 0.1
    _SAMPLING_BIMODAL_MEAN_FRACTION = 0.9
    _SAMPLING_BIMODAL_STDEV_FRACTION = 0.2
    _SAMPLING_MAX_ATTEMPTS = 20

    def __init__(self, parent_obj, clip_into_object, sampling_kwargs=None, **kwargs):
        super(_Dirt, self).__init__(parent_obj, **kwargs)
        if sampling_kwargs is None:
            sampling_kwargs = {}
        self._sampling_kwargs = sampling_kwargs
        self._clip_into_object = clip_into_object

    def randomize(self):
        assert self.get_num_stashed() == self.get_num(), "All particles should be stashed before sampling."

        bbox_sizes = [particle.bounding_box for particle in self.get_stashed_particles()]

        # If we are going to clip into object we need half the height.
        if self._clip_into_object:
            bbox_sizes = [bbox_size * np.array([1, 1, 0.5]) for bbox_size in bbox_sizes]

        results = sampling_utils.sample_cuboid_on_object(
            self.parent_obj,
            self.get_num_stashed(),
            [list(x) for x in bbox_sizes],
            self._SAMPLING_BIMODAL_MEAN_FRACTION,
            self._SAMPLING_BIMODAL_STDEV_FRACTION,
            self._SAMPLING_AXIS_PROBABILITIES,
            max_sampling_attempts=self._SAMPLING_MAX_ATTEMPTS,
            undo_padding=True,
            aabb_offset=self._SAMPLING_AABB_OFFSET,
            refuse_downwards=True,
            **self._sampling_kwargs
        )

        # Reset the activated particle history
        self.reset_particles_activated_at_any_time()

        # Use the sampled points to set the dirt positions.
        for i, particle in enumerate(self.get_stashed_particles()):
            position, normal, quaternion, hit_link, reasons = results[i]

            if position is not None:
                # Compute the point to stick the particle to.
                surface_point = position
                if self._clip_into_object:
                    # Shift the object halfway down.
                    cuboid_base_to_center = bbox_sizes[2] / 2.0
                    surface_point -= normal * cuboid_base_to_center

                # Unstash the particle (and make sure we get the correct one!)
                assert self.unstash_particle(surface_point, quaternion, link_id=hit_link, particle=particle) == particle


class Dust(_Dirt):
    def __init__(self, parent_obj, **kwargs):
        super(Dust, self).__init__(
            parent_obj,
            clip_into_object=True,
            num=20,
            size=[0.015] * 3,
            visual_only=True,
            mass=0,
            color=(0.87, 0.80, 0.74, 1),
            **kwargs
        )


class Stain(_Dirt):
    _PARTICLE_COUNT = 20

    _BOUNDING_BOX_LOWER_LIMIT_FRACTION_OF_AABB = 0.06
    _BOUNDING_BOX_LOWER_LIMIT_MIN = 0.01
    _BOUNDING_BOX_LOWER_LIMIT_MAX = 0.02

    _BOUNDING_BOX_UPPER_LIMIT_FRACTION_OF_AABB = 0.1
    _BOUNDING_BOX_UPPER_LIMIT_MIN = 0.02
    _BOUNDING_BOX_UPPER_LIMIT_MAX = 0.1

    _MESH_FILENAME = os.path.join(igibson.assets_path, "models/stain/stain.obj")
    _MESH_BOUNDING_BOX = np.array([0.0368579992, 0.03716399827, 0.004])

    def __init__(self, parent_obj, initial_dump=None, **kwargs):
        if initial_dump:
            self.random_bbox_dims = np.array(initial_dump["random_bbox_dims"])
        else:
            # Particle size range changes based on parent object size.
            from igibson.object_states import AABB

            median_aabb_dim = np.median(
                parent_obj.bounding_box
                if hasattr(parent_obj, "bounding_box") and parent_obj.bounding_box is not None
                else get_aabb_extent(parent_obj.states[AABB].get_value())
            )

            bounding_box_lower_limit_from_aabb = self._BOUNDING_BOX_LOWER_LIMIT_FRACTION_OF_AABB * median_aabb_dim
            bounding_box_lower_limit = np.clip(
                bounding_box_lower_limit_from_aabb,
                self._BOUNDING_BOX_LOWER_LIMIT_MIN,
                self._BOUNDING_BOX_LOWER_LIMIT_MAX,
            )

            bounding_box_upper_limit_from_aabb = self._BOUNDING_BOX_UPPER_LIMIT_FRACTION_OF_AABB * median_aabb_dim
            bounding_box_upper_limit = np.clip(
                bounding_box_upper_limit_from_aabb,
                self._BOUNDING_BOX_UPPER_LIMIT_MIN,
                self._BOUNDING_BOX_UPPER_LIMIT_MAX,
            )

            # Here we randomize the size of the base (XY plane) of the stain while keeping height constant.
            random_bbox_base_size = np.random.uniform(
                bounding_box_lower_limit, bounding_box_upper_limit, self._PARTICLE_COUNT
            )
            self.random_bbox_dims = np.stack(
                [
                    random_bbox_base_size,
                    random_bbox_base_size,
                    np.full_like(random_bbox_base_size, self._MESH_BOUNDING_BOX[2]),
                ],
                axis=-1,
            )

        super(Stain, self).__init__(
            parent_obj,
            clip_into_object=False,
            num=self._PARTICLE_COUNT,
            size=self.random_bbox_dims,
            base_shape="mesh",
            mesh_filename=self._MESH_FILENAME,
            mesh_bounding_box=self._MESH_BOUNDING_BOX,
            visual_only=True,
            initial_dump=initial_dump,
            **kwargs
        )

    def reset_to_dump(self, dump):
        # Assert that the dump is compatible.
        assert np.all(np.array(dump["random_bbox_dims"]) == self.random_bbox_dims)

        # Call the dump resetter of the parent.
        super(Stain, self).reset_to_dump(dump["dirt_dump"])

    def dump(self):
        return {
            "dirt_dump": super(Stain, self).dump(),
            "random_bbox_dims": [tuple(bbox) for bbox in self.random_bbox_dims],
        }
