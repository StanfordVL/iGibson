import gibson2.object_states
from gibson2.objects.object_base import Object
import pybullet as p
import numpy as np

class Particle(Object):
    """
    A particle object, used to simulate water stream and dust/stain
    """

    def __init__(self, pos=[0,0,0], dim=0.1, visual_only=False, mass=0.1, color=[1, 1, 1, 1], base_shape="box"):
        super(Particle, self).__init__()
        self.base_pos = pos
        self.dimension = [dim, dim, dim]
        self.visual_only = visual_only
        self.mass = mass
        self.color = color
        self.active = True
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

        return body_id

    def force_sleep(self):
        activationState = p.ACTIVATION_STATE_ENABLE_SLEEPING + p.ACTIVATION_STATE_SLEEP
        p.changeDynamics(self.body_id, -1, activationState=activationState)

    def force_wakeup(self):
        activationState = p.ACTIVATION_STATE_ENABLE_SLEEPING + p.ACTIVATION_STATE_WAKE_UP
        p.changeDynamics(self.body_id, -1, activationState=activationState)

class ParticleSystem:
    def __init__(self, pos=[0,0,0], dim=0.1, offset=0.4, num=15, visual_only=False, mass=0.1, color=[1, 1, 1, 1],
                 base_shape="box"):
        self.particles = []
        self.offset = offset
        self.num = num
        for i in range(num):
            self.particles.append(Particle(pos=[pos[0], pos[1], pos[2] + i * offset],
                                           dim=dim,
                                           visual_only=visual_only,
                                           mass=mass,
                                           color=color,
                                           base_shape=base_shape))
        self.visual_only = visual_only

    def register_parent_obj(self, obj):
        self.parent_obj = obj
        obj.attached_particle_system.append(self)

    def get_num(self):
        return self.num

    def get_num_active(self):
        s = 0
        for i in range(self.num):
            if self.particles[i].active:
                s += 1
        return s

    def stash_particle(self, particle):
        particle.set_position([np.random.uniform(-10, 10), np.random.uniform(-10, 10), -100])
        particle.force_sleep()
        particle.active = False

class WaterStreamAnimation(ParticleSystem):
    def __init__(self, pos=[0,0,0], dim=0.01, offset=-0.04, num=15, visual_only=True, mass=0, color=[0,0,1,1]):
        super(WaterStreamAnimation, self).__init__(
            pos=pos,
            dim=dim,
            offset=offset,
            num=num,
            visual_only=visual_only,
            mass=mass,
            color=color,
            base_shape="sphere",
        )
        self.animation_step = 0

    def animate(self):
        self.animation_step += 1
        if self.animation_step % 10 == 0:
            for particle in self.particles:
                particle.set_position(particle.get_position() - np.array([0,0,self.offset]))

        for particle in self.particles:
            particle.set_position(particle.get_position() + np.array([0,0,self.offset * 0.1]))

    def step(self):
        # detect soakable around it, and change soakable state
        self.animate()

class WaterStreamPhysicsBased(ParticleSystem):
    def __init__(self, pos=[0, 0, 0], dim=0.01, offset=-0.04, num=15, visual_only=False, mass=0.1, color=[0, 0, 1, 1]):
        super(WaterStreamPhysicsBased, self).__init__(
            pos=pos,
            dim=dim,
            offset=offset,
            num=num,
            visual_only=visual_only,
            mass=mass,
            color=color,
            base_shape="sphere",
        )
        self.step_elapsed = 0
        self.water_source_pos = pos
        # set a rest position somewhere
        self.on = False

    def set_value(self, on):
        self.on = on

    def step(self):
        if self.on:
            # every n steps, move to a particle the water source
            # detect sinks soakable around it, and change soakable state
            self.step_elapsed += 1
            period = 30 # assuming a 30 fps simulation, 1 second 1 drop seems reasonable
            n_particle = len(self.particles)
            if self.step_elapsed % period == 0:
                particle_idx = self.step_elapsed // period % n_particle
                particle = self.particles[particle_idx]
                if not particle.active:
                    particle.set_position(self.water_source_pos)
                    particle.force_wakeup()
                    particle.active = True

class Dust(ParticleSystem):
    def __init__(self, pos=[0,0,0], dim=0.01, offset=-0.04, num=15, visual_only=True, mass=0, color=[0,0,0,1]):
        super(Dust, self).__init__(
            pos=pos,
            dim=dim,
            offset=offset,
            num=num,
            visual_only=visual_only,
            mass=mass,
            color=color
        )

    def attach(self, obj):
        aabb = obj.states[object_states.AABB].get_value()
        for i in range(self.num):
            good_hit = False
            iter = 0
            while not good_hit and iter < 100:
                x = np.random.uniform(aabb[0][0], aabb[1][0])
                y = np.random.uniform(aabb[0][1], aabb[1][1])
                zmax = aabb[1][2] + 0.1
                zmin = aabb[0][2]
                res = p.rayTest(rayFromPosition=[x,y,zmax], rayToPosition=[x,y,zmin])
                # print(x,y,zmin, zmax,res, iter)
                if len(res) > 0 and res[0][0] == obj.get_body_id():
                    good_hit = True
                    hit_pos = res[0][3]
                    self.particles[i].set_position(hit_pos)
                iter += 1

class Stain(Dust):
    def __init__(self, pos=[0,0,0], dim=0.01, offset=-0.04, num=15, visual_only=True, mass=0, color=[0,0,0,1]):
        super(Stain, self).__init__(
            pos=pos,
            dim=dim,
            offset=offset,
            num=num,
            visual_only=visual_only,
            mass=mass,
            color=color
        )
