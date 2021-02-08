from gibson2.objects.object_base import Object
import pybullet as p
import numpy as np

class Particle(Object):
    """
    Cube shape primitive
    """

    def __init__(self, pos=[0,0,0], dim=0.1, visual_only=False, mass=0.1, color=[1, 1, 1, 1]):
        super(Particle, self).__init__()
        self.basePos = pos
        self.dimension = [dim, dim, dim]
        self.visual_only = visual_only
        self.mass = mass
        self.color = color
        self.states = dict()

    def _load(self):
        """
        Load the object into pybullet
        """
        baseOrientation = [0, 0, 0, 1]
        colBoxId = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=self.dimension)
        visualShapeId = p.createVisualShape(
            p.GEOM_BOX, halfExtents=self.dimension, rgbaColor=self.color)
        if self.visual_only:
            body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visualShapeId)
        else:
            body_id = p.createMultiBody(baseMass=self.mass,
                                        baseCollisionShapeIndex=colBoxId,
                                        baseVisualShapeIndex=visualShapeId)

        p.resetBasePositionAndOrientation(
            body_id, np.array(self.basePos), baseOrientation)

        return body_id

class ParticleSystem:
    def __init__(self, pos=[0,0,0], dim=0.1, offset=0.4, num=15, visual_only=False, mass=0.1, color=[1, 1, 1, 1]):
        self.particles = []
        self.offset = offset
        self.num = num
        for i in range(num):
            self.particles.append(Particle(pos=[pos[0], pos[1], pos[2] + i * offset],
                                           dim=dim,
                                           visual_only=visual_only,
                                           mass=mass,
                                           color=color))

class WaterStream(ParticleSystem):
    def __init__(self, pos=[0,0,0], dim=0.01, offset=-0.04, num=15, visual_only=True, mass=0, color=[0,0,1,1]):
        super(WaterStream, self).__init__(
            pos=pos,
            dim=dim,
            offset=offset,
            num=num,
            visual_only=visual_only,
            mass=mass,
            color=color
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
        pass


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

    def animate(self):
        pass

    def attach(self, obj):
        aabb = obj.states['aabb'].get_value()
        for i in range(self.num):
            good_hit = False
            iter = 0
            while not good_hit and iter < 100:
                x = np.random.uniform(aabb[0][0], aabb[1][0])
                y = np.random.uniform(aabb[0][1], aabb[1][1])
                zmax = aabb[1][2] + 0.1
                zmin = aabb[0][2]
                res = p.rayTest(rayFromPosition=[x,y,zmax], rayToPosition=[x,y,zmin])
                print(x,y,zmin, zmax,res, iter)
                if len(res) > 0 and res[0][0] == obj.get_body_id():
                    good_hit = True
                    hit_pos = res[0][3]
                    self.particles[i].set_position(hit_pos)
                iter += 1

    def step(self):
        # detect sponges around it and remove particles
        pass

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

    def step(self):
        # detect wet sponges around it and remove particles
        pass