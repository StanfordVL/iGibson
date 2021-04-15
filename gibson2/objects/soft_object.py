from gibson2.objects.object_base import Object
import pybullet as p


class SoftObject(Object):
    """
    Soft object
    """

    def __init__(self, fileName, simFileName, basePosition=[0, 0, -1], baseOrientation = [0,0,0,1], scale=1,
                        mass=1, useMassSpring=0, useBendingSprings=0, useNeoHookean=0, springElasticStiffness=1,
                        springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0.1,
                        NeoHookeanMu=1, NeoHookeanLambda=1, NeoHookeanDamping=1, frictionCoeff=.5, useFaceContact=0,
                        useSelfCollision=0, collisionMargin=0.001, repulsionStiffness=0.5):
        super(SoftObject, self).__init__()
        self.simFileName = simFileName
        self.fileName = fileName
        self.scale = scale
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.mass = mass
        self.useMassSpring = useMassSpring
        self.useBendingSprings = useBendingSprings
        self.collisionMargin = collisionMargin
        self.useNeoHookean = useNeoHookean
        self.springElasticStiffness = springElasticStiffness
        self.springDampingStiffness = springDampingStiffness
        self.springDampingAllDirections = springDampingAllDirections
        self.springBendingStiffness = springBendingStiffness
        self.NeoHookeanMu = NeoHookeanMu
        self.NeoHookeanLambda = NeoHookeanLambda
        self.NeoHookeanDamping = NeoHookeanDamping
        self.frictionCoeff = frictionCoeff
        self.useSelfCollision = useSelfCollision
        self.useFaceContact = useFaceContact
        self.repulsionStiffness = repulsionStiffness

    def _load(self):
        """
        Load the object into pybullet
        """
        body_id = p.loadSoftBody(self.fileName, simFileName=self.simFileName,
                                 scale=self.scale, basePosition=self.basePosition,
                                 baseOrientation=self.baseOrientation, mass=self.mass,
                                 useMassSpring=self.useMassSpring, useBendingSprings=self.useBendingSprings,
                                 collisionMargin=self.collisionMargin,  useNeoHookean=self.useNeoHookean,
                                 springElasticStiffness=self.springElasticStiffness, springDampingStiffness=self.springDampingStiffness,
                                 springDampingAllDirections=self.springDampingAllDirections, springBendingStiffness=self.springBendingStiffness,
                                 NeoHookeanMu=self.NeoHookeanMu, NeoHookeanLambda=self.NeoHookeanLambda,
                                 NeoHookeanDamping=self.NeoHookeanDamping, frictionCoeff=self.frictionCoeff,
                                 useSelfCollision=self.useSelfCollision, useFaceContact=self.useFaceContact,
                                 repulsionStiffness=self.repulsionStiffness)



        return body_id

    def add_anchor(self, nodeIndex=-1, bodyUniqueId=-1, linkIndex=-1,
                   bodyFramePosition=[0, 0, 0], physicsClientId=0):
        """
        Create soft body anchor
        """
        p.createSoftBodyAnchor(self.body_id, nodeIndex, bodyUniqueId,
                               linkIndex, bodyFramePosition, physicsClientId)


class RiggedSoftObject(Object):
    def __init__(self, skeleton_filename, visual_filename):
        super(RiggedSoftObject, self).__init__()

    def _load(self):
        pass