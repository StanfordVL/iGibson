from gibson2.objects.object_base import Object
import pybullet as p


class SoftObject(Object):
    """
    Soft object
    """

    def __init__(self, fileName, simFileName, basePosition=[0, 0, -1], baseOrientation = [0,0,0,1], scale=1,
                        mass=1, useNeoHookean=1,
                        NeoHookeanMu=1, NeoHookeanLambda=1, NeoHookeanDamping=1, useSelfCollision=1,
                        frictionCoeff=.5, collisionMargin=0.001):
        super(SoftObject, self).__init__()
        self.simFileName = simFileName
        self.fileName = fileName
        self.scale = scale
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.mass = mass
        self.collisionMargin = collisionMargin
        self.useNeoHookean = useNeoHookean
        self.NeoHookeanMu = NeoHookeanMu
        self.NeoHookeanLambda = NeoHookeanLambda
        self.NeoHookeanDamping = NeoHookeanDamping
        self.frictionCoeff = frictionCoeff
        self.useSelfCollision = useSelfCollision

    def _load(self):
        """
        Load the object into pybullet
        """
        body_id = p.loadSoftBody(self.fileName, simFileName=self.simFileName,
                                 scale=self.scale, basePosition=self.basePosition,
                                 baseOrientation=self.baseOrientation, mass=self.mass,
                                 collisionMargin=self.collisionMargin,  useNeoHookean=self.useNeoHookean,
                                 NeoHookeanMu=self.NeoHookeanMu, NeoHookeanLambda=self.NeoHookeanLambda,
                                 NeoHookeanDamping=self.NeoHookeanDamping, frictionCoeff=self.frictionCoeff,
                                 useSelfCollision=self.useSelfCollision)



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