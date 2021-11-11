class PhysicsSettings(object):
    def __init__(
        self,
        solver_iterations=100,
        enhanced_determinism=False,
    ):
        self.solver_iterations = solver_iterations
        enhanced_determinism = enhanced_determinism
