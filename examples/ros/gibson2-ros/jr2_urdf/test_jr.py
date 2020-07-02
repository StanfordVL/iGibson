import pybullet as p
p.connect(p.GUI)
p.loadURDF('jr2_kinova.urdf')

while True:
    p.stepSimulation()
