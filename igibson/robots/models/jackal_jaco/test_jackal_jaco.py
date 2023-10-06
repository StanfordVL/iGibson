import pybullet as p
p.connect(p.GUI)
p.loadURDF('jackal_jaco.urdf')

while True:
    p.stepSimulation()
