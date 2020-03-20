from VRUtils import VRSystem

vrsys = VRSystem()
recommendedWidth, recommendedHeight = vrsys.initVR()

print("Initialized VR with width: %d and height: %d" % (recommendedWidth, recommendedHeight))

vrsys.releaseVR()