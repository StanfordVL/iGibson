from igibson.core.render.mesh_renderer.Release import MeshRendererContext

vrsys = MeshRendererContext.VRSystem()
recommendedWidth, recommendedHeight = vrsys.initVR()

print("Initialized VR with width: %d and height: %d" % (recommendedWidth, recommendedHeight))

vrsys.releaseVR()
