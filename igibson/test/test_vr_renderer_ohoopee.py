import time

from igibson.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.core.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR

from igibson import dataset_path

renderer = MeshRendererVR(MeshRenderer, msaa=True)
# Note that it is necessary to load the full path of an object!
renderer.load_object(dataset_path + "\\Ohoopee\\Ohoopee_mesh_texture.obj")
renderer.add_instance(0)

while True:
    startFrame = time.time()
    renderer.render()

    endFrame = time.time()
    deltaT = endFrame - startFrame
    fps = 1 / float(deltaT)

    print("Current fps: %f" % fps)

renderer.release()
