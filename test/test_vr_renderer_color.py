from gibson2.core.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
import cv2
import numpy as np

# Test program that renders the color red onto the vr headset

renderer = MeshRendererVR(MeshRenderer)
renderer.setup_debug_framebuffer()

while True:
    renderer.render_debug_framebuffer()

renderer.release()