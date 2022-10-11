from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.simulator import Simulator
import cv2
import numpy as np
from IPython import embed

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1024

sim = Simulator(
    image_height=IMAGE_HEIGHT,
    image_width=IMAGE_WIDTH,
    mode = 'headless',
)

scene = InteractiveIndoorScene(scene_id="Ihlen_1_int")
sim.import_scene(scene)
render = sim.renderer

render.set_camera([-5, 1.8, 1.1], [3.4, 1.8, 1.1], [0, 0, 1])

frames = render.render(modes=("rgb", "3d"))
depth = np.linalg.norm(frames[1][:, :, :3], axis=2)
depth /= depth.max()
frames[1][:, :, :3] = depth[..., None]

cv2.imshow("image", cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
sim.step()

embed()

