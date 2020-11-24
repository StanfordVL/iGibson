from flask import Flask, render_template, Response
import sys
import pickle
from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
import numpy as np
from gibson2.render.profiler import Profiler
import cv2
from PIL import Image
from io import BytesIO
import base64
import binascii

app = Flask(__name__)


def pil_image_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    config = parse_config('../../examples/configs/turtlebot_demo.yaml')
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(mode='headless', image_width=256,
                  image_height=256, rendering_settings=settings)
    scene = StaticIndoorScene('Rs',
                              build_graph=True,
                              pybullet_load_texture=True)
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    for _ in range(10):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)
        obj.set_position_orientation(np.random.uniform(
            low=0, high=2, size=3), [0, 0, 0, 1])
    print(s.renderer.instances)

    while True:
        turtlebot.apply_action([0.1, -0.1])
        s.step()
        frame = s.renderer.render_robot_cameras(modes=('rgb'))[0]
        frame = (frame[:, :, :3] * 255).astype(np.uint8)
        frame = pil_image_to_base64(Image.fromarray(frame))
        frame = binascii.a2b_base64(frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app = Flask(__name__)
    port = "5552"
    if len(sys.argv) > 1:
        port = sys.argv[1]

    if len(sys.argv) > 2:
        port_web = sys.argv[2]
        port_web = int(port_web)
    else:
        port_web = 5001
    app.run(host='0.0.0.0', port=port_web, debug=False)
