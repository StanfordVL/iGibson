from flask import Flask, render_template, Response, request, session
import sys
import pickle
import json
from tasknet.parsing import construct_full_pddl

from gibson2.simulator import Simulator
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
import gibson2
import os

from gibson2.objects.ycb_object import YCBObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from gibson2.render.profiler import Profiler
import cv2
from PIL import Image
from io import BytesIO
import base64
import binascii
import multiprocessing
import traceback
import atexit
import time
import cv2
import uuid

interactive = True

def pil_image_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())


class ProcessPyEnvironment(object):
    """Step a single env in a separate process for lock free paralellism."""

    # Message types for communication via the pipe.
    _READY = 1
    _ACCESS = 2
    _CALL = 3
    _RESULT = 4
    _EXCEPTION = 5
    _CLOSE = 6

    def __init__(self, env_constructor):
        self._env_constructor = env_constructor


    def start(self):
        """Start the process."""
        print("STARTING")
        self._conn, conn = multiprocessing.Pipe()
        print("CREATED CONN")
        self._process = multiprocessing.Process(target=self._worker,
                                                args=(conn, self._env_constructor))
        print("CREATED PROCESS")
        atexit.register(self.close)
        print("REGISTER?")
        self._process.start()
        print("STARTED PROCESS")
        result = self._conn.recv()
        print("GOT RESULT FROM CONN")
        if isinstance(result, Exception):
            self._conn.close()
            self._process.join(5)
            raise result
        assert result is self._READY, result

    def __getattr__(self, name):
        """Request an attribute from the environment.
        Note that this involves communication with the external process, so it can
        be slow.

        :param name: attribute to access.
        :return: value of the attribute.
        """
        print('gettinng', name)
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

        :param name: name of the method to call.
        :param args: positional arguments to forward to the method.
        :param kwargs: keyword arguments to forward to the method.
        :return: promise object that blocks and provides the return value when called.
        """
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join(5)

    def step(self, action, blocking=True):
        """Step the environment.

        :param action: the action to apply to the environment.
        :param blocking: whether to wait for the result.
        :return: (next_obs, reward, done, info) tuple when blocking, otherwise callable that returns that tuple
        """
        promise = self.call('step', action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=True):
        """Reset the environment.

        :param blocking: whether to wait for the result.
        :return: next_obs when blocking, otherwise callable that returns next_obs
        """
        promise = self.call('reset')
        if blocking:
            return promise()
        else:
            return promise
    
    def sample(self, pddl, blocking=True):
        """Run a sampling in the environment

        :param pddl (str): the pddl being sampled in the environment
        :param blocking (bool): whether to wait for the result
        :return (bool, str): (success, feedback) from the sampling process
        """
        promise = self.call("sample", pddl)
        if blocking: 
            return promise()
        else:
            return promise

    def _receive(self):
        """Wait for a message from the worker process and return its payload.

        :raise Exception: an exception was raised inside the worker process.
        :raise KeyError: the reveived message is of an unknown type.

        :return: payload object of the message.
        """
        message, payload = self._conn.recv()
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        self.close()
        raise KeyError(
            'Received message of unexpected type {}'.format(message))

    def _worker(self, conn, env_constructor):
        """The process waits for actions and sends back environment results.

        :param conn: connection for communication to the main process.
        :param env_constructor: env_constructor for the OpenAI Gym environment.


        :raise KeyError: when receiving a message of unknown type.
        """
        print("ENTERED WORKER")
        try:
            print("TRYING")
            np.random.seed()
            print("SEEDED")
            env = env_constructor()
            print("MADE THE ENVIRONMENT")
            conn.send(self._READY)    # Ready.
            print("SENT READY MESSAGE")
            while True:
                print("WHILE")
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    if name == 'step' or name == 'reset':
                        result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    getattr(env, 'close')()
                    assert payload is None
                    break
                raise KeyError(
                    'Received message of unknown type {}'.format(message))
        except Exception:    # pylint: disable=broad-except
            etype, evalue, tb = sys.exc_info()
            stacktrace = ''.join(traceback.format_exception(etype, evalue, tb))
            message = 'Error in environment process: {}'.format(stacktrace)
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            conn.close()


class ToyEnv(object):
    def __init__(self):
        config = parse_config(os.path.join(gibson2.example_config_path, 'turtlebot_demo.yaml'))
        hdr_texture = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
        hdr_texture2 = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
        light_modulation_map_filename = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
        background_texture = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

        settings = MeshRendererSettings(enable_shadow=False, enable_pbr=False)
       

        self.s = Simulator(mode='headless', image_width=400,
                      image_height=400, rendering_settings=settings)
        scene = StaticIndoorScene('Rs')
        self.s.import_scene(scene)
        #self.s.import_ig_scene(scene)

        for _ in range(5):
            obj = YCBObject('003_cracker_box')
            self.s.import_object(obj)
            obj.set_position_orientation(np.random.uniform(
                low=0, high=2, size=3), [0, 0, 0, 1])
        print(self.s.renderer.instances)

    def step(self, a):
        self.s.step()
        frame = self.s.renderer.render_robot_cameras(modes=('rgb'))[0]
        return frame

    def close(self):
        self.s.disconnect()



class ToyEnvInt(object):
    def __init__(self, scene='Rs_int'):
        # TODO this config may need to change 
        config = parse_config(os.path.join(gibson2.example_config_path, 'turtlebot_demo.yaml'))
        hdr_texture = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
        hdr_texture2 = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
        light_modulation_map_filename = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
        background_texture = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

        self.scene_id = scene
        scene = InteractiveIndoorScene(
            scene, texture_randomization=False, object_randomization=False)
        #scene._set_first_n_objects(5)
        scene.open_all_doors()

        settings = MeshRendererSettings(env_texture_filename=hdr_texture,
                                        env_texture_filename2=hdr_texture2,
                                        env_texture_filename3=background_texture,
                                        light_modulation_map_filename=light_modulation_map_filename,
                                        enable_shadow=True, msaa=True,
                                        light_dimming_factor=1.0,
                                        optimized=True)
        print("STARTING SIM INITIALIZATION")
        self.s = Simulator(mode='headless', image_width=400,
                      image_height=400, rendering_settings=settings)
        print("FINISHED SIM INIT")
        self.s.import_ig_scene(scene)
        print("FINISHED SCENE IMPORT")

        for _ in range(5):
            obj = YCBObject('003_cracker_box')
            self.s.import_object(obj)
            obj.set_position_orientation(np.random.uniform(
                low=0, high=2, size=3), [0, 0, 0, 1])
        print(self.s.renderer.instances)

        self.locked = False

    def step(self, a):
        self.s.step()
        frame = self.s.renderer.render_robot_cameras(modes=('rgb'))[0]
        return frame

    def sample(self, pddl):
        # TODO implement 
        return False, "Tester feedback"

    def lock(self):
        self.locked = True
    
    def unlock(self):
        self.locked = False 

    def close(self):
        self.s.disconnect()


class iGFlask(Flask):
    def __init__(self, args, **kwargs):
        super(iGFlask, self).__init__(args, **kwargs)
        self.action= {}
        self.envs = {}
        self.envs_inception_time = {}
    def cleanup(self):
        print("ENVS TO CLEANUP:", self.envs)
        for k,v in self.envs_inception_time.items():
            if time.time() - v > 200:
                # clean up an old environment
                self.stop_app(k)

    def prepare_app(self, uuid, scene):
        print("ABOUT TO CLEANUP")
        self.cleanup()
        print("CLEANED UP")
        def env_constructor():
            if interactive:
                return ToyEnvInt(scene=scene)
            else:
                return ToyEnv()
        print("ABOUT TO CONSTRUCT")
        self.envs[uuid] = ProcessPyEnvironment(env_constructor)
        print("CONSTRUCTED ONE")
        self.envs[uuid].start()
        print("STARTED ONE")
        self.envs_inception_time[uuid] = time.time()
        print("ENVS:", self.envs)

    def stop_app(self, uuid):
        self.envs[uuid].close()
        del self.envs[uuid]
        del self.envs_inception_time[uuid]


app = iGFlask(__name__)


########### REQUEST HANDLERS ###########

@app.route('/', methods=["POST"])
def index():
    id = uuid.uuid4()
    return render_template('index.html', uuid=id)

@app.route("/setup", methods=["POST"])
def setup():
    """Set up the three environments when requested by annotation React app"""
    print("STARTING SETUP")
    scenes = json.loads(request.data)       # TODO check what this looks like
    ids = [str(uuid.uuid4()) for __ in range(len(scenes))]
    print("MADE IDS")
    for scene, unique_id in zip(scenes, ids):
        print("PREPARING ONE APP")
        app.prepare_app(scene, unique_id)             # TODO uncomment when basic infra is done 
        print(f"Instantiated {scene} with uuid {unique_id}")

    return Response(json.dumps({"uuids": ids}))

    # TODO need to send uuids to the upcoming POSTs somehow, I guess in the response


@app.route("/check_sampling", methods=["POST"])
def check_sampling():
    """Check PDDL sent by React app in all three relevant scenes 

    :return (Response): response indicating success of sampling in all three 
                         scenes, feedback given from each 
    """
    # Prepare data 
    data = json.loads(request.data)
    atus_activity = data["activityName"]
    init_state = data["initialConditions"]     
    goal_state = data["goalConditions"]
    object_list = data["objectList"]
    # pddl = init_state + goal_state + object_list        # TODO fix using existing utils
    pddl = construct_full_pddl(
                atus_activity, 
                "feasibility_check", 
                object_list,
                init_state,
                goal_state)
    ids = data["uuids"]

    # Try sampling
    num_successful_scenes = 0
    feedback_instances = []
    for unique_id in ids:
        init_success, goal_success, init_feedback, goal_feedback = app.envs[unique_id].sample(pddl)
        if init_success and goal_success:
            num_successful_scenes += 1
            feedback_instances.append((init_feedback, goal_feedback))
    success = num_successful_scenes >= 3
    feedback = str(feedback_instances)      # TODO make prettier 

    return Response(json.dumps({"success": success, "feedback": feedback}))


@app.route("/teardown", methods=["POST"])
def teardown():
    """Tear down the three environments created for this instance of the React app"""
    data = json.loads(request.data)
    unique_ids = data["uuids"]
    for unique_id in unique_ids:
        print(f"uuid {unique_id} pretend-stopped")
        # app.stop_app(unique_id)       # TODO uncomment when ready 
    
    return Response(json.dumps({"success": True}))      # TODO need anything else? 


if __name__ == '__main__':
    port = int(sys.argv[1])
    # app.run(host="0.0.0.0", port=port, debug=True)
    app.run(port=port, debug=True)
