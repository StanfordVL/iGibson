from flask import Flask, render_template, Response, request, session
from flask_apscheduler import APScheduler
from flask_cors import CORS 
import sys
import json
import tasknet 
from tasknet.parsing import construct_full_pddl
from tasknet.logic_base import UncontrolledCategoryError

from gibson2.simulator import Simulator
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.task.task_base import iGTNTask
import gibson2
import os

from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from PIL import Image
from io import BytesIO
import multiprocessing
import traceback
import atexit
import time
import uuid

interactive = True
NUM_REQUIRED_SUCCESSFUL_SCENES = 3

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
        self.last_active_time = time.time()

    def start(self):
        """Start the process."""
        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._worker,
                                                args=(conn, self._env_constructor))
        atexit.register(self.close)
        self.last_active_time = time.time()
        self._process.start()
        result = self._conn.recv()
        self.last_active_time = time.time()
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
        self.last_active_time = time.time()
        promise = self.call("sample", pddl)
        self.last_active_time = time.time()     
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
        try:
            np.random.seed()
            env = env_constructor()
            conn.send(self._READY)    # Ready.
            while True:
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
                    if name == 'step' or name == 'reset' or name == "sample":
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
        self.s = Simulator(mode='headless', image_width=400,
                      image_height=400, rendering_settings=settings)
        self.s.import_ig_scene(scene)

        # self.last_active_time = time.time()

    def step(self, a):
        self.s.step()
        frame = self.s.renderer.render_robot_cameras(modes=('rgb'))[0]
        return frame

    def sample(self, pddl):
        # TODO implement 
        print("ENTERED ENV SAMPLE")
        tasknet.set_backend("iGibson")
        igtn_task = iGTNTask("tester", "tester", predefined_problem=pddl)
        try:
            # init_success = igtn_task.initialize_simulator(simulator=self.s, 
            #             scene_id=self.scene_id, 
            #             online_sampling=True)
            print("STARTING SLEEP")
            time.sleep(5)
            print("FINISHED SLEEP")

            goal_success = True                    # TODO implement and update 
            init_feedback = "Initial conditions are good to go!" if init_success else "Initial conditions don't work."      # TODO update
            goal_feedback = "Goal conditions are good to go!" if goal_success else "Goal conditions don't work"             # TODO update 
        except UncontrolledCategoryError:
            init_success = False 
            init_feedback = "Cannot check until goal state is fixed."
            goal_success = False 
            goal_feedback = "Goal state has uncontrolled categories."

        print("EXITING ENV SAMPLE")
        self.last_active_time = time.time() 
        return init_success, goal_success, init_feedback, goal_feedback

    def close(self):
        self.s.disconnect()


class iGFlask(Flask):
    def __init__(self, args, **kwargs):
        super(iGFlask, self).__init__(args, **kwargs)
        self.action= {}
        self.envs = {}
        self.envs_inception_time = {}
        self.envs_last_use_time = {}

    def cleanup(self):
        # TODO change this to allow people to make the conditions 
        for k,v in self.envs_inception_time.items():
            if time.time() - v > 200:
                # clean up an old environment
                self.stop_env(k)

    def prepare_env(self, uuid, scene):
        # self.cleanup()
        def env_constructor():
            if interactive:
                return ToyEnvInt(scene=scene)
            else:
                return ToyEnv()
        self.envs[uuid] = ProcessPyEnvironment(env_constructor)
        self.envs[uuid].start()
        self.envs_inception_time[uuid] = time.time()
        self.envs_last_use_time[uuid] = time.time()
    
    def periodic_cleanup(self):
        print("Starting periodic cleanup")
        for uid, env in self.envs:
            last_active_time = env.last_active_time
            print(f"uuid: {uid}, time since last use: {int(time.time() - last_active_time)}")
            if time.time() - last_active_time > 60:                   # TODO magic number
                print("stale uuid:", uid)
                self.stop_env(uid)
    
    # TODO how to update envs_last_use_time? Maybe make it a field in 
    #   the ToyEnvInt or ProcessPyEnv and update it while some call is running? 

    def stop_env(self, uuid):
        self.envs[uuid].close()
        del self.envs[uuid]
        del self.envs_inception_time[uuid]
        del self.envs_last_use_time[uuid]


app = iGFlask(__name__)
CORS(app)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

PERIODIC_CLEANUP_TASK_ID = "interval-task-id"


########### REQUEST HANDLERS ###########

@app.route('/', methods=["POST"])
def index():
    id = uuid.uuid4()
    return render_template('index.html', uuid=id)

@app.route("/setup", methods=["POST"])
def setup():
    """Set up the three environments when requested by annotation React app"""
    scenes = json.loads(request.data)       # TODO check what this looks like
    ids = [str(uuid.uuid4()) for __ in range(len(scenes))]
    for scene, unique_id in zip(scenes, ids):
        app.prepare_env(unique_id, scene)             # TODO uncomment when basic infra is done 
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
    success = num_successful_scenes >= min(NUM_REQUIRED_SUCCESSFUL_SCENES, len(ids))
    feedback = str(feedback_instances)      # TODO make prettier 

    return Response(json.dumps({"success": success, "feedback": feedback}))


@app.route("/teardown", methods=["POST"])
def teardown():
    """Tear down the three environments created for this instance of the React app"""
    data = json.loads(request.data)
    unique_ids = data["uuids"]
    for unique_id in unique_ids:
        app.stop_env(unique_id)       
        print(f"uuid {unique_id} stopped")
    
    return Response(json.dumps({"success": True}))      # TODO need anything else? 


########### PERIODIC CLEANUP ###########

def periodic_cleanup(): 
    app.periodic_cleanup()

scheduler.add_job(
    id=PERIODIC_CLEANUP_TASK_ID, 
    func=periodic_cleanup, 
    seconds=5,
    trigger="interval"
)


if __name__ == '__main__': 
    port = int(sys.argv[1])
    # app.run(host="0.0.0.0", port=port, debug=True)
    app.run(host="0.0.0.0", port=port)
