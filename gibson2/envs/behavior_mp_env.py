import argparse
import numpy as np
import time
import tasknet
import types
import gym.spaces
import pybullet as p

from collections import OrderedDict
from gibson2.robots.behavior_robot import BehaviorRobot
from gibson2.envs.behavior_env import BehaviorEnv
from enum import IntEnum
from gibson2.object_states import *
from gibson2.robots.behavior_robot import BREye, BRBody, BRHand
from gibson2.object_states.utils import sample_kinematics
from gibson2.objects.articulated_object import URDFObject
from gibson2.object_states.on_floor import RoomFloor
from gibson2.utils.behavior_robot_planning_utils import plan_hand_motion_br, plan_base_motion_br, \
                                                 dry_run_base_plan, dry_run_arm_plan

from gibson2.external.pybullet_tools.utils import MAX_DISTANCE, CIRCULAR_LIMITS, get_base_difference_fn, \
    get_base_distance_fn, circular_difference, set_base_values, pairwise_collision, get_base_values, direct_path, birrt, \
    PI

NUM_ACTIONS = 6

class ActionPrimitives(IntEnum):
    NAVIGATE_TO = 0
    GRASP = 1
    PLACE_ONTOP = 2
    PLACE_INSIDE = 3
    OPEN = 4
    CLOSE = 5

def get_aabb_volume(lo, hi):
    dimension = hi - lo
    return dimension[0] * dimension[1] * dimension[2]

def detect_collision(bodyA, object_in_hand=None):
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id == bodyA or body_id == object_in_hand:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision

def detect_robot_collision(robot):
    object_in_hand = robot.parts['right_hand'].object_in_hand
    return detect_collision(robot.parts['body'].body_id) or \
           detect_collision(robot.parts['left_hand'].body_id) or \
           detect_collision(robot.parts['right_hand'].body_id, object_in_hand)

class BehaviorMPEnv(BehaviorEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        seed=0,
        action_filter='mobile_manipulation',
        use_motion_planning=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        super(BehaviorMPEnv, self).__init__(config_file=config_file,
                                            scene_id=scene_id,
                                            mode=mode,
                                            action_timestep=action_timestep,
                                            physics_timestep=physics_timestep,
                                            device_idx=device_idx,
                                            render_to_tensor=render_to_tensor,
                                            action_filter=action_filter,
                                            seed=seed,
                                            automatic_reset=automatic_reset)

        self.obj_in_hand = None
        self.use_motion_planning = use_motion_planning
        self.robots[0].initial_z_offset = 0.7

    def load_action_space(self):
        self.task_relevant_objects = [item for item in self.task.object_scope.values() if isinstance(item, URDFObject)
                                      or isinstance(item, RoomFloor)]
        self.num_objects = len(self.task_relevant_objects)
        self.action_space = gym.spaces.Discrete(self.num_objects * NUM_ACTIONS)

    def get_body_ids(self, include_self=False):
        ids = []
        for object in self.scene.get_objects():
            if isinstance(object, URDFObject):
                ids.extend(object.body_ids)

        if include_self:
            ids.append(self.robots[0].parts['left_hand'].get_body_id())
            ids.append(self.robots[0].parts['body'].get_body_id())

        return ids

    def step(self, action):
        obj_list_id = int(action) % self.num_objects
        action_primitive = int(action) // self.num_objects

        # from IPython import embed; embed()
        obj = self.task_relevant_objects[obj_list_id]
        if not (isinstance(obj, BRBody) or isinstance(obj, BRHand) or isinstance(obj, BREye)):
            if action_primitive == ActionPrimitives.NAVIGATE_TO:
                if self.navigate_to_obj(obj, use_motion_planning=self.use_motion_planning):
                    print('PRIMITIVE: navigate to {} success'.format(obj.name))
                else:
                    print('PRIMITIVE: navigate to {} fail'.format(obj.name))

            elif action_primitive == ActionPrimitives.GRASP:
                if self.obj_in_hand is None:
                    if isinstance(obj, URDFObject) and hasattr(obj, 'states') and AABB in obj.states:
                        lo, hi = obj.states[AABB].get_value()
                        volume = get_aabb_volume(lo, hi)
                        if volume < 0.2 * 0.2 * 0.2 and not obj.main_body_is_fixed: # say we can only grasp small objects
                            if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
                                self.grasp_obj(obj, use_motion_planning=self.use_motion_planning)
                                print('PRIMITIVE: grasp {} success, obj in hand {}'.format(obj.name, self.obj_in_hand))
                            else:
                                print('PRIMITIVE: grasp {} fail, too far'.format(obj.name))
                        else:
                            print('PRIMITIVE: grasp {} fail, too big or fixed'.format(obj.name))
            elif action_primitive == ActionPrimitives.PLACE_ONTOP:
                if self.obj_in_hand is not None and self.obj_in_hand != obj:
                    print('PRIMITIVE:attempt to place {} ontop {}'.format(self.obj_in_hand.name, obj.name))

                    if isinstance(obj, URDFObject):
                        if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
                            state = p.saveState()
                            result = sample_kinematics('onTop', self.obj_in_hand, obj, True, use_ray_casting_method=True,
                                                        max_trials=20)
                            if result:
                                print('PRIMITIVE: place {} ontop {} success'.format(self.obj_in_hand.name, obj.name))
                                pos = self.obj_in_hand.get_position()
                                orn = self.obj_in_hand.get_orientation()
                                self.place_obj(state, pos, orn, use_motion_planning=self.use_motion_planning)
                            else:
                                p.removeState(state)
                                print('PRIMITIVE: place {} ontop {} fail, sampling fail'.format(self.obj_in_hand.name, obj.name))

                        else:
                            print('PRIMITIVE: place {} ontop {} fail, too far'.format(self.obj_in_hand.name, obj.name))
                    else:
                        state = p.saveState()
                        result = sample_kinematics('onFloor', self.obj_in_hand, obj, True, use_ray_casting_method=True,
                                                   max_trials=20)
                        if result:
                            print('PRIMITIVE: place {} ontop {} success'.format(self.obj_in_hand.name, obj.name))
                            pos = self.obj_in_hand.get_position()
                            orn = self.obj_in_hand.get_orientation()
                            self.place_obj(state, pos, orn, use_motion_planning=self.use_motion_planning)
                        else:
                            print('PRIMITIVE: place {} ontop {} fail, sampling fail'.format(self.obj_in_hand.name,
                                                                                            obj.name))
                            p.removeState(state)

            elif action_primitive == ActionPrimitives.PLACE_INSIDE:
                if self.obj_in_hand is not None and self.obj_in_hand != obj and isinstance(obj, URDFObject):
                    print('PRIMITIVE:attempt to place {} inside {}'.format(self.obj_in_hand.name, obj.name))
                    if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
                        if (hasattr(obj, 'states') and Open in obj.states and obj.states[Open].get_value()) \
                                or (hasattr(obj, 'states') and not Open in obj.states):
                            state = p.saveState()
                            result = sample_kinematics('inside', self.obj_in_hand, obj, True, use_ray_casting_method=True,
                                                       max_trials=20)
                            if result:
                                print('PRIMITIVE: place {} inside {} success'.format(self.obj_in_hand.name, obj.name))
                                pos = self.obj_in_hand.get_position()
                                orn = self.obj_in_hand.get_orientation()
                                self.place_obj(state, pos, orn, use_motion_planning=self.use_motion_planning)
                            else:
                                print('PRIMITIVE: place {} inside {} fail, sampling fail'.format(self.obj_in_hand.name, obj.name))
                                p.removeState(state)
                        else:
                            print('PRIMITIVE: place {} inside {} fail, need open not open'.format(self.obj_in_hand.name,
                                                                                             obj.name))
                    else:
                        print('PRIMITIVE: place {} inside {} fail, too far'.format(self.obj_in_hand.name, obj.name))
            elif action_primitive == ActionPrimitives.OPEN:
                if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
                    if hasattr(obj, 'states') and Open in obj.states:
                        obj.states[Open].set_value(True)
                    else:
                        print('PRIMITIVE open failed, cannot be opened')
                else:
                    print('PRIMITIVE open failed, too far')

            elif action_primitive == ActionPrimitives.CLOSE:
                if np.linalg.norm(np.array(obj.get_position()) - np.array(self.robots[0].get_position())) < 2:
                    if hasattr(obj, 'states') and Open in obj.states:
                        obj.states[Open].set_value(False)
                    else:
                        print('PRIMITIVE close failed, cannot be opened')
                else:
                    print('PRIMITIVE close failed, too far')

        state, reward, done, info = super(BehaviorMPEnv, self).step(np.zeros(17))
        print("PRIMITIVE satisfied predicates:", info["satisfied_predicates"])
        return state, reward, done, info

    def grasp_obj(self, obj, use_motion_planning=False):
        if use_motion_planning:
            x,y,_ = obj.get_position()
            z = obj.states[AABB].get_value()[1][2]
            hand_x, hand_y, hand_z = self.robots[0].parts['right_hand'].get_position()

            x += np.random.uniform(-0.025, 0.025)
            y += np.random.uniform(-0.025, 0.025)
            z += np.random.uniform(-0.025, 0.025)

            minx = min(x, hand_x) - 0.5
            miny = min(y, hand_y) - 0.5
            minz = min(z, hand_z) - 0.5
            maxx = max(x, hand_x) + 0.5
            maxy = max(y, hand_y) + 0.5
            maxz = max(z, hand_z) + 0.5

            state = p.saveState()
            plan = plan_hand_motion_br(self.robots[0], None, [x, y, z+0.05, 0, np.pi * 5/6.0, np.random.uniform(-np.pi, np.pi)], ((minx, miny, minz), (maxx, maxy, maxz)),
                                       obstacles=self.get_body_ids(include_self=True))
            p.restoreState(state)
            p.removeState(state)

            if plan is not None:
                grasp_success = self.execute_grasp_plan(plan, obj)
                print('grasp success', grasp_success)
                if grasp_success:
                    self.obj_in_hand = obj
                else:
                    print("grasp failed")
                    for _ in range(100):
                        self.robots[0].parts['right_hand'].set_close_fraction(0)
                        self.robots[0].parts['right_hand'].trigger_fraction = 0
                        p.stepSimulation()
            else:
                print('plan is None')
                self.robots[0].set_position_orientation(self.robots[0].get_position(), self.robots[0].get_orientation())
                for _ in range(100):
                    self.robots[0].parts['right_hand'].set_close_fraction(0)
                    self.robots[0].parts['right_hand'].trigger_fraction = 0
                    p.stepSimulation()
                #reset hand
        else:
            self.obj_in_hand = obj
            obj.set_position(np.array(self.robots[0].parts['right_hand'].get_position())) #  - np.array([0,0,0.05])
            self.robots[0].parts['right_hand'].set_close_fraction(1)
            self.robots[0].parts['right_hand'].trigger_fraction = 1
            p.stepSimulation()
            obj.set_position(np.array(self.robots[0].parts['right_hand'].get_position()))
            self.robots[0].parts['right_hand'].handle_assisted_grasping(np.zeros(28, ),
                                                                        override_ag_data=(obj.body_id[0], -1))

    def execute_grasp_plan(self, plan, obj):
        for x,y,z,roll,pitch,yaw in plan:
            self.robots[0].parts['right_hand'].move([x,y,z], p.getQuaternionFromEuler([roll, pitch, yaw]))
            p.stepSimulation()

        x,y,z,roll,pitch,yaw = plan[-1]


        for i in range(25):
            self.robots[0].parts['right_hand'].move([x, y, z-i * 0.005], p.getQuaternionFromEuler([roll, pitch, yaw]))
            p.stepSimulation()

        for _ in range(50):
            self.robots[0].parts['right_hand'].set_close_fraction(1)
            self.robots[0].parts['right_hand'].trigger_fraction = 1
            p.stepSimulation()

        grasp_success = self.robots[0].parts['right_hand'].handle_assisted_grasping(np.zeros(28,),
                                                                    override_ag_data=(obj.body_id[0], -1))

        for x, y, z, roll, pitch, yaw in plan[::-1]:
            self.robots[0].parts['right_hand'].move([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
            p.stepSimulation()

        return grasp_success


    def place_obj(self, original_state, target_pos, target_orn, use_motion_planning=False):
        pos = self.obj_in_hand.get_position()
        p.restoreState(original_state)
        p.removeState(original_state)
        if not use_motion_planning:
            for _ in range(50):
                self.robots[0].parts['right_hand'].set_close_fraction(0)
                self.robots[0].parts['right_hand'].trigger_fraction = 0
                p.stepSimulation()

            self.robots[0].parts['right_hand'].force_release_obj()
            self.obj_in_hand.set_position_orientation(target_pos, target_orn)
            self.obj_in_hand = None

        else:
            x,y,z = target_pos
            hand_x, hand_y, hand_z = self.robots[0].parts['right_hand'].get_position()

            minx = min(x, hand_x) - 1
            miny = min(y, hand_y) - 1
            minz = min(z, hand_z) - 0.5
            maxx = max(x, hand_x) + 1
            maxy = max(y, hand_y) + 1
            maxz = max(z, hand_z) + 0.5

            state = p.saveState()
            obstacles = self.get_body_ids()
            obstacles.remove(self.obj_in_hand.body_id[0])
            plan = plan_hand_motion_br(self.robots[0], self.obj_in_hand, [x, y, z + 0.1, 0, np.pi * 5 / 6.0, 0],
                                       ((minx, miny, minz), (maxx, maxy, maxz)),
                                       obstacles=obstacles) #
            p.restoreState(state)
            p.removeState(state)

            if plan:
                for x, y, z, roll, pitch, yaw in plan:
                    self.robots[0].parts['right_hand'].move([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
                    p.stepSimulation()
                released_obj = self.obj_in_hand
                self.obj_in_hand = None
                for _ in range(50):
                    self.robots[0].parts['right_hand'].set_close_fraction(0)
                    self.robots[0].parts['right_hand'].trigger_fraction = 0
                    p.stepSimulation()

                self.robots[0].parts['right_hand'].force_release_obj()
                self.robots[0].set_position_orientation(self.robots[0].get_position(), self.robots[0].get_orientation())
                # reset hand

                p.resetBaseVelocity(released_obj.get_body_id(), linearVelocity=[0,0,0], angularVelocity=[0,0,0])

                # let object fall
                for _ in range(100):
                    p.stepSimulation()

    def sample_fn(self):
        random_point = self.scene.get_random_point()
        x, y = random_point[1][:2]
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        return (x, y, theta)

    def navigate_to_obj(self, obj, use_motion_planning=False):
        # test agent positions around an obj
        # try to place the agent near the object, and rotate it to the object
        valid_position = None  # ((x,y,z),(roll, pitch, yaw))
        original_position = self.robots[0].get_position()
        original_orientation = self.robots[0].get_orientation()
        # from IPython import embed; embed()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        if isinstance(obj, URDFObject):
            distance_to_try = [0.6, 1.2, 1.8, 2.4]
            obj_pos = obj.get_position()
            for distance in distance_to_try:
                for _ in range(20):
                    # p.restoreState(state_id)
                    yaw = np.random.uniform(-np.pi, np.pi)
                    pos = [obj_pos[0] + distance * np.cos(yaw), obj_pos[1] + distance * np.sin(yaw), 0.7]
                    orn = [0,0,yaw-np.pi]
                    self.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
                    #from IPython import embed; embed()
                    eye_pos = self.robots[0].parts['eye'].get_position()
                    obj_pos = obj.get_position()
                    ray_test_res = p.rayTest(eye_pos, obj_pos)
                    blocked = False
                    if len(ray_test_res) > 0 and ray_test_res[0][0] != obj.get_body_id():
                        blocked = True

                    valid_room = True
                    in_rooms = obj.in_rooms
                    if len(in_rooms) == 0:
                        valid_room = False
                    else:
                        in_room = in_rooms[0]
                        xy = np.array([pos[0], pos[1]])
                        print(self.scene.get_room_instance_by_point(xy), in_room)
                        if not self.scene.get_room_instance_by_point(xy) == in_room:
                            valid_room = False

                    if not detect_robot_collision(self.robots[0]) and not blocked and valid_room:
                        valid_position = (pos, orn)
                        break
                if valid_position is not None:
                    break
        else:
            for _ in range(60):
                _, pos = obj.scene.get_random_point_by_room_instance(
                    obj.room_instance)
                yaw = np.random.uniform(-np.pi, np.pi)
                orn = [0,0,yaw]
                self.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
                if not detect_robot_collision(self.robots[0]):
                    valid_position = (pos, orn)
                    break
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)


        if valid_position is not None:
            box_planning_domain = True
            if box_planning_domain:
                target_x = valid_position[0][0]
                target_y = valid_position[0][1]
                x = original_position[0]
                y = original_position[1]
                minx = min(x, target_x) - 1
                miny = min(y, target_y) - 1
                maxx = max(x, target_x) + 1
                maxy = max(y, target_y) + 1


            if use_motion_planning:
                self.robots[0].set_position_orientation(original_position, original_orientation)
                #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
                plan = plan_base_motion_br(robot=self.robots[0],
                                           end_conf=[valid_position[0][0], valid_position[0][1], valid_position[1][2]],
                                           base_limits=[(minx,miny), (maxx,maxy)],
                                           obstacles=self.get_body_ids(),
                                           override_sample_fn=self.sample_fn)
                #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)

                if plan is not None:
                    dry_run_base_plan(robot=self.robots[0],
                                      plan=plan
                                      )
                    return True
                else:
                    self.robots[0].set_position_orientation(original_position, original_orientation)
                    return False
            else:
                self.robots[0].set_position_orientation(valid_position[0], p.getQuaternionFromEuler(valid_position[1]))
                return True
        else:
            self.robots[0].set_position_orientation(original_position, original_orientation)
            return False


    def reset(self, resample_objects=False):
        obs = super(BehaviorMPEnv, self).reset(resample_objects=resample_objects)
        self.obj_in_hand = None
        self.robots[0].obj_in_hand = None
        self.robots[0].parts['right_hand'].set_close_fraction(0)
        self.robots[0].parts['right_hand'].trigger_fraction = 0
        self.robots[0].parts['right_hand'].force_release_obj()
        return obs
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default = 'gibson2/examples/configs/behavior.yaml',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui', 'pbgui'],
                        default='gui',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    env = BehaviorMPEnv(config_file=args.config,
                      mode=args.mode,
                      action_timestep=1.0 / 300.0,
                      physics_timestep=1.0 / 300.0,
                      use_motion_planning=True)
    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        env.reset()

        env.robots[0].set_position_orientation([0,0,0.7], [0,0,0,1])
        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(reward, info)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
