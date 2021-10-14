import os
import time
from math import ceil

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

import igibson
from igibson.external.motion.motion_planners.rrt_connect import birrt
from igibson.external.pybullet_tools.utils import PI, circular_difference, direct_path, pairwise_collision
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots.behavior_robot import BODY_OFFSET_FROM_FLOOR, BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config

SEARCHED = []

MAX_DISTANCE = 0  # Setting this higher unfortunately causes things to become impossible to pick up
# (they touch their hosts)


def plan_base_motion_br(
    robot: BehaviorRobot,
    obj_in_hand,
    end_conf,
    sample_fn,
    obstacles=[],
    direct=False,
    resolution=0.05,
    max_distance=MAX_DISTANCE,
    iterations=200,
    restarts=5,
    smoothing=500,
    **kwargs
):
    difference_fn = lambda q1, q2: np.array(q2) - np.array(q1)
    distance_fn = lambda q1, q2: np.linalg.norm(difference_fn(q1, q2))
    obstacles = set(obstacles)
    if obj_in_hand is not None:
        obstacles -= {obj_in_hand.get_body_id()}

    def extend_fn(q1, q2):
        aq1 = np.array(q1)
        aq2 = np.array(q2)
        diff = aq2 - aq1
        dist = np.linalg.norm(diff)
        steps = int(ceil(dist / resolution))
        for i in range(steps):
            dist_ratio = (i + 1) / steps
            q = tuple(aq1 + dist_ratio * diff)
            yield q

    body_ids = []
    for part in ["body", "left_hand", "right_hand"]:
        body_ids.append(robot.parts[part].get_body_id())

    def collision_fn(q):
        for rot in np.linspace(-np.pi, np.pi, 8, endpoint=False):
            robot.set_position_orientation([q[0], q[1], BODY_OFFSET_FROM_FLOOR], p.getQuaternionFromEuler((0, 0, rot)))
            collisions = [
                (obs, pairwise_collision(body_id, obs, max_distance=max_distance))
                for obs in obstacles
                for body_id in body_ids
            ]
            colliding_bids = [obs for obs, col in collisions if col]
            if colliding_bids:
                return True

        # The below code is useful for plotting the RRT tree.
        # SEARCHED.append(np.flip(scene.world_to_map((q[0], q[1]))))
        #
        # fig = plt.figure()
        # plt.imshow(scene.floor_map[0])
        # plt.scatter(*zip(*SEARCHED), 5)
        # fig.canvas.draw()
        #
        # # Convert the canvas to image
        # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # plt.close(fig)
        #
        # # Convert to BGR for cv2-based viewing.
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #
        # cv2.imshow("SceneGraph", img)
        # cv2.waitKey(1)

        return False

    pos = robot.get_position()
    start_conf = tuple(pos[:2])
    # if collision_fn(start_conf):
    #     print("Warning: initial configuration is in collision")
    #     return None
    if collision_fn(end_conf):
        print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(
        start_conf,
        end_conf,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        iterations=iterations,
        restarts=restarts,
        smooth=smoothing,
    )


def hand_difference_fn(q2, q1):
    dx, dy, dz = np.array(q2[:3]) - np.array(q1[:3])
    droll = circular_difference(q2[3], q1[3])
    dpitch = circular_difference(q2[4], q1[4])
    dyaw = circular_difference(q2[5], q1[5])
    return (dx, dy, dz, droll, dpitch, dyaw)


def hand_distance_fn(q1, q2, weights=(1, 1, 1, 5, 5, 5)):
    difference = np.array(hand_difference_fn(q2, q1))
    return np.sqrt(np.dot(np.array(weights), difference * difference))


def plan_hand_motion_br(
    robot: BehaviorRobot,
    obj_in_hand,
    end_conf,
    hand_limits,
    obstacles=[],
    direct=False,
    resolutions=(0.05, 0.05, 0.05, 0.2, 0.2, 0.2),
    max_distance=MAX_DISTANCE,
    iterations=200,
    restarts=5,
    smoothing=500,
):
    def sample_fn():
        x, y, z = np.random.uniform(*hand_limits)
        r, p, yaw = np.random.uniform((-PI, -PI, -PI), (PI, PI, PI))
        return (x, y, z, r, p, yaw)

    pos = robot.parts["right_hand"].get_position()
    orn = robot.parts["right_hand"].get_orientation()
    rpy = p.getEulerFromQuaternion(orn)
    start_conf = [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]

    def extend_fn(q1, q2):
        steps = np.abs(np.divide(hand_difference_fn(q2, q1), resolutions))
        n = int(np.max(steps)) + 1

        for i in range(n):
            q = ((i + 1) / float(n)) * np.array(hand_difference_fn(q2, q1)) + np.array(q1)
            q = tuple(q)
            yield q

    collision_fn = get_xyzrpy_hand_collision_fn(robot, obj_in_hand, obstacles, max_distance=MAX_DISTANCE)

    # if collision_fn(start_conf):
    #     print("Warning: initial configuration is in collision")
    #     return None
    if collision_fn(end_conf):
        print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(
        start_conf,
        end_conf,
        hand_distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        iterations=iterations,
        restarts=restarts,
        smooth=smoothing,
    )


def get_xyzrpy_hand_collision_fn(robot, obj_in_hand, obstacles, max_distance=MAX_DISTANCE):
    collision_fn = get_pose3d_hand_collision_fn(robot, obj_in_hand, obstacles, max_distance)

    def wrapper(q):
        quat = p.getQuaternionFromEuler(q[3:])
        pose3d = (q[:3], quat)
        return collision_fn(tuple(pose3d))

    return wrapper


def get_pose3d_hand_collision_fn(robot, obj_in_hand, obstacles, max_distance=MAX_DISTANCE):
    pos = robot.parts["right_hand"].get_position()
    orn = robot.parts["right_hand"].get_orientation()

    if obj_in_hand is not None:
        obj_pos = obj_in_hand.get_position()
        obj_orn = obj_in_hand.get_orientation()
        local_pos, local_orn = p.multiplyTransforms(*p.invertTransform(pos, orn), obj_pos, obj_orn)

    non_hand_non_body_obstacles = [
        obs
        for obs in obstacles
        if (
            (obj_in_hand is None or obs != obj_in_hand.get_body_id())
            and (obs != robot.parts["right_hand"].get_body_id())
        )
    ]

    def collision_fn(pose3d):
        robot.parts["right_hand"].set_position_orientation(*pose3d)
        # if obj_in_hand is not None:
        #     obj_in_hand.set_position_orientation(*p.multiplyTransforms(*pose3d, local_pos, local_orn))

        collisions = [
            (obs, pairwise_collision(robot.parts["right_hand"].get_body_id(), obs, max_distance=max_distance))
            for obs in non_hand_non_body_obstacles
        ]
        colliding_bids = [obs for obs, col in collisions if col]
        if colliding_bids:
            print("Hand collision with objects: ", colliding_bids)
        collision = bool(colliding_bids)

        if obj_in_hand is not None:
            obj_collisions = [
                (obs, pairwise_collision(obj_in_hand.get_body_id(), obs, max_distance=max_distance))
                for obs in non_hand_non_body_obstacles
            ]
            obj_colliding_bids = [obs for obs, col in obj_collisions if col]
            if obj_colliding_bids:
                print("Held object collision with objects: ", obj_colliding_bids)
            collision = collision or bool(obj_colliding_bids)

        return collision

    return collision_fn


if __name__ == "__main__":
    config = parse_config(os.path.join(igibson.example_config_path, "behavior.yaml"))
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(mode="gui", image_width=256, image_height=256, rendering_settings=settings)

    scene = EmptyScene()
    scene.objects_by_id = {}
    s.import_scene(scene, render_floor_plane=True)

    agent = BehaviorRobot(s, use_tracked_body_override=True, show_visual_head=True, use_ghost_hands=False)
    s.import_behavior_robot(agent)
    s.register_main_vr_robot(agent)
    initial_pos_z_offset = 0.7

    s.robots.append(agent)
    agent.initial_z_offset = initial_pos_z_offset
    agent.set_position_orientation([0, 0, initial_pos_z_offset], [0, 0, 0, 1])
    # plan = plan_base_motion_br(agent, [3,3,1], [(-5,-5), (5,5)])
    plan = plan_hand_motion_br(agent, [3, 3, 3, 0, 0, 0], ((-5, -5, -5), (5, 5, 5)))
    print(plan)
    for q in plan:
        agent.parts["right_hand"].set_position_orientation(
            [q[0], q[1], q[2]], p.getQuaternionFromEuler([q[3], q[4], q[5]])
        )
        time.sleep(0.05)

    for i in range(10000):
        action = np.zeros((28,))
        if i < 2:
            action[19] = 1
            action[27] = 1
        agent.apply_action(action)
        s.step()


def dry_run_base_plan(robot: BehaviorRobot, plan):
    for (x, y, yaw) in plan:
        robot.set_position_orientation([x, y, robot.initial_z_offset], p.getQuaternionFromEuler([0, 0, yaw]))
        time.sleep(0.01)


def dry_run_arm_plan(robot: BehaviorRobot, plan):
    for (x, y, z, roll, pitch, yaw) in plan:
        robot.parts["right_hand"].set_position_orientation([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
        time.sleep(0.01)
