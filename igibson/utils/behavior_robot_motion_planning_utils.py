import os
import random
import time
from math import ceil

import numpy as np
import pybullet as p

import igibson
from igibson.external.motion.motion_planners.rrt_connect import birrt
from igibson.external.pybullet_tools.utils import PI, circular_difference, direct_path, get_aabb, pairwise_collision
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots.behavior_robot import DEFAULT_BODY_OFFSET_FROM_FLOOR, BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config

SEARCHED = []
# Setting this higher unfortunately causes things to become impossible to pick up (they touch their hosts)
BODY_MAX_DISTANCE = 0.05
HAND_MAX_DISTANCE = 0


def plan_base_motion_br(
    robot: BehaviorRobot,
    obj_in_hand,
    end_conf,
    sample_fn,
    obstacles=[],
    direct=False,
    resolution=0.05,
    turn_resolution=np.pi / 6,
    max_distance=BODY_MAX_DISTANCE,
    iterations=100,
    restarts=2,
    shortening=0,
    **kwargs,
):
    distance_fn = lambda q1, q2: np.linalg.norm(np.array(q2[:2]) - np.array(q1[:2]))
    obstacles = set(obstacles)
    if obj_in_hand is not None:
        obstacles -= set(obj_in_hand.get_body_ids())

    def slippery_extender(q1, q2):
        aq1 = np.array(q1[:2])
        aq2 = np.array(q2[:2])
        diff = aq2 - aq1
        start_yaw = q1[2]
        end_yaw = q2[2]
        diff_yaw = circular_difference(end_yaw, start_yaw)

        if np.all(q1 != q2):
            dist = np.linalg.norm(diff)
            steps = int(max(ceil(dist / resolution), ceil(abs(diff_yaw) / turn_resolution)))
            for i in range(steps):
                dist_ratio = (i + 1) / steps
                extension = aq1 + dist_ratio * diff
                yaw = start_yaw + dist_ratio * diff_yaw
                yield (extension[0], extension[1], yaw)

    def turn_and_move_extender(q1, q2):
        aq1 = np.array(q1[:2])
        aq2 = np.array(q2[:2])
        diff = aq2 - aq1
        start_yaw = q1[2]
        road_yaw = np.arctan2(diff[1], diff[0])
        diff_yaw = circular_difference(road_yaw, start_yaw)

        # Do the yaw change first.
        if diff_yaw != 0:
            turn_steps = int(ceil(abs(diff_yaw) / turn_resolution))
            for i in range(turn_steps):
                turn_ratio = (i + 1) / turn_steps
                extension_yaw = start_yaw + turn_ratio * diff_yaw
                yield (aq1[0], aq1[1], extension_yaw)

        # Then do the dist change.
        dist = np.linalg.norm(diff)
        if dist != 0:
            steps = int(ceil(dist / resolution))
            for i in range(steps):
                dist_ratio = (i + 1) / steps
                extension = aq1 + dist_ratio * diff
                yield (extension[0], extension[1], road_yaw)

        # Now do another yaw change.
        end_yaw = q2[2]
        diff_yaw = circular_difference(end_yaw, road_yaw)
        if diff_yaw != 0:
            turn_steps = int(ceil(abs(diff_yaw) / turn_resolution))
            for i in range(turn_steps):
                turn_ratio = (i + 1) / turn_steps
                extension_yaw = road_yaw + turn_ratio * diff_yaw
                yield (aq2[0], aq2[1], extension_yaw)

    extend_fn = slippery_extender

    body_ids = robot.get_body_ids()

    if obj_in_hand is not None:
        body_ids.extend(obj_in_hand.get_body_ids())

    def collision_fn(q):
        robot.set_position_orientation(
            [q[0], q[1], DEFAULT_BODY_OFFSET_FROM_FLOOR], p.getQuaternionFromEuler((0, 0, q[2]))
        )
        for body_id in body_ids:
            close_objects = set(x[0] for x in p.getOverlappingObjects(*get_aabb(body_id)))
            close_obstacles = close_objects & obstacles
            collisions = [
                (obs, pairwise_collision(body_id, obs, max_distance=max_distance))
                for obs in close_obstacles
                for body_id in body_ids
            ]
            colliding_bids = [obs for obs, col in collisions if col]
            if colliding_bids:
                # indented_print("Found collision w/ ", colliding_bids)
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
    yaw = p.getEulerFromQuaternion(robot.get_orientation())[2]
    start_conf = (pos[0], pos[1], yaw)
    # if collision_fn(start_conf):
    #     indented_print("Warning: initial configuration is in collision")
    #     return None
    if collision_fn(end_conf):
        print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    path = birrt(
        start_conf,
        end_conf,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        iterations=iterations,
        restarts=restarts,
    )
    if path is None:
        return None
    return shorten_path(path, extend_fn, collision_fn, shortening)


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
    max_distance=HAND_MAX_DISTANCE,
    iterations=50,
    restarts=2,
    shortening=0,
):
    def sample_fn():
        x, y, z = np.random.uniform(*hand_limits)
        r, p, yaw = np.random.uniform((-PI, -PI, -PI), (PI, PI, PI))
        return (x, y, z, r, p, yaw)

    pos = robot.eef_links["right_hand"].get_position()
    orn = robot.eef_links["right_hand"].get_orientation()
    rpy = p.getEulerFromQuaternion(orn)
    start_conf = [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]

    def extend_fn(q1, q2):
        # TODO: Use scipy's slerp
        steps = np.abs(np.divide(hand_difference_fn(q2, q1), resolutions))
        n = int(np.max(steps)) + 1

        for i in range(n):
            q = ((i + 1) / float(n)) * np.array(hand_difference_fn(q2, q1)) + np.array(q1)
            q = tuple(q)
            yield q

    collision_fn = get_xyzrpy_hand_collision_fn(robot, obj_in_hand, obstacles, max_distance=max_distance)

    # if collision_fn(start_conf):
    #     indented_print("Warning: initial configuration is in collision")
    #     return None
    if collision_fn(end_conf):
        print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    path = birrt(
        start_conf,
        end_conf,
        hand_distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        iterations=iterations,
        restarts=restarts,
    )
    if path is None:
        return None
    return shorten_path(path, extend_fn, collision_fn, shortening)


def get_xyzrpy_hand_collision_fn(robot, obj_in_hand, obstacles, max_distance=HAND_MAX_DISTANCE):
    collision_fn = get_pose3d_hand_collision_fn(robot, obj_in_hand, obstacles, max_distance)

    def wrapper(q):
        quat = p.getQuaternionFromEuler(q[3:])
        pose3d = (q[:3], quat)
        return collision_fn(tuple(pose3d))

    return wrapper


def get_pose3d_hand_collision_fn(robot, obj_in_hand, obstacles, max_distance=HAND_MAX_DISTANCE):
    non_hand_non_oih_obstacles = {
        obs
        for obs in obstacles
        if (
            (obj_in_hand is None or obs not in obj_in_hand.get_body_ids())
            and (obs != robot.eef_links["right_hand"].body_id)  # TODO(MP): Generalize
        )
    }

    def collision_fn(pose3d):
        # TODO: Generalize
        robot.set_eef_position_orientation(*pose3d, "right_hand")
        close_objects = set(x[0] for x in p.getOverlappingObjects(*get_aabb(robot.eef_links["right_hand"].body_id)))
        close_obstacles = close_objects & non_hand_non_oih_obstacles
        collisions = [
            (obs, pairwise_collision(robot.eef_links["right_hand"].body_id, obs, max_distance=max_distance))
            for obs in close_obstacles
        ]
        colliding_bids = [obs for obs, col in collisions if col]
        if colliding_bids:
            print("Hand collision with objects: ", colliding_bids)
        collision = bool(colliding_bids)

        if obj_in_hand is not None:
            # Generalize more.
            [oih_bid] = obj_in_hand.get_body_ids()  # Generalize.
            oih_close_objects = set(x[0] for x in p.getOverlappingObjects(*get_aabb(oih_bid)))
            oih_close_obstacles = (oih_close_objects & non_hand_non_oih_obstacles) | close_obstacles
            obj_collisions = [
                (obs, pairwise_collision(oih_bid, obs, max_distance=max_distance)) for obs in oih_close_obstacles
            ]
            obj_colliding_bids = [obs for obs, col in obj_collisions if col]
            if obj_colliding_bids:
                print("Held object collision with objects: ", obj_colliding_bids)
            collision = collision or bool(obj_colliding_bids)

        return collision

    return collision_fn


def dry_run_base_plan(robot: BehaviorRobot, plan):
    for (x, y, yaw) in plan:
        robot.set_position_orientation([x, y, robot.initial_z_offset], p.getQuaternionFromEuler([0, 0, yaw]))
        time.sleep(0.01)


def dry_run_arm_plan(robot: BehaviorRobot, plan):
    for (x, y, z, roll, pitch, yaw) in plan:
        robot.set_eef_position_orientation([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]), "right_hand")
        time.sleep(0.01)


def shorten_path(path, extend, collision, iterations=50):
    shortened_path = path
    for _ in range(iterations):
        if len(shortened_path) <= 2:
            return shortened_path
        i = random.randint(0, len(shortened_path) - 1)
        j = random.randint(0, len(shortened_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(shortened_path[i], shortened_path[j]))
        if all(not collision(q) for q in shortcut):
            shortened_path = shortened_path[: i + 1] + shortened_path[j:]
    return shortened_path
