import logging

import numpy as np
import pybullet as p

from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets

download_assets()


def test_fetch():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    fetch = REGISTERED_ROBOTS["Fetch"]()
    s.import_object(fetch)
    for i in range(100):
        fetch.calc_state()
        s.step()
    s.disconnect()


def test_turtlebot():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = REGISTERED_ROBOTS["Turtlebot"]()
    s.import_object(turtlebot)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_jr2():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    jr2 = REGISTERED_ROBOTS["JR2"]()
    s.import_object(jr2)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_ant():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    ant = REGISTERED_ROBOTS["Ant"]()
    s.import_object(ant)
    ant2 = REGISTERED_ROBOTS["Ant"]()
    s.import_object(ant2)
    ant2.set_position([0, 2, 2])
    nbody = p.getNumBodies()
    for i in range(100):
        s.step()
    s.disconnect()
    assert nbody == 6


def test_husky():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    husky = REGISTERED_ROBOTS["Husky"]()
    s.import_object(husky)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_turtlebot_position():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = REGISTERED_ROBOTS["Turtlebot"]()
    s.import_object(turtlebot)

    turtlebot.set_position([0, 0, 5])

    nbody = p.getNumBodies()
    pos = turtlebot.get_position()
    s.disconnect()
    assert nbody == 5
    assert np.allclose(pos, np.array([0, 0, 5]))


def test_multiagent():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot1 = REGISTERED_ROBOTS["Turtlebot"]()
    turtlebot2 = REGISTERED_ROBOTS["Turtlebot"]()
    turtlebot3 = REGISTERED_ROBOTS["Turtlebot"]()

    s.import_object(turtlebot1)
    s.import_object(turtlebot2)
    s.import_object(turtlebot3)

    turtlebot1.set_position([1, 0, 0.5])
    turtlebot2.set_position([0, 0, 0.5])
    turtlebot3.set_position([-1, 0, 0.5])

    nbody = p.getNumBodies()
    for i in range(100):
        s.step()

    s.disconnect()
    assert nbody == 7


def show_action_sensor_space():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)

    turtlebot = REGISTERED_ROBOTS["Turtlebot"]()
    s.import_object(turtlebot)
    turtlebot.set_position([0, 1, 0.5])

    ant = REGISTERED_ROBOTS["Ant"]()
    s.import_object(ant)
    ant.set_position([0, 2, 0.5])

    jr = REGISTERED_ROBOTS["JR2"]()
    s.import_object(jr)
    jr.set_position([0, 4, 0.5])

    jr2 = REGISTERED_ROBOTS["JR2"]()
    s.import_object(jr2)
    jr2.set_position([0, 5, 0.5])

    husky = REGISTERED_ROBOTS["Husky"]()
    s.import_object(husky)
    husky.set_position([0, 6, 0.5])

    for robot in scene.robots:
        print(type(robot), len(robot.joints), robot.calc_state().shape)

    for i in range(100):
        s.step()

    s.disconnect()


def test_behavior_robot():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)

    robot = REGISTERED_ROBOTS["BehaviorRobot"](higher_limits=True)  # Tests are intended for VR version of limits.
    s.import_object(robot)

    parts_and_controllers = [
        ("eye", "camera"),
        ("left_hand", "arm_left_hand"),
        ("right_hand", "arm_right_hand"),
    ]

    # Rotate the robot CCW 90 degrees, and each of the limbs a further CCW 90 degrees. This makes the X of each outer
    # frame align with the Y axis of the inner one.
    BODY_POS = np.array([10, 20, 0.7])
    robot.set_position_orientation(BODY_POS, p.getQuaternionFromEuler([0, 0, np.pi / 2]))
    for part_name, _ in parts_and_controllers:
        part = robot._parts[part_name]
        part_pos = part.get_position()
        part.set_position_orientation(part_pos, p.getQuaternionFromEuler([0, 0, np.pi]))

    # Now we'll take some steps to settle things down.
    for _ in range(50):
        robot.apply_action(np.zeros(robot.action_dim))
        s.step()

    # # ============================ PART 1 : BODY DELTA POSITION ============================
    action = np.zeros(robot.action_dim)
    action[robot.controller_action_idx["base"]] = [0, 0, 0, 0, 0, 0]

    # Execute.
    robot.apply_action(action)
    for _ in range(10):
        s.step()

    # Check that the body has moved where we expected it to.
    EXPECTED_BODY_DELTA = np.array([0, 0, 0])
    assert np.all(np.isclose(robot.get_position(), BODY_POS + EXPECTED_BODY_DELTA, atol=1e-2))

    # ============================ PART 2 : LIMB DELTA POSITION ============================
    action = np.zeros(robot.action_dim)
    for part_name, controller_name in parts_and_controllers:
        action[robot.controller_action_idx[controller_name]] = [0, 0, 0, 0, 0, 0]

    # Execute.
    robot.apply_action(action)
    for _ in range(10):
        s.step()

    EXPECTED_PART_DELTA = np.array([0, 0, 0])

    # The constants below are the EYE_LOC_POSE_TRACKED etc constants from behavior_robot, rotated CCW 90deg.
    assert np.all(
        np.isclose(
            robot._parts["eye"].get_position(),
            BODY_POS + [0, 0.05, 0.4] + EXPECTED_BODY_DELTA + EXPECTED_PART_DELTA,  # rotated EYE_LOC_POSE_TRACKED
            atol=1e-2,
        )
    )
    assert np.all(
        np.isclose(
            robot._parts["right_hand"].get_position(),
            BODY_POS
            + [0.12, 0.1, 0.05]
            + EXPECTED_BODY_DELTA
            + EXPECTED_PART_DELTA,  # rotated RIGHT_HAND_LOC_POSE_TRACKED
            atol=1e-2,
        )
    )
    assert np.all(
        np.isclose(
            robot._parts["left_hand"].get_position(),
            BODY_POS
            + [-0.12, 0.1, 0.05]
            + EXPECTED_BODY_DELTA
            + EXPECTED_PART_DELTA,  # rotated LEFT_HAND_LOC_POSE_TRACKED
            atol=1e-2,
        )
    )

    # ============================ PART 3 : LIMB DELTA ORIENTATION ============================
    # Try rotating the hand CCW 90deg to make sure that this rotation happens in the base frame (and not world/hand).
    action = np.zeros(robot.action_dim)
    action[robot.controller_action_idx["arm_right_hand"]] = [0, 0, 0, 0, 0, np.pi / 4]
    robot.apply_action(action)
    for _ in range(10):
        s.step()

    action = np.zeros(robot.action_dim)
    action[robot.controller_action_idx["arm_right_hand"]] = [0, 0, 0, 0, 0, np.pi / 4]
    robot.apply_action(action)
    for _ in range(10):
        s.step()

    action = np.zeros(robot.action_dim)
    action[robot.controller_action_idx["arm_right_hand"]] = [0, 0, 0, 0, np.pi / 4, 0]
    robot.apply_action(action)
    for _ in range(10):
        s.step()

    action = np.zeros(robot.action_dim)
    action[robot.controller_action_idx["arm_right_hand"]] = [0, 0, 0, 0, np.pi / 4, 0]
    robot.apply_action(action)
    for _ in range(10):
        s.step()

    EXPECTED_HAND_ROTATION = p.getQuaternionFromEuler([-np.pi / 2, -np.pi / 2, 0])
    assert np.all(np.isclose(robot._parts["right_hand"].get_orientation(), EXPECTED_HAND_ROTATION, atol=1e-2))


def test_behavior_robot_stability():
    s = Simulator(physics_timestep=1 / 30, render_timestep=1 / 30, mode="headless")
    scene = EmptyScene()
    s.import_scene(scene)

    robot = REGISTERED_ROBOTS["BehaviorRobot"](higher_limits=True)  # Tests are intended for VR version of limits.
    s.import_object(robot)

    BODY_POS = np.array([0, 0, 0.7])
    robot.set_position_orientation(BODY_POS, p.getQuaternionFromEuler([0, 0, np.pi / 2]))

    # Now we'll take some steps to settle things down.
    for _ in range(50):
        robot.apply_action(np.zeros(robot.action_dim))
        s.step()

    # Save our existing pose.
    cached_local_poses = {}
    for part_name, part in robot._parts.items():
        if part_name == "body":
            continue
        cached_local_poses[part_name] = part.get_local_position_orientation()

    # Now we'll spin the robot. We'll keep adding 1 degree yaw and moving forward by 0.1m.
    action = np.zeros(robot.action_dim)
    action[robot.controller_action_idx["base"]] = [0.1, 0, 0, 0, 0, np.pi / 180]

    # We keep doing this for 1800 timesteps.
    for _ in range(1800):
        robot.apply_action(action)
        s.step()

    # Check if the robot is still upright (tilt is less than 1 deg)
    rot_axis, angle = p.getAxisAngleFromQuaternion(robot.get_orientation())
    assert np.linalg.norm(np.cross([0, 0, 1], rot_axis) < np.sin(np.deg2rad(1)))

    # Check if the robot is still facing near the Y+ axis.
    facing = p.multiplyTransforms([0, 0, 0], robot.get_orientation(), [1, 0, 0], [0, 0, 0, 1])[0]
    assert np.dot(facing, [0, 1, 0]) > np.cos(np.deg2rad(1))

    # Now we check if the parts are still where they are supposed to be.
    for part_name, part in robot._parts.items():
        if part_name == "body":
            continue
        new_local_pos, new_local_orn = part.get_local_position_orientation()
        assert np.allclose(
            new_local_pos, cached_local_poses[part_name][0], atol=1e-2
        ), "Part position moved relative to robot."

        _, rel_orn = p.multiplyTransforms(
            *p.invertTransform([0, 0, 0], cached_local_poses[part_name][1]), [0, 0, 0], new_local_orn
        )
        _, angle = p.getAxisAngleFromQuaternion(rel_orn)
        assert angle < np.deg2rad(5), "Part orientation moved relative to robot."


def test_behavior_robot_legacy_proprioception():
    s = Simulator(physics_timestep=1 / 30, render_timestep=1 / 30, mode="headless")
    scene = EmptyScene()
    s.import_scene(scene)

    robot = REGISTERED_ROBOTS["BehaviorRobot"](legacy_proprioception=True)
    s.import_object(robot)

    proprio_length = robot.get_proprioception().shape
    assert proprio_length == (22,), f"Proprioception shape should be 22. Instead, it is {proprio_length}"
