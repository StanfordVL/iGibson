"""
This demo shows how to reproduce the issue
"""
import os
import time

import numpy as np
import pybullet as p
import pybullet_data
from IPython import embed

import igibson
from igibson.external.pybullet_tools.utils import get_joint_info, get_joint_state, get_link_info, get_link_name
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.robots.vladimir_robot import Vladimir
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.constants import SemanticClass


def main():
    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=False,
        fullscreen=False,
        enable_shadow=True,
        enable_pbr=True,
        msaa=True,
        light_dimming_factor=1.0,
    )
    s = Simulator(
        mode="gui_interactive",
        rendering_settings=vr_rendering_settings,
        image_height=512,
        image_width=512,
        use_pb_gui=True,
    )

    scene = EmptyScene()

    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    vr_agent = BehaviorRobot(s, show_visual_head=True)
    s.import_robot(vr_agent)
    vr_agent.set_position_orientation([0, 2, 0.5], [0, 0, 0, 1])
    vr_agent.activate()
    vr_agent.apply_action(np.zeros(28))

    vladimir = Vladimir(s, show_visual_head=True, use_ghost_hands=False, use_tracked_body=True)
    s.import_object(vladimir)
    vladimir.set_position_orientation([0, 0, 0.5], [0, 0, 0, 1])
    # vladimir.activate()
    vladimir.apply_action(np.zeros(28))

    v = vladimir.get_body_id()
    for j in range(p.getNumJoints(v)):
        p.enableJointForceTorqueSensor(v, j, True)

    for j in range(p.getNumJoints(v)):
        ji = get_joint_info(v, j)
        jp = get_joint_state(v, j).jointPosition
        vio = jp < ji.jointLowerLimit or jp > ji.jointUpperLimit
        print(
            "Joint %s. Violation: %s. Position: %.4f. Range: %.4f to %.4f."
            % (ji.jointName.decode("utf-8"), vio, jp, ji.jointLowerLimit, ji.jointUpperLimit)
        )

    # Get corresponding pairs of joints and compare their dynamics.
    joints_vlad = vladimir.parts["left_hand"].all_joints
    joints_vlad_data = {}
    for j in joints_vlad:
        ji = get_joint_info(vladimir.get_body_id(), j)
        di = p.getDynamicsInfo(vladimir.get_body_id(), j)
        js = get_joint_state(vladimir.get_body_id(), j)
        joints_vlad_data[ji.jointName.decode("utf-8").replace("lh_", "")] = (ji, di, js)

    joints_br = range(p.getNumJoints(vr_agent.links["left_hand"].get_body_id()))
    joints_br_data = {}
    for j in joints_br:
        ji = get_joint_info(vr_agent.links["left_hand"].get_body_id(), j)
        di = p.getDynamicsInfo(vr_agent.links["left_hand"].get_body_id(), j)
        js = p.getJointState(vr_agent.links["left_hand"].get_body_id(), j)
        joints_br_data[ji.jointName.decode("utf-8")] = (ji, di, js)

    assert set(joints_vlad_data.keys()) == set(joints_br_data.keys())

    for key in joints_vlad_data:
        vlad_data = joints_vlad_data[key]
        br_data = joints_br_data[key]

        # Compare the dynamics info
        assert all(x == y for x, y in zip(vlad_data[1], br_data[1]))

        # Compare the joint state
        assert all(x == y for x, y in zip(vlad_data[2], br_data[2]))

    while True:
        action = np.zeros(28)
        # action[5] = -0.01  # rotate body
        # action[11] = -0.01  # rotate head only
        # vr_agent.apply_action(action)
        vladimir.apply_action(action)
        print("+" * 50)
        for j in range(p.getNumJoints(v)):
            ji = get_joint_info(v, j)
            print(get_link_name(v, j), p.getDynamicsInfo(v, j)[0])

            if not ji.jointName.decode("utf-8").startswith("head"):
                continue

            if ji.jointType == p.JOINT_SPHERICAL:
                js = p.getJointStateMultiDof(v, j)
                print(ji.jointName, ":", js[2])
            else:
                js = p.getJointState(v, j)
                print(ji.jointName, ":", js[2])

        s.step()
        # print(vr_agent.parts["body"].get_orientation())
        # print(vr_agent.parts["eye"].get_orientation())

    s.disconnect()


if __name__ == "__main__":
    main()
