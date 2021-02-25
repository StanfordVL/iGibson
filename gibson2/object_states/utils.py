import numpy as np
import pybullet as p
import cv2
from gibson2.external.pybullet_tools.utils import get_link_pose, matrix_from_quat, get_aabb_center, get_aabb_extent, quat_from_matrix, stable_z_on_aabb
from gibson2.object_states.object_state_base import CachingEnabledObjectState
from IPython import embed


def get_center_extent(obj_states):
    assert 'aabb' in obj_states
    aabb = obj_states['aabb'].get_value()
    center, extent = get_aabb_center(aabb), get_aabb_extent(aabb)
    return center, extent


def clear_cached_states(obj):
    for _, obj_state in obj.states.items():
        if isinstance(obj_state, CachingEnabledObjectState):
            obj_state.clear_cached_value()


def sample_kinematics(predicate, objA, objB, binary_state):
    if not binary_state:
        raise NotImplementedError()

    if predicate not in objB.supporting_surfaces:
        return False

    max_trials = 100
    z_offset = 0.05
    if objA.orientations is not None:
        orientation = objA.sample_orientation()
    else:
        orientation = [0, 0, 0, 1]

    old_pos, old_orn = objA.get_position_orientation()

    # objBorientation = objB.get_orientation()
    # orientation = quat_from_matrix(
    #     matrix_from_quat(objBorientation) @ matrix_from_quat(orientation))

    objA.force_wakeup()
    objB.force_wakeup()

    state_id = p.saveState()
    for i in range(max_trials):
        random_idx = np.random.randint(
            len(objB.supporting_surfaces[predicate].keys()))
        body_id, link_id = list(objB.supporting_surfaces[predicate].keys())[
            random_idx]
        random_height_idx = np.random.randint(
            len(objB.supporting_surfaces[predicate][(body_id, link_id)]))
        height, height_map = objB.supporting_surfaces[predicate][(
            body_id, link_id)][random_height_idx]
        obj_half_size = np.max(objA.bounding_box) / 2 * 100
        obj_half_size_scaled = np.array(
            [obj_half_size / objB.scale[1], obj_half_size / objB.scale[0]])
        obj_half_size_scaled = np.ceil(obj_half_size_scaled).astype(np.int)
        height_map_eroded = cv2.erode(
            height_map, np.ones(obj_half_size_scaled, np.uint8))

        valid_pos = np.array(height_map_eroded.nonzero())
        if valid_pos.shape[1] == 0:
            return False
        random_pos_idx = np.random.randint(valid_pos.shape[1])
        random_pos = valid_pos[:, random_pos_idx]
        y_map, x_map = random_pos
        y = y_map / 100.0 - 2
        x = x_map / 100.0 - 2
        z = height

        pos = np.array([x, y, z])
        pos *= objB.scale

        # the supporting surface is defined w.r.t to the link frame, not
        # the inertial frame
        if link_id == -1:
            link_pos, link_orn = p.getBasePositionAndOrientation(body_id)
            dynamics_info = p.getDynamicsInfo(body_id, -1)
            inertial_pos = dynamics_info[3]
            inertial_orn = dynamics_info[4]
            inv_inertial_pos, inv_inertial_orn =\
                p.invertTransform(inertial_pos, inertial_orn)
            link_pos, link_orn = p.multiplyTransforms(
                link_pos, link_orn, inv_inertial_pos, inv_inertial_orn)
        else:
            link_pos, link_orn = get_link_pose(body_id, link_id)
        pos = matrix_from_quat(link_orn).dot(pos) + np.array(link_pos)

        pos[2] += z_offset

        z = stable_z_on_aabb(
            objA.get_body_id(), ([0, 0, pos[2]], [0, 0, pos[2]]))

        pos[2] = z
        objA.set_position_orientation(pos, orientation)

        p.stepSimulation()
        success = len(p.getContactPoints(objA.get_body_id())) == 0
        p.restoreState(state_id)

        if success:
            break

    p.removeState(state_id)

    if success:
        objA.set_position_orientation(pos, orientation)
        # Let it fall for 0.2 second
        physics_timestep = p.getPhysicsEngineParameters()['fixedTimeStep']
        for _ in range(int(0.2 / physics_timestep)):
            p.stepSimulation()
            if len(p.getContactPoints(bodyA=objA.get_body_id())) > 0:
                break
        return True
    else:
        # move back so it's not in scene anymore
        objA.set_position_orientation(old_pos, old_orn)
        return False
