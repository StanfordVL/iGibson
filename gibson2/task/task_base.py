import numpy as np
import os

import tasknet as tn
from tasknet.task_base import TaskNetTask
import gibson2
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.objects.articulated_object import URDFObject, ArticulatedObject
from gibson2.external.pybullet_tools.utils import *
from gibson2.utils.constants import NON_SAMPLEABLE_OBJECTS
from gibson2.utils.assets_utils import get_ig_category_path, get_ig_model_path, get_ig_avg_category_specs
import random
import pybullet as p
import cv2


class iGTNTask(TaskNetTask):
    def __init__(self, atus_activity, task_instance=0):
        '''
        Initialize simulator with appropriate scene and sampled objects.
        :param atus_activity: string, official ATUS activity label
        :param task_instance: int, specific instance of atus_activity init/final conditions
                                   optional, randomly generated if not specified
        '''
        super().__init__(atus_activity,
                         task_instance=task_instance,
                         scene_path=os.path.join(gibson2.ig_dataset_path, 'scenes'))

    def initialize_simulator(self,
                             mode='iggui',
                             scene_id=None,
                             simulator=None):
        '''
        Get scene populated with objects such that scene satisfies initial conditions
        :param simulator: Simulator class, populated simulator that should completely
                                   replace this function. Use if you would like to bypass internal
                                   Simulator instantiation and population based on initial conditions
                                   and use your own. Warning that if you use this option, we cannot
                                   guarantee that the final conditions will be reachable.
        '''
        # Set self.scene_name, self.scene, self.sampled_simulator_objects, and self.sampled_dsl_objects
        if simulator is None:
            self.simulator = Simulator(
                mode=mode, image_width=960, image_height=720, device_idx=0)
            self.initialize(InteractiveIndoorScene,
                            scene_id=scene_id)
        else:
            print('INITIALIZING TASK WITH PREDEFINED SIMULATOR')
            self.simulator = simulator
            self.initialize(InteractiveIndoorScene,
                            scene_id=scene_id)

    def check_scene(self):
        for obj_cat in self.objects:
            if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                continue
            if obj_cat not in self.scene.objects_by_category or \
                    len(self.objects[obj_cat]) > len(self.scene.objects_by_category[obj_cat]):
                return False

        room_type_to_obj_inst = {}
        for cond in self.parsed_initial_conditions:
            if cond[0] == 'inroom':
                obj_inst, room_type = cond[1], cond[2]

                # Room type missing in the scene
                if room_type not in self.scene.room_sem_name_to_ins_name:
                    return False

                if room_type not in room_type_to_obj_inst:
                    room_type_to_obj_inst[room_type] = []

                room_type_to_obj_inst[room_type].append(obj_inst)

        selected_obj_names = set()
        for room_type in room_type_to_obj_inst:
            room_type_success = False
            # Loop through all instances of the room, e.g. bedroom_0, bedroom_1
            for room_inst in self.scene.room_sem_name_to_ins_name[room_type]:
                tmp_scope = {}
                tmp_selected_obj_names = set()
                room_inst_success = True
                room_objs = self.scene.objects_by_room[room_inst]
                # Try to assign obj instances to this room instance
                for obj_inst in room_type_to_obj_inst[room_type]:
                    obj_cat = self.obj_inst_to_obj_cat[obj_inst]
                    # Assume inroom relationship can only be defined w.r.t non-sampleable object
                    assert obj_cat in NON_SAMPLEABLE_OBJECTS
                    room_objs_of_cat = [
                        obj for obj in room_objs
                        if obj.category == obj_cat
                        and obj.name not in tmp_selected_obj_names]
                    if len(room_objs_of_cat) == 0:
                        room_inst_success = False
                        break

                    selected_obj = np.random.choice(room_objs_of_cat)
                    tmp_scope[obj_inst] = selected_obj
                    tmp_selected_obj_names.add(selected_obj.name)

                # If successful, permanently assign object scope
                if room_inst_success:
                    for obj_inst in tmp_scope:
                        self.object_scope[obj_inst] = tmp_scope[obj_inst]
                        selected_obj_names.add(tmp_scope[obj_inst].name)
                    room_type_success = True
                    break

            # Fail to assign obj instances to any room instance;
            # Hence, initial condition cannot be fulfilled
            if not room_type_success:
                return False

        avg_category_spec = get_ig_avg_category_specs()
        for obj_cat in self.objects:
            if obj_cat in NON_SAMPLEABLE_OBJECTS:
                # Remaining non-sampleable objects that have NOT been sampled
                # in the previous room assignment phase
                obj_inst_remain = [
                    obj_inst for obj_inst in self.objects[obj_cat]
                    if self.object_scope[obj_inst] is None]

                obj_remain = [
                    obj for obj in self.scene.objects_by_category[obj_cat]
                    if obj.name not in selected_obj_names
                ]

                simulator_objs = np.random.choice(
                    obj_remain, len(obj_inst_remain), replace=False)
                for obj_inst, simulator_obj in \
                        zip(obj_inst_remain, simulator_objs):
                    self.object_scope[obj_inst] = simulator_obj
            else:
                category_path = get_ig_category_path(obj_cat)
                for i, obj_inst in enumerate(self.objects[obj_cat]):
                    model = random.choice(os.listdir(category_path))
                    model_path = get_ig_model_path(obj_cat, model)
                    filename = os.path.join(model_path, model + ".urdf")
                    obj_name = '{}_{}'.format(
                        obj_cat,
                        len(self.scene.objects_by_category.get(obj_cat, [])))
                    simulator_obj = URDFObject(
                        filename,
                        name=obj_name,
                        category=obj_cat,
                        model_path=model_path,
                        avg_obj_dims=avg_category_spec.get(obj_cat),
                        fit_avg_dim_volume=True,
                        texture_randomization=False,
                        overwrite_inertial=True)
                    self.scene.add_object(simulator_obj)
                    self.object_scope[obj_inst] = simulator_obj

        return True

    def import_scene(self):
        self.simulator.reload()
        self.simulator.import_ig_scene(self.scene)

    def sample(self, failed_conditions):
        failed_conditions = [cond.children[0] for cond in failed_conditions]
        # TODO: assume initial condition is always true
        for failed_condition in failed_conditions:
            success = failed_condition.sample(binary_state=True)
            if not success:
                return False
        return True

    #### CHECKERS ####
    def onTop(self, objA, objB):
        '''
        Checks if one object is on top of another.
        True iff the (x, y) coordinates of objA's AABB center are within the (x, y) projection
            of objB's AABB, and the z-coordinate of objA's AABB lower is within a threshold around
            the z-coordinate of objB's AABB upper, and objA and objB are touching.
        :param objA: simulator object
        :param objB: simulator object
        '''
        below_epsilon, above_epsilon = 0.025, 0.025

        center, extent = get_center_extent(
            objA.get_body_id())  # TODO: approximate_as_prism
        bottom_aabb = get_aabb(objB.get_body_id())

        base_center = center - np.array([0, 0, extent[2]])/2
        top_z_min = base_center[2]
        bottom_z_max = bottom_aabb[1][2]
        height_correct = (bottom_z_max - abs(below_epsilon)
                          ) <= top_z_min <= (bottom_z_max + abs(above_epsilon))
        bbox_contain = (aabb_contains_point(
            base_center[:2], aabb2d_from_aabb(bottom_aabb)))
        touching = body_collision(objA.get_body_id(), objB.get_body_id())

        return height_correct and bbox_contain and touching
        # return is_placement(objA.get_body_id(), objB.get_body_id()) or is_center_stable(objA.get_body_id(), objB.get_body_id())

    def inside(self, objA, objB):
        '''
        Checks if one object is inside another.
        True iff the AABB of objA does not extend past the AABB of objB TODO this might not be the right spec anymore
        :param objA: simulator object
        :param objB: simulator object
        '''
        # return aabb_contains_aabb(get_aabb(objA.get_body_id()), get_aabb(objB.get_body_id()))
        aabbA, aabbB = get_aabb(
            objA.get_body_id()), get_aabb(objB.get_body_id())
        center_inside = aabb_contains_point(get_aabb_center(aabbA), aabbB)
        volume_lesser = get_aabb_volume(aabbA) < get_aabb_volume(aabbB)
        extentA, extentB = get_aabb_extent(aabbA), get_aabb_extent(aabbB)
        two_dimensions_lesser = np.sum(np.less_equal(extentA, extentB)) >= 2
        above = center_inside and aabbB[1][2] <= aabbA[0][2]
        return (center_inside and volume_lesser and two_dimensions_lesser) or above

    def nextTo(self, objA, objB):
        '''
        Checks if one object is next to another.
        True iff the distance between the objects is TODO less than 2/3 of the average
                 side length across both objects' AABBs
        :param objA: simulator object
        :param objB: simulator object
        '''
        objA_aabb, objB_aabb = get_aabb(
            objA.get_body_id()), get_aabb(objB.get_body_id())
        objA_lower, objA_upper = objA_aabb
        objB_lower, objB_upper = objB_aabb
        distance_vec = []
        for dim in range(3):
            glb = max(objA_lower[dim], objB_lower[dim])
            lub = min(objA_upper[dim], objB_upper[dim])
            distance_vec.append(max(0, glb - lub))
        distance = np.linalg.norm(np.array(distance_vec))
        objA_dims = objA_upper - objA_lower
        objB_dims = objB_upper - objB_lower
        avg_aabb_length = np.mean(objA_dims + objB_dims)

        return distance <= (avg_aabb_length * (1./6.))  # TODO better function

    def under(self, objA, objB):
        '''
        Checks if one object is underneath another.
        True iff the (x, y) coordinates of objA's AABB center are within the (x, y) projection
                 of objB's AABB, and the z-coordinate of objA's AABB upper is less than the
                 z-coordinate of objB's AABB lower.
        :param objA: simulator object
        :param objB: simulator object
        '''
        within = aabb_contains_point(
            get_aabb_center(get_aabb(objA.get_body_id()))[:2],
            aabb2d_from_aabb(get_aabb(objB.get_body_id())))
        objA_aabb = get_aabb(objA.get_body_id())
        objB_aabb = get_aabb(objB.get_body_id())

        within = aabb_contains_point(
            get_aabb_center(objA_aabb)[:2],
            aabb2d_from_aabb(objB_aabb))
        below = objA_aabb[1][2] <= objB_aabb[0][2]
        return within and below

    def touching(self, objA, objB):
        return body_collision(objA.get_body_id(), objB.get_body_id())

    #### SAMPLERS ####
    def sampleOnTop(self, objA, objB):
        return self.sampleOnTopOrInside(objA, objB, 'onTop')

    def sampleOnTopOrInside(self, objA, objB, predicate):
        if predicate not in objB.supporting_surfaces:
            return False

        max_trials = 100
        z_offset = 0.01
        objA.set_position_orientation([100, 100, 100], [0, 0, 0, 1])
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
            obj_half_size = np.max(objA.bounding_box[0:2]) / 2 * 100
            obj_half_size_scaled = np.array(
                [obj_half_size / objB.scale[1], obj_half_size / objB.scale[0]])
            obj_half_size_scaled = np.ceil(obj_half_size_scaled).astype(np.int)
            height_map_eroded = cv2.erode(
                height_map, np.ones(obj_half_size_scaled, np.uint8))

            valid_pos = np.array(height_map_eroded.nonzero())
            random_pos_idx = np.random.randint(valid_pos.shape[1])
            random_pos = valid_pos[:, random_pos_idx]
            y_map, x_map = random_pos
            y = y_map / 100.0 - 2
            x = x_map / 100.0 - 2
            z = height

            pos = np.array([x, y, z])
            pos *= objB.scale

            link_pos, link_orn = get_link_pose(body_id, link_id)
            pos = matrix_from_quat(link_orn).dot(pos) + np.array(link_pos)

            pos[2] += z_offset
            z = stable_z_on_aabb(
                objA.get_body_id(), ([0, 0, pos[2]], [0, 0, pos[2]]))
            pos[2] = z
            objA.set_position_orientation(pos, [0, 0, 0, 1])

            p.stepSimulation()
            success = len(p.getContactPoints(objA.get_body_id())) == 0
            p.restoreState(state_id)

            if success:
                break

        if success:
            objA.set_position_orientation(pos, [0, 0, 0, 1])
            # Let it fall for 0.1 second
            for _ in range(int(0.1 / self.simulator.physics_timestep)):
                p.stepSimulation()
                if self.touching(objA, objB):
                    break
            return True
        else:
            return False

    def sampleInside(self, objA, objB):
        return self.sampleOnTopOrInside(objA, objB, 'inside')

    def sampleNextTo(self, objA, objB):
        pass

    def sampleUnder(self, objA, objB):
        pass

    def sampleTouching(self, objA, objB):
        pass


def main():
    igtn_task = iGTNTask('kinematic_checker_testing', 2)
    igtn_task.initialize_simulator()

    for i in range(500):
        igtn_task.simulator.step()
    success, failed_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', failed_conditions)
    igtn_task.simulator.disconnect()


if __name__ == '__main__':
    main()
