import pprint 
import numpy as np
import os

from tasknet.task_base import TaskNetTask
import gibson2
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import URDFObject
from gibson2.object_states.on_floor import RoomFloor
from gibson2.external.pybullet_tools.utils import *
from gibson2.utils.constants import NON_SAMPLEABLE_OBJECTS, FLOOR_SYNSET
from gibson2.utils.assets_utils import get_ig_category_path, get_ig_model_path, get_ig_avg_category_specs
from gibson2.objects.vr_objects import VrAgent
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import pybullet as p
import cv2
from tasknet.condition_evaluation import Negation
from tasknet.logic_base import AtomicPredicate
import logging
import networkx as nx
from IPython import embed


class iGTNTask(TaskNetTask):
    def __init__(self, atus_activity, task_instance=0, predefined_problem=None):
        '''
        Initialize simulator with appropriate scene and sampled objects.
        :param atus_activity: string, official ATUS activity label
        :param task_instance: int, specific instance of atus_activity init/final conditions
                                   optional, randomly generated if not specified
        :param predefined_problem: string, in format of a BEHAVIOR problem file read
        '''
        super().__init__(atus_activity,
                         task_instance=task_instance,
                         scene_path=os.path.join(
                             gibson2.ig_dataset_path, 'scenes'),
                         predefined_problem=predefined_problem)

    def initialize_simulator(self,
                             simulator=None,
                             mode='iggui',
                             scene_id=None,
                             scene_kwargs=None,
                             load_clutter=False,
                             should_debug_sampling=False,
                             online_sampling=True):
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
            settings = MeshRendererSettings(texture_scale=0.01)
            self.simulator = Simulator(
                mode=mode, image_width=960, image_height=720, device_idx=0, rendering_settings=settings)
        else:
            self.simulator = simulator
        self.load_clutter = load_clutter
        self.should_debug_sampling = should_debug_sampling
        if online_sampling:
            scene_kwargs['merge_fixed_links'] = False
        return self.initialize(InteractiveIndoorScene,
                               scene_id=scene_id,
                               scene_kwargs=scene_kwargs,
                               online_sampling=online_sampling,
                               )

    def check_scene(self):
        feedback = {
            'init_success': 'yes',
            'goal_success': 'untested',
            'init_feedback': '',
            'goal_feedback': ''
        }
        self.newly_added_objects = set()
        room_type_to_obj_inst = {}
        self.non_sampleable_object_inst = set()
        for cond in self.parsed_initial_conditions:
            if cond[0] == 'inroom':
                obj_inst, room_type = cond[1], cond[2]
                obj_cat = self.obj_inst_to_obj_cat[obj_inst]
                if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                    error_msg = 'You have assigned room type for [{}], but [{}] is sampleable. Only non-sampleable objects can have room assignment.'.format(
                        obj_cat, obj_cat)
                    logging.warning(error_msg)
                    feedback['init_success'] = 'no'
                    feedback['init_feedback'] = error_msg
                    return False, feedback
                # Room type missing in the scene
                if room_type not in self.scene.room_sem_name_to_ins_name:
                    error_msg = 'Room type [{}] missing in scene [{}].'.format(
                        room_type, self.scene.scene_id)
                    logging.warning(error_msg)
                    feedback['init_success'] = 'no'
                    feedback['init_feedback'] = error_msg
                    return False, feedback

                if room_type not in room_type_to_obj_inst:
                    room_type_to_obj_inst[room_type] = []

                room_type_to_obj_inst[room_type].append(obj_inst)
                if obj_inst in self.non_sampleable_object_inst:
                    error_msg = 'Object [{}] has more than one room assignment'.format(
                        obj_inst)
                    logging.warning(error_msg)
                    feedback['init_success'] = 'no'
                    feedback['init_feedback'] = error_msg
                    return False, feedback
                self.non_sampleable_object_inst.add(obj_inst)

        self.sampling_orders = []
        cur_batch = self.non_sampleable_object_inst
        while len(cur_batch) > 0:
            self.sampling_orders.append(cur_batch)
            next_batch = set()
            for cond in self.parsed_initial_conditions:
                if len(cond) == 3 and cond[2] in cur_batch:
                    next_batch.add(cond[1])
            cur_batch = next_batch
        if len(self.sampling_orders) > 0:
            remaining_objs = self.object_scope.keys() - set.union(*self.sampling_orders)
        else:
            remaining_objs = self.object_scope.keys()
        if len(remaining_objs) != 0:
            error_msg = 'Some objects do not have any kinematic condition defined for them in the initial conditions: {}'.format(
                ', '.join(remaining_objs))
            logging.warning(error_msg)
            feedback['init_success'] = 'no'
            feedback['init_feedback'] = error_msg
            return False, feedback

        for obj_cat in self.objects:
            if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                continue
            for obj_inst in self.objects[obj_cat]:
                if obj_inst not in self.non_sampleable_object_inst:
                    error_msg = 'All non-sampleable objects should have room assignment. [{}] does not have one.'.format(
                        obj_inst)
                    logging.warning(error_msg)
                    feedback['init_success'] = 'no'
                    feedback['init_feedback'] = error_msg
                    return False, feedback

        room_type_to_scene_objs = {}
        for room_type in room_type_to_obj_inst:
            room_type_to_scene_objs[room_type] = {}
            for obj_inst in room_type_to_obj_inst[room_type]:
                room_type_to_scene_objs[room_type][obj_inst] = {}
                obj_cat = self.obj_inst_to_obj_cat[obj_inst]
                # We allow burners to be used as if they are stoves
                categories = \
                    self.object_taxonomy.get_subtree_igibson_categories(
                        obj_cat)
                if obj_cat == 'stove.n.01':
                    categories += \
                        self.object_taxonomy.get_subtree_igibson_categories(
                            'burner.n.01')
                for room_inst in self.scene.room_sem_name_to_ins_name[room_type]:
                    room_objs = self.scene.objects_by_room[room_inst]
                    if obj_cat == FLOOR_SYNSET:
                        # Create a RoomFloor for each room instance
                        # This object is NOT imported by the simulator
                        scene_objs = [
                            RoomFloor(category='room_floor',
                                      name='room_floor_{}'.format(room_inst),
                                      scene=self.scene,
                                      room_instance=room_inst)
                        ]
                    else:
                        scene_objs = [obj for obj in room_objs
                                      if obj.category in categories]
                    if len(scene_objs) != 0:
                        room_type_to_scene_objs[room_type][obj_inst][room_inst] = scene_objs

        # Store options for non-sampleable objects in self.non_sampleable_object_scope
        # {
        #     "table1": {
        #         "living_room_0": [URDFObject, URDFObject, URDFObject],
        #         "living_room_1": [URDFObject]
        #     },
        #     "table2": {
        #         "living_room_0": [URDFObject, URDFObject],
        #         "living_room_1": [URDFObject, URDFObject]
        #     },
        #     "chair1": {
        #         "living_room_0": [URDFObject],
        #         "living_room_1": [URDFObject]
        #     },
        # }
        for room_type in room_type_to_scene_objs:
            # For each room_type, filter in room_inst that has non-empty
            # options for all obj_inst in this room_type
            room_inst_satisfied = set.intersection(
                *[set(room_type_to_scene_objs[room_type][obj_inst].keys())
                  for obj_inst in room_type_to_scene_objs[room_type]]
            )
            if len(room_inst_satisfied) == 0:
                error_msg = 'Room type [{}] of scene [{}] does not contain all the objects needed.\nThe following are the possible room instances for each object, the intersection of which is an empty set.\n'.format(
                    room_type, self.scene.scene_id)
                for obj_inst in room_type_to_scene_objs[room_type]:
                    error_msg += '{}: '.format(obj_inst) + ', '.join(
                        room_type_to_scene_objs[room_type][obj_inst].keys()) + '\n'
                logging.warning(error_msg)
                feedback['init_success'] = 'no'
                feedback['init_feedback'] = error_msg
                return False, feedback

            for obj_inst in room_type_to_scene_objs[room_type]:
                room_type_to_scene_objs[room_type][obj_inst] = \
                    {key: val for key, val
                     in room_type_to_scene_objs[room_type][obj_inst].items()
                     if key in room_inst_satisfied}

        self.non_sampleable_object_scope = room_type_to_scene_objs

        num_new_obj = 0
        # Only populate self.object_scope for sampleable objects
        avg_category_spec = get_ig_avg_category_specs()
        for obj_cat in self.objects:
            if "agent" in obj_cat:
                continue
            if obj_cat in NON_SAMPLEABLE_OBJECTS:
                continue
            categories = \
                self.object_taxonomy.get_subtree_igibson_categories(
                    obj_cat)
            existing_scene_objs = []
            for category in categories:
                existing_scene_objs += self.scene.objects_by_category.get(
                    category, [])
            for obj_inst in self.objects[obj_cat]:
                # This obj category already exists in the scene
                # Priortize using those objects first before importing new ones
                if len(existing_scene_objs) > 0:
                    simulator_obj = np.random.choice(existing_scene_objs)
                    self.object_scope[obj_inst] = simulator_obj
                    existing_scene_objs.remove(simulator_obj)
                    continue

                category = np.random.choice(categories)
                # we always select pop, not pop_case
                if 'pop' in categories:
                    category = 'pop'
                # # cantaloup is a suitable category for melon.n.01
                # if 'cantaloup' in categories:
                #     category = 'cantaloup'
                category_path = get_ig_category_path(category)
                model = np.random.choice(os.listdir(category_path))
                # we can ONLY put stuff into this specific bag model
                if category == 'bag':
                    model = 'bag_001'
                model_path = get_ig_model_path(category, model)
                filename = os.path.join(model_path, model + ".urdf")
                obj_name = '{}_{}'.format(
                    category,
                    len(self.scene.objects_by_category.get(category, [])))
                simulator_obj = URDFObject(
                    filename,
                    name=obj_name,
                    category=category,
                    model_path=model_path,
                    avg_obj_dims=avg_category_spec.get(category),
                    fit_avg_dim_volume=True,
                    texture_randomization=False,
                    overwrite_inertial=True,
                    initial_pos=[100 + num_new_obj, 100, -100])
                if not self.scene.loaded:
                    self.scene.add_object(simulator_obj)
                else:
                    self.simulator.import_object(simulator_obj)
                self.newly_added_objects.add(simulator_obj)
                self.object_scope[obj_inst] = simulator_obj
                num_new_obj += 1

        return True, feedback

    def import_agent(self):
        # TODO: replace this with self.simulator.import_robot(VrAgent(self.simulator)) once VrAgent supports
        # baserobot api
        agent = VrAgent(self.simulator)
        self.agent = agent
        self.simulator.robots.append(agent)
        assert(len(self.simulator.robots) ==
               1), "Error, multiple agents is not currently supported"
        agent.vr_dict['body'].set_base_link_position_orientation(
            [300, 300, 300], [0, 0, 0, 1]
        )
        agent.vr_dict['left_hand'].set_base_link_position_orientation(
            [300, 300, -300], [0, 0, 0, 1]
        )
        agent.vr_dict['right_hand'].set_base_link_position_orientation(
            [300, -300, 300], [0, 0, 0, 1]
        )
        agent.vr_dict['left_hand'].ghost_hand.set_base_link_position_orientation(
            [300, 300, -300], [0, 0, 0, 1]
        )
        agent.vr_dict['right_hand'].ghost_hand.set_base_link_position_orientation(
            [300, -300, 300], [0, 0, 0, 1]
        )
        self.object_scope['agent.n.01_1'] = agent.vr_dict['body']
        if self.online_sampling == False and self.scene.agent != {}:
            agent.vr_dict['body'].set_base_link_position_orientation(
                self.scene.agent['VrBody']['xyz'], quat_from_euler(
                    self.scene.agent['VrBody']['rpy'])
            )
            agent.vr_dict['left_hand'].set_base_link_position_orientation(
                self.scene.agent['left_hand']['xyz'], quat_from_euler(
                    self.scene.agent['left_hand']['rpy'])
            )
            agent.vr_dict['right_hand'].set_base_link_position_orientation(
                self.scene.agent['right_hand']['xyz'], quat_from_euler(
                    self.scene.agent['right_hand']['rpy'])
            )
            agent.vr_dict['left_hand'].ghost_hand.set_base_link_position_orientation(
                self.scene.agent['left_hand']['xyz'], quat_from_euler(
                    self.scene.agent['left_hand']['rpy'])
            )
            agent.vr_dict['right_hand'].ghost_hand.set_base_link_position_orientation(
                self.scene.agent['right_hand']['xyz'], quat_from_euler(
                    self.scene.agent['right_hand']['rpy'])
            )

    def move_agent(self):
        if self.online_sampling == False and self.scene.agent == {}:
            agent = self.agent
            agent.vr_dict['body'].set_base_link_position_orientation(
                [0, 0, 0.5], [0, 0, 0, 1]
            )
            agent.vr_dict['left_hand'].set_base_link_position_orientation(
                [0, 0.2, 0.5], [0, 0, 0, 1]
            )
            agent.vr_dict['right_hand'].set_base_link_position_orientation(
                [0, -0.2, 0.5], [0, 0, 0, 1]
            )
            agent.vr_dict['left_hand'].ghost_hand.set_base_link_position_orientation(
                [0, 0.2, 0.5], [0, 0, 0, 1]
            )
            agent.vr_dict['right_hand'].ghost_hand.set_base_link_position_orientation(
                [0, 0.2, 0.5], [0, 0, 0, 1]
            )

    def import_scene(self):
        self.simulator.reload()
        self.simulator.import_ig_scene(self.scene)

        if not self.online_sampling:
            for obj_inst in self.object_scope:
                matched_sim_obj = None

                if 'floor.n.01' in obj_inst:
                    for _, sim_obj in self.scene.objects_by_name.items():
                        if sim_obj.tasknet_object_scope is not None and \
                                obj_inst in sim_obj.tasknet_object_scope:
                            tasknet_object_scope = \
                                sim_obj.tasknet_object_scope.split(',')
                            tasknet_object_scope = {
                                item.split(':')[0]: item.split(':')[1]
                                for item in tasknet_object_scope
                            }
                            assert obj_inst in tasknet_object_scope
                            room_inst = tasknet_object_scope[obj_inst].replace(
                                'room_floor_', '')

                            matched_sim_obj = \
                                RoomFloor(category='room_floor',
                                          name=tasknet_object_scope[obj_inst],
                                          scene=self.scene,
                                          room_instance=room_inst)
                elif 'agent' in obj_inst:
                    # Skip adding agent to object scope, handled later by import_agent()
                    continue
                else:
                    for _, sim_obj in self.scene.objects_by_name.items():
                        if sim_obj.tasknet_object_scope == obj_inst:
                            matched_sim_obj = sim_obj
                            break
                assert matched_sim_obj is not None, obj_inst
                self.object_scope[obj_inst] = matched_sim_obj

    def sample(self, kinematic_only=False):
        feedback = {
            'init_success': 'yes',
            'goal_success': 'yes',
            'init_feedback': '',
            'goal_feedback': ''
        }
        non_sampleable_obj_conditions = []
        sampleable_obj_conditions = []

        # TODO: currently we assume self.initial_conditions is a list of
        # tasknet.condition_evaluation.HEAD, each with one child.
        # This chid is either a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate or
        # a Negation of a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate
        for condition in self.initial_conditions:
            if not isinstance(condition.children[0], Negation) and not isinstance(condition.children[0], AtomicPredicate):
                print("Skipping over sampling of predicate that is not a negation or an atomic predicate")
                continue
            if kinematic_only:
                if isinstance(condition.children[0], Negation):
                    if condition.children[0].children[0].STATE_NAME not in ["ontop", "inside", "under"]:
                        continue
                else:
                    if condition.children[0].STATE_NAME not in ["ontop", "inside", "under"]:
                        continue
            if isinstance(condition.children[0], Negation):
                condition = condition.children[0].children[0]
                positive = False
            else:
                condition = condition.children[0]
                positive = True
            condition_body = set(condition.body)
            if len(self.non_sampleable_object_inst.intersection(condition_body)) > 0:
                non_sampleable_obj_conditions.append((condition, positive))
            else:
                sampleable_obj_conditions.append((condition, positive))

        # First, try to fulfill the initial conditions that involve non-sampleable objects
        # Filter in all simulator objects that allow successful sampling for each object inst
        init_sampling_log = []
        scene_object_scope_filtered = {}
        for room_type in self.non_sampleable_object_scope:
            scene_object_scope_filtered[room_type] = {}
            for scene_obj in self.non_sampleable_object_scope[room_type]:
                scene_object_scope_filtered[room_type][scene_obj] = {}
                for room_inst in self.non_sampleable_object_scope[room_type][scene_obj]:
                    for obj in self.non_sampleable_object_scope[room_type][scene_obj][room_inst]:
                        self.object_scope[scene_obj] = obj

                        success = True
                        # If this object is not involved in any initial conditions,
                        # success will be True by default and any simulator obj will qualify
                        for condition, positive in non_sampleable_obj_conditions:
                            # Only sample conditions that involve this object
                            if scene_obj not in condition.body:
                                continue
                            success = condition.sample(binary_state=positive)
                            log_msg = ' '.join(['initial condition sampling',
                                                room_type,
                                                scene_obj,
                                                room_inst,
                                                obj.name,
                                                condition.STATE_NAME,
                                                str(condition.body),
                                                str(success)])
                            print(log_msg)
                            init_sampling_log.append(log_msg)

                            if not success:
                                break

                        if not success:
                            continue

                        if room_inst not in scene_object_scope_filtered[room_type][scene_obj]:
                            scene_object_scope_filtered[room_type][scene_obj][room_inst] = [
                            ]
                        scene_object_scope_filtered[room_type][scene_obj][room_inst].append(
                            obj)

        for room_type in scene_object_scope_filtered:
            # For each room_type, filter in room_inst that has successful
            # sampling options for all obj_inst in this room_type
            room_inst_satisfied = set.intersection(
                *[set(scene_object_scope_filtered[room_type][obj_inst].keys())
                  for obj_inst in scene_object_scope_filtered[room_type]]
            )

            if len(room_inst_satisfied) == 0:
                error_msg = 'Room type [{}] of scene [{}] cannot sample all the objects needed.\nThe following are the possible room instances for each object, the intersection of which is an empty set.\n'.format(
                    room_type, self.scene.scene_id)
                for obj_inst in scene_object_scope_filtered[room_type]:
                    error_msg += '{}: '.format(obj_inst) + ', '.join(
                        scene_object_scope_filtered[room_type][obj_inst].keys()) + '\n'
                error_msg += 'The following are the initial condition sampling history:\n'
                error_msg += '\n'.join(init_sampling_log)
                logging.warning(error_msg)
                feedback['init_success'] = 'no'
                feedback['goal_success'] = 'untested'
                feedback['init_feedback'] = error_msg

                if self.should_debug_sampling:
                    self.debug_sampling(scene_object_scope_filtered,
                                        non_sampleable_obj_conditions)
                return False, feedback

            for obj_inst in scene_object_scope_filtered[room_type]:
                scene_object_scope_filtered[room_type][obj_inst] = \
                    {key: val for key, val
                     in scene_object_scope_filtered[room_type][obj_inst].items()
                     if key in room_inst_satisfied}

        # For each room instance, perform maximum bipartite matching between object instance in scope to simulator objects
        # Left nodes: a list of object instance in scope
        # Right nodes: a list of simulator objects
        # Edges: if the simulator object can support the sampling requirement of ths object instance
        for room_type in scene_object_scope_filtered:
            # The same room instances will be shared across all scene obj in a given room type
            some_obj = list(scene_object_scope_filtered[room_type].keys())[0]
            room_insts = list(scene_object_scope_filtered[room_type][some_obj].keys(
            ))
            success = False
            init_mbm_log = []
            # Loop through each room instance
            for room_inst in room_insts:
                graph = nx.Graph()
                # For this given room instance, gether mapping from obj instance to a list of simulator obj
                obj_inst_to_obj_per_room_inst = {}
                for obj_inst in scene_object_scope_filtered[room_type]:
                    obj_inst_to_obj_per_room_inst[obj_inst] = scene_object_scope_filtered[room_type][obj_inst][room_inst]
                top_nodes = []
                log_msg = 'MBM for room instance [{}]'.format(room_inst)
                print(log_msg)
                init_mbm_log.append(log_msg)
                for obj_inst in obj_inst_to_obj_per_room_inst:
                    for obj in obj_inst_to_obj_per_room_inst[obj_inst]:
                        # Create an edge between obj instance and each of the simulator obj that supports sampling
                        graph.add_edge(obj_inst, obj)
                        log_msg = 'Adding edge: {} <-> {}'.format(
                            obj_inst, obj.name)
                        print(log_msg)
                        init_mbm_log.append(log_msg)
                        top_nodes.append(obj_inst)
                # Need to provide top_nodes that contain all nodes in one bipartite node set
                # The matches will have two items for each match (e.g. A -> B, B -> A)
                matches = nx.bipartite.maximum_matching(
                    graph, top_nodes=top_nodes)
                if len(matches) == 2 * len(obj_inst_to_obj_per_room_inst):
                    print('Object scope finalized:')
                    for obj_inst, obj in matches.items():
                        if obj_inst in obj_inst_to_obj_per_room_inst:
                            self.object_scope[obj_inst] = obj
                            print(obj_inst, obj.name)
                    success = True
                    break
            if not success:
                error_msg = 'Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n'.format(
                    room_type, self.scene.scene_id)
                error_msg += 'The following are the initial condition matching history:\n'
                error_msg += '\n'.join(init_mbm_log)
                logging.warning(error_msg)
                feedback['init_success'] = 'no'
                feedback['goal_success'] = 'untested'
                feedback['init_feedback'] = error_msg
                return False, feedback

        np.random.shuffle(self.ground_goal_state_options)
        print('number of ground_goal_state_options',
              len(self.ground_goal_state_options))
        num_goal_condition_set_to_test = 10

        goal_sampling_error_msgs = []
        # Next, try to fulfill different set of ground goal conditions (maximum num_goal_condition_set_to_test)
        for goal_condition_set in \
                self.ground_goal_state_options[:num_goal_condition_set_to_test]:
            goal_condition_set_success = True
            goal_sampling_log = []
            # Try to fulfill the current set of ground goal conditions
            scene_object_scope_filtered_goal_cond = {}
            for room_type in scene_object_scope_filtered:
                scene_object_scope_filtered_goal_cond[room_type] = {}
                for scene_obj in scene_object_scope_filtered[room_type]:
                    scene_object_scope_filtered_goal_cond[room_type][scene_obj] = {
                    }
                    for room_inst in scene_object_scope_filtered[room_type][scene_obj]:
                        for obj in scene_object_scope_filtered[room_type][scene_obj][room_inst]:
                            self.object_scope[scene_obj] = obj

                            success = True
                            for goal_condition in goal_condition_set:
                                goal_condition = goal_condition.children[0]
                                # do not sample negative goal condition
                                if isinstance(goal_condition, Negation):
                                    continue
                                # only sample kinematic goal condition
                                if goal_condition.STATE_NAME not in ['inside', 'ontop', 'under']:
                                    continue
                                if scene_obj not in goal_condition.body:
                                    continue
                                success = goal_condition.sample(
                                    binary_state=True)
                                log_msg = ' '.join(['goal condition sampling',
                                                    room_type,
                                                    scene_obj,
                                                    room_inst,
                                                    obj.name,
                                                    goal_condition.STATE_NAME,
                                                    str(goal_condition.body),
                                                    str(success)])
                                print(log_msg)
                                goal_sampling_log.append(log_msg)
                                if not success:
                                    break
                            if not success:
                                continue

                            if room_inst not in scene_object_scope_filtered_goal_cond[room_type][scene_obj]:
                                scene_object_scope_filtered_goal_cond[room_type][scene_obj][room_inst] = [
                                ]
                            scene_object_scope_filtered_goal_cond[room_type][scene_obj][room_inst].append(
                                obj)

            for room_type in scene_object_scope_filtered_goal_cond:
                # For each room_type, filter in room_inst that has successful
                # sampling options for all obj_inst in this room_type
                room_inst_satisfied = set.intersection(
                    *[set(scene_object_scope_filtered_goal_cond[room_type][obj_inst].keys())
                      for obj_inst in scene_object_scope_filtered_goal_cond[room_type]]
                )

                if len(room_inst_satisfied) == 0:
                    error_msg = 'Room type [{}] of scene [{}] cannot sample all the objects needed.\nThe following are the possible room instances for each object, the intersection of which is an empty set.\n'.format(
                        room_type, self.scene.scene_id)
                    for obj_inst in scene_object_scope_filtered_goal_cond[room_type]:
                        error_msg += '{}: '.format(obj_inst) + ', '.join(
                            scene_object_scope_filtered_goal_cond[room_type][obj_inst].keys()) + '\n'
                    error_msg += 'The following are the goal condition sampling history:\n'
                    error_msg += '\n'.join(goal_sampling_log)
                    logging.warning(error_msg)
                    goal_sampling_error_msgs.append(error_msg)
                    if self.should_debug_sampling:
                        self.debug_sampling(scene_object_scope_filtered_goal_cond,
                                            non_sampleable_obj_conditions,
                                            goal_condition_set)
                    goal_condition_set_success = False
                    break

                for obj_inst in scene_object_scope_filtered_goal_cond[room_type]:
                    scene_object_scope_filtered_goal_cond[room_type][obj_inst] = \
                        {key: val for key, val
                         in scene_object_scope_filtered_goal_cond[room_type][obj_inst].items()
                         if key in room_inst_satisfied}

            if not goal_condition_set_success:
                continue
            # For each room instance, perform maximum bipartite matching between object instance in scope to simulator objects
            # Left nodes: a list of object instance in scope
            # Right nodes: a list of simulator objects
            # Edges: if the simulator object can support the sampling requirement of ths object instance
            for room_type in scene_object_scope_filtered_goal_cond:
                # The same room instances will be shared across all scene obj in a given room type
                some_obj = list(
                    scene_object_scope_filtered_goal_cond[room_type].keys())[0]
                room_insts = list(scene_object_scope_filtered_goal_cond[room_type][some_obj].keys(
                ))
                success = False
                goal_mbm_log = []
                # Loop through each room instance
                for room_inst in room_insts:
                    graph = nx.Graph()
                    # For this given room instance, gether mapping from obj instance to a list of simulator obj
                    obj_inst_to_obj_per_room_inst = {}
                    for obj_inst in scene_object_scope_filtered_goal_cond[room_type]:
                        obj_inst_to_obj_per_room_inst[obj_inst] = scene_object_scope_filtered_goal_cond[room_type][obj_inst][room_inst]
                    top_nodes = []
                    log_msg = 'MBM for room instance [{}]'.format(room_inst)
                    print(log_msg)
                    goal_mbm_log.append(log_msg)
                    for obj_inst in obj_inst_to_obj_per_room_inst:
                        for obj in obj_inst_to_obj_per_room_inst[obj_inst]:
                            # Create an edge between obj instance and each of the simulator obj that supports sampling
                            graph.add_edge(obj_inst, obj)
                            log_msg = 'Adding edge: {} <-> {}'.format(
                                obj_inst, obj.name)
                            print(log_msg)
                            goal_mbm_log.append(log_msg)
                            top_nodes.append(obj_inst)
                    # Need to provide top_nodes that contain all nodes in one bipartite node set
                    # The matches will have two items for each match (e.g. A -> B, B -> A)
                    matches = nx.bipartite.maximum_matching(
                        graph, top_nodes=top_nodes)
                    if len(matches) == 2 * len(obj_inst_to_obj_per_room_inst):
                        print('Object scope finalized:')
                        for obj_inst, obj in matches.items():
                            if obj_inst in obj_inst_to_obj_per_room_inst:
                                self.object_scope[obj_inst] = obj
                                print(obj_inst, obj.name)
                        success = True
                        break
                if not success:
                    error_msg = 'Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n'.format(
                        room_type, self.scene.scene_id)
                    error_msg += 'The following are the goal condition matching history:\n'
                    error_msg += '\n'.join(goal_mbm_log)
                    logging.warning(error_msg)
                    goal_sampling_error_msgs.append(error_msg)
                    goal_condition_set_success = False
                    break

            if not goal_condition_set_success:
                continue

            # if one set of goal conditions (and initial conditions) are satisfied, sampling is successful
            break

        if not goal_condition_set_success:
            goal_sampling_error_msg_compiled = ''
            for i, log_msg in enumerate(goal_sampling_error_msgs):
                goal_sampling_error_msg_compiled += '-' * 30 + '\n'
                goal_sampling_error_msg_compiled += 'Ground condition set #{}/{}:\n'.format(
                    i + 1, len(goal_sampling_error_msgs))
                goal_sampling_error_msg_compiled += log_msg + '\n'
            feedback['goal_success'] = 'no'
            feedback['goal_feedback'] = goal_sampling_error_msg_compiled
            return False, feedback

        # Do sampling again using the object instance -> simulator object mapping from maximum bipartite matching
        for condition, positive in non_sampleable_obj_conditions:
            num_trials = 10
            for _ in range(num_trials):
                success = condition.sample(binary_state=positive)
                # This should always succeed because it has succeeded before.
                if success:
                    break
            if not success:
                logging.warning(
                    'Non-sampleable object conditions failed even after successful matching: {}'.format(
                        condition.body))
                feedback['init_success'] = 'no'
                feedback['init_feedback'] = 'Please run test sampling again.'
                return False, feedback

        # Use ray casting for ontop and inside sampling for non-sampleable objects
        for condition, positive in sampleable_obj_conditions:
#             success = condition.sample(binary_state=positive)
#             if not success:
#                 logging.warning(
#                     'Sampleable object conditions failed: {}'.format(
#                         condition.body))
#                 feedback['init_success'] = 'no'
#                 feedback['init_feedback'] = 'Please run test sampling again.'
#                 return False, feedback
            if condition.STATE_NAME in ['inside', 'ontop']:
                condition.kwargs['use_ray_casting_method'] = True

        # Pop non-sampleable objects
        if len(self.sampling_orders) > 0:
            self.sampling_orders.pop(0)
            for cur_batch in self.sampling_orders:
                for condition, positive in sampleable_obj_conditions:
                    # Sample conditions that involve the current batch of objects
                    if condition.body[0] in cur_batch:
                        success = condition.sample(binary_state=positive)
                        if not success:
                            error_msg = 'Sampleable object conditions failed: {}'.format(
                                condition.body)
                            logging.warning(error_msg)
                            feedback['init_success'] = 'no'
                            feedback['init_feedback'] = error_msg
                            return False, feedback

        return True, feedback

    def debug_sampling(self,
                       scene_object_scope_filtered,
                       non_sampleable_obj_conditions,
                       goal_condition_set=None):
        gibson2.debug_sampling = True
        for room_type in self.non_sampleable_object_scope:
            for scene_obj in self.non_sampleable_object_scope[room_type]:
                if len(scene_object_scope_filtered[room_type][scene_obj].keys()) != 0:
                    continue
                for room_inst in self.non_sampleable_object_scope[room_type][scene_obj]:
                    for obj in self.non_sampleable_object_scope[room_type][scene_obj][room_inst]:
                        self.object_scope[scene_obj] = obj

                        for condition, positive in non_sampleable_obj_conditions:
                            # Only sample conditions that involve this object
                            if scene_obj not in condition.body:
                                continue
                            print('debug initial condition sampling',
                                  room_type,
                                  scene_obj,
                                  room_inst,
                                  obj.name,
                                  condition.STATE_NAME,
                                  condition.body)
                            obj_pos = obj.get_position()
                            # Set the pybullet camera to have a bird's eye view
                            # of the sampling process
                            p.resetDebugVisualizerCamera(
                                cameraDistance=3.0, cameraYaw=0,
                                cameraPitch=-89.99999,
                                cameraTargetPosition=obj_pos)
                            success = condition.sample(binary_state=positive)
                            print('success', success)

                        if goal_condition_set is None:
                            continue

                        for goal_condition in goal_condition_set:
                            goal_condition = goal_condition.children[0]
                            if isinstance(goal_condition, Negation):
                                continue
                            if goal_condition.STATE_NAME not in ['inside', 'ontop', 'under']:
                                continue
                            if scene_obj not in goal_condition.body:
                                continue
                            print('goal condition sampling',
                                  room_type,
                                  scene_obj,
                                  room_inst, obj.name,
                                  goal_condition.STATE_NAME,
                                  goal_condition.body)
                            obj_pos = obj.get_position()
                            # Set the pybullet camera to have a bird's eye view
                            # of the sampling process
                            p.resetDebugVisualizerCamera(
                                cameraDistance=3.0, cameraYaw=0,
                                cameraPitch=-89.99999,
                                cameraTargetPosition=obj_pos)
                            success = goal_condition.sample(
                                binary_state=True)
                            print('success', success)

    def clutter_scene(self):
        if not self.load_clutter:
            return

        scene_id = self.scene.scene_id
        clutter_ids = [''] + list(range(2, 5))
        clutter_id = np.random.choice(clutter_ids)
        clutter_scene = InteractiveIndoorScene(
            scene_id, '{}_clutter{}'.format(scene_id, clutter_id))
        existing_objects = [
            value for key, value in self.object_scope.items()
            if 'floor.n.01' not in key]
        self.simulator.import_non_colliding_objects(
            objects=clutter_scene.objects_by_name,
            existing_objects=existing_objects,
            min_distance=0.5)

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
