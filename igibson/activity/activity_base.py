import logging
from collections import OrderedDict

import cv2
import networkx as nx
import pybullet as p
from bddl.activity_base import BEHAVIORActivityInstance
from bddl.condition_evaluation import Negation
from bddl.logic_base import AtomicFormula

import igibson
from igibson.external.pybullet_tools.utils import *
from igibson.object_states.on_floor import RoomFloor
from igibson.objects.articulated_object import URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.checkpoint_utils import load_internal_states, save_internal_states
from igibson.utils.constants import (
    AGENT_POSE_DIM,
    FLOOR_SYNSET,
    MAX_TASK_RELEVANT_OBJS,
    NON_SAMPLEABLE_OBJECTS,
    TASK_RELEVANT_OBJS_OBS_DIM,
)

KINEMATICS_STATES = frozenset({"inside", "ontop", "under", "onfloor"})


class iGBEHAVIORActivityInstance(BEHAVIORActivityInstance):
    def __init__(self, behavior_activity, activity_definition=0, predefined_problem=None):
        """
        Initialize simulator with appropriate scene and sampled objects.
        :param behavior_activity: string, official ATUS activity label
        :param activity_definition: int, specific instance of behavior_activity init/final conditions
                                   optional, randomly generated if not specified
        :param predefined_problem: string, in format of a BEHAVIOR problem file read
        """
        super().__init__(
            behavior_activity,
            activity_definition=activity_definition,
            scene_path=os.path.join(igibson.ig_dataset_path, "scenes"),
            predefined_problem=predefined_problem,
        )
        self.state_history = {}

    def initialize_simulator(
        self,
        simulator=None,
        mode="headless",
        scene_id=None,
        scene_kwargs=None,
        load_clutter=False,
        should_debug_sampling=False,
        online_sampling=True,
    ):
        """
        Get scene populated with objects such that scene satisfies initial conditions
        :param simulator: Simulator class, populated simulator that should completely
                                   replace this function. Use if you would like to bypass internal
                                   Simulator instantiation and population based on initial conditions
                                   and use your own. Warning that if you use this option, we cannot
                                   guarantee that the final conditions will be reachable.
        """
        # Set self.scene_name, self.scene, self.sampled_simulator_objects, and self.sampled_dsl_objects
        if simulator is None:
            self.simulator = Simulator(mode=mode, image_width=960, image_height=720, device_idx=0)
        else:
            self.simulator = simulator
        self.load_clutter = load_clutter
        self.should_debug_sampling = should_debug_sampling
        if online_sampling:
            scene_kwargs["merge_fixed_links"] = False
        result = self.initialize(
            InteractiveIndoorScene,
            scene_id=scene_id,
            scene_kwargs=scene_kwargs,
            online_sampling=online_sampling,
        )
        self.initial_state = self.save_scene()
        self.task_obs_dim = MAX_TASK_RELEVANT_OBJS * TASK_RELEVANT_OBJS_OBS_DIM + AGENT_POSE_DIM
        return result

    def save_scene(self):
        snapshot_id = p.saveState()
        self.state_history[snapshot_id] = save_internal_states(self.simulator)
        return snapshot_id

    def reset_scene(self, snapshot_id):
        p.restoreState(snapshot_id)
        load_internal_states(self.simulator, self.state_history[snapshot_id])

    def check_scene(self):
        feedback = {"init_success": "yes", "goal_success": "untested", "init_feedback": "", "goal_feedback": ""}
        self.newly_added_objects = set()
        room_type_to_obj_inst = {}
        self.non_sampleable_object_inst = set()
        for cond in self.parsed_initial_conditions:
            if cond[0] == "inroom":
                obj_inst, room_type = cond[1], cond[2]
                obj_cat = self.obj_inst_to_obj_cat[obj_inst]
                if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                    error_msg = "You have assigned room type for [{}], but [{}] is sampleable. Only non-sampleable objects can have room assignment.".format(
                        obj_cat, obj_cat
                    )
                    logging.warning(error_msg)
                    feedback["init_success"] = "no"
                    feedback["init_feedback"] = error_msg
                    return False, feedback
                # Room type missing in the scene
                if room_type not in self.scene.room_sem_name_to_ins_name:
                    error_msg = "Room type [{}] missing in scene [{}].".format(room_type, self.scene.scene_id)
                    logging.warning(error_msg)
                    feedback["init_success"] = "no"
                    feedback["init_feedback"] = error_msg
                    return False, feedback

                if room_type not in room_type_to_obj_inst:
                    room_type_to_obj_inst[room_type] = []

                room_type_to_obj_inst[room_type].append(obj_inst)
                if obj_inst in self.non_sampleable_object_inst:
                    error_msg = "Object [{}] has more than one room assignment".format(obj_inst)
                    logging.warning(error_msg)
                    feedback["init_success"] = "no"
                    feedback["init_feedback"] = error_msg
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
            error_msg = "Some objects do not have any kinematic condition defined for them in the initial conditions: {}".format(
                ", ".join(remaining_objs)
            )
            logging.warning(error_msg)
            feedback["init_success"] = "no"
            feedback["init_feedback"] = error_msg
            return False, feedback

        for obj_cat in self.objects:
            if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                continue
            for obj_inst in self.objects[obj_cat]:
                if obj_inst not in self.non_sampleable_object_inst:
                    error_msg = (
                        "All non-sampleable objects should have room assignment. [{}] does not have one.".format(
                            obj_inst
                        )
                    )
                    logging.warning(error_msg)
                    feedback["init_success"] = "no"
                    feedback["init_feedback"] = error_msg
                    return False, feedback

        room_type_to_scene_objs = {}
        for room_type in room_type_to_obj_inst:
            room_type_to_scene_objs[room_type] = {}
            for obj_inst in room_type_to_obj_inst[room_type]:
                room_type_to_scene_objs[room_type][obj_inst] = {}
                obj_cat = self.obj_inst_to_obj_cat[obj_inst]
                # We allow burners to be used as if they are stoves
                categories = self.object_taxonomy.get_subtree_igibson_categories(obj_cat)
                if obj_cat == "stove.n.01":
                    categories += self.object_taxonomy.get_subtree_igibson_categories("burner.n.01")
                for room_inst in self.scene.room_sem_name_to_ins_name[room_type]:
                    if obj_cat == FLOOR_SYNSET:
                        # TODO: remove after split floors
                        # Create a RoomFloor for each room instance
                        # This object is NOT imported by the simulator
                        room_floor = RoomFloor(
                            category="room_floor",
                            name="room_floor_{}".format(room_inst),
                            scene=self.scene,
                            room_instance=room_inst,
                            floor_obj=self.scene.objects_by_name["floors"],
                        )
                        scene_objs = [room_floor]
                    else:
                        room_objs = []
                        if room_inst in self.scene.objects_by_room:
                            room_objs = self.scene.objects_by_room[room_inst]
                        scene_objs = [obj for obj in room_objs if obj.category in categories]
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
                *[
                    set(room_type_to_scene_objs[room_type][obj_inst].keys())
                    for obj_inst in room_type_to_scene_objs[room_type]
                ]
            )
            if len(room_inst_satisfied) == 0:
                error_msg = "Room type [{}] of scene [{}] does not contain all the objects needed.\nThe following are the possible room instances for each object, the intersection of which is an empty set.\n".format(
                    room_type, self.scene.scene_id
                )
                for obj_inst in room_type_to_scene_objs[room_type]:
                    error_msg += (
                        "{}: ".format(obj_inst) + ", ".join(room_type_to_scene_objs[room_type][obj_inst].keys()) + "\n"
                    )
                logging.warning(error_msg)
                feedback["init_success"] = "no"
                feedback["init_feedback"] = error_msg
                return False, feedback

            for obj_inst in room_type_to_scene_objs[room_type]:
                room_type_to_scene_objs[room_type][obj_inst] = {
                    key: val
                    for key, val in room_type_to_scene_objs[room_type][obj_inst].items()
                    if key in room_inst_satisfied
                }

        self.non_sampleable_object_scope = room_type_to_scene_objs

        num_new_obj = 0
        # Only populate self.object_scope for sampleable objects
        avg_category_spec = get_ig_avg_category_specs()
        for obj_cat in self.objects:
            if obj_cat == "agent.n.01":
                continue
            if obj_cat in NON_SAMPLEABLE_OBJECTS:
                continue
            is_sliceable = self.object_taxonomy.has_ability(obj_cat, "sliceable")
            categories = self.object_taxonomy.get_subtree_igibson_categories(obj_cat)

            # TODO: temporary hack
            remove_categories = [
                "pop_case",  # too large
                "jewel",  # too small
                "ring",  # too small
            ]
            for remove_category in remove_categories:
                if remove_category in categories:
                    categories.remove(remove_category)

            if is_sliceable:
                categories = [cat for cat in categories if "half_" not in cat]

            for obj_inst in self.objects[obj_cat]:
                category = np.random.choice(categories)
                category_path = get_ig_category_path(category)
                model_choices = os.listdir(category_path)

                # Filter object models if the object category is openable
                synset = self.object_taxonomy.get_class_name_from_igibson_category(category)
                if self.object_taxonomy.has_ability(synset, "openable"):
                    model_choices = [m for m in model_choices if "articulated_" in m]
                    if len(model_choices) == 0:
                        error_msg = "{} is Openable, but does not have articulated models.".format(category)
                        logging.warning(error_msg)
                        feedback["init_success"] = "no"
                        feedback["init_feedback"] = error_msg
                        return False, feedback

                model = np.random.choice(model_choices)

                # for "collecting aluminum cans", we need pop cans (not bottles)
                if category == "pop" and self.behavior_activity in ["collecting_aluminum_cans"]:
                    model = np.random.choice([str(i) for i in range(40, 46)])
                if category == "spoon" and self.behavior_activity in ["polishing_silver"]:
                    model = np.random.choice([str(i) for i in [2, 5, 6]])

                model_path = get_ig_model_path(category, model)
                filename = os.path.join(model_path, model + ".urdf")
                obj_name = "{}_{}".format(category, len(self.scene.objects_by_name))
                simulator_obj = URDFObject(
                    filename,
                    name=obj_name,
                    category=category,
                    model_path=model_path,
                    avg_obj_dims=avg_category_spec.get(category),
                    fit_avg_dim_volume=True,
                    texture_randomization=False,
                    overwrite_inertial=True,
                    initial_pos=[100 + num_new_obj, 100, -100],
                )
                num_new_obj += 1

                if is_sliceable:
                    whole_object = simulator_obj
                    object_parts = []
                    assert "object_parts" in simulator_obj.metadata, "object_parts not found in metadata: [{}]".format(
                        model_path
                    )

                    for i, part in enumerate(simulator_obj.metadata["object_parts"]):
                        category = part["category"]
                        model = part["model"]
                        # Scale the offset accordingly
                        part_pos = part["pos"] * whole_object.scale
                        part_orn = part["orn"]
                        model_path = get_ig_model_path(category, model)
                        filename = os.path.join(model_path, model + ".urdf")
                        obj_name = whole_object.name + "_part_{}".format(i)
                        simulator_obj_part = URDFObject(
                            filename,
                            name=obj_name,
                            category=category,
                            model_path=model_path,
                            avg_obj_dims=avg_category_spec.get(category),
                            fit_avg_dim_volume=False,
                            scale=whole_object.scale,
                            texture_randomization=False,
                            overwrite_inertial=True,
                            initial_pos=[100 + num_new_obj, 100, -100],
                        )
                        num_new_obj += 1
                        object_parts.append((simulator_obj_part, (part_pos, part_orn)))

                    assert len(object_parts) > 0
                    grouped_obj_parts = ObjectGrouper(object_parts)
                    simulator_obj = ObjectMultiplexer(
                        whole_object.name + "_multiplexer", [whole_object, grouped_obj_parts], 0
                    )

                if not self.scene.loaded:
                    self.scene.add_object(simulator_obj)
                else:
                    self.simulator.import_object(simulator_obj)
                self.newly_added_objects.add(simulator_obj)
                self.object_scope[obj_inst] = simulator_obj

        return True, feedback

    def import_agent(self):
        # TODO: replace this with self.simulator.import_robot(BehaviorRobot(self.simulator)) once BehaviorRobot supports
        # baserobot api
        agent = BehaviorRobot(self.simulator)
        self.simulator.import_behavior_robot(agent)
        self.simulator.register_main_vr_robot(agent)
        self.agent = agent
        self.simulator.robots.append(agent)
        assert len(self.simulator.robots) == 1, "Error, multiple agents is not currently supported"
        agent.parts["body"].set_base_link_position_orientation([300, 300, 300], [0, 0, 0, 1])
        agent.parts["left_hand"].set_base_link_position_orientation([300, 300, -300], [0, 0, 0, 1])
        agent.parts["right_hand"].set_base_link_position_orientation([300, -300, 300], [0, 0, 0, 1])
        agent.parts["left_hand"].ghost_hand.set_base_link_position_orientation([300, 300, -300], [0, 0, 0, 1])
        agent.parts["right_hand"].ghost_hand.set_base_link_position_orientation([300, -300, 300], [0, 0, 0, 1])
        agent.parts["eye"].set_base_link_position_orientation([300, -300, -300], [0, 0, 0, 1])
        self.object_scope["agent.n.01_1"] = agent.parts["body"]
        if not self.online_sampling and self.scene.agent != {}:
            agent.parts["body"].set_base_link_position_orientation(
                self.scene.agent["BRBody_1"]["xyz"], quat_from_euler(self.scene.agent["BRBody_1"]["rpy"])
            )
            agent.parts["left_hand"].set_base_link_position_orientation(
                self.scene.agent["left_hand_1"]["xyz"], quat_from_euler(self.scene.agent["left_hand_1"]["rpy"])
            )
            agent.parts["right_hand"].set_base_link_position_orientation(
                self.scene.agent["right_hand_1"]["xyz"], quat_from_euler(self.scene.agent["right_hand_1"]["rpy"])
            )
            agent.parts["left_hand"].ghost_hand.set_base_link_position_orientation(
                self.scene.agent["left_hand_1"]["xyz"], quat_from_euler(self.scene.agent["left_hand_1"]["rpy"])
            )
            agent.parts["right_hand"].ghost_hand.set_base_link_position_orientation(
                self.scene.agent["right_hand_1"]["xyz"], quat_from_euler(self.scene.agent["right_hand_1"]["rpy"])
            )
            agent.parts["eye"].set_base_link_position_orientation(
                self.scene.agent["BREye_1"]["xyz"], quat_from_euler(self.scene.agent["BREye_1"]["rpy"])
            )

    def move_agent(self):
        agent = self.agent
        if not self.online_sampling and self.scene.agent == {}:
            agent.parts["body"].set_base_link_position_orientation([0, 0, 0.5], [0, 0, 0, 1])
            agent.parts["left_hand"].set_base_link_position_orientation(
                [0, 0.2, 0.7],
                [0.5, 0.5, -0.5, 0.5],
            )
            agent.parts["right_hand"].set_base_link_position_orientation([0, -0.2, 0.7], [-0.5, 0.5, 0.5, 0.5])
            agent.parts["left_hand"].ghost_hand.set_base_link_position_orientation([0, 0.2, 0.7], [0.5, 0.5, -0.5, 0.5])
            agent.parts["right_hand"].ghost_hand.set_base_link_position_orientation(
                [0, -0.2, 0.7], [-0.5, 0.5, 0.5, 0.5]
            )
            agent.parts["eye"].set_base_link_position_orientation([0, 0, 1.5], [0, 0, 0, 1])

    def import_scene(self):
        self.simulator.reload()
        self.simulator.import_ig_scene(self.scene)

        if not self.online_sampling:
            for obj_inst in self.object_scope:
                matched_sim_obj = None

                # TODO: remove after split floors
                if "floor.n.01" in obj_inst:
                    for _, sim_obj in self.scene.objects_by_name.items():
                        if sim_obj.bddl_object_scope is not None and obj_inst in sim_obj.bddl_object_scope:
                            bddl_object_scope = sim_obj.bddl_object_scope.split(",")
                            bddl_object_scope = {item.split(":")[0]: item.split(":")[1] for item in bddl_object_scope}
                            assert obj_inst in bddl_object_scope
                            room_inst = bddl_object_scope[obj_inst].replace("room_floor_", "")
                            matched_sim_obj = RoomFloor(
                                category="room_floor",
                                name=bddl_object_scope[obj_inst],
                                scene=self.scene,
                                room_instance=room_inst,
                                floor_obj=self.scene.objects_by_name["floors"],
                            )
                elif obj_inst == "agent.n.01_1":
                    # Skip adding agent to object scope, handled later by import_agent()
                    continue
                else:
                    for _, sim_obj in self.scene.objects_by_name.items():
                        if sim_obj.bddl_object_scope == obj_inst:
                            matched_sim_obj = sim_obj
                            break
                assert matched_sim_obj is not None, obj_inst
                self.object_scope[obj_inst] = matched_sim_obj

    def sample(self, kinematic_only=False):
        feedback = {"init_success": "yes", "goal_success": "yes", "init_feedback": "", "goal_feedback": ""}
        non_sampleable_obj_conditions = []
        sampleable_obj_conditions = []

        # TODO: currently we assume self.initial_conditions is a list of
        # bddl.condition_evaluation.HEAD, each with one child.
        # This chid is either a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate or
        # a Negation of a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate
        for condition in self.initial_conditions:
            if not isinstance(condition.children[0], Negation) and not isinstance(condition.children[0], AtomicFormula):
                logging.warning(("Skipping over sampling of predicate that is not a negation or an atomic formula"))
                continue
            if kinematic_only:
                if isinstance(condition.children[0], Negation):
                    if condition.children[0].children[0].STATE_NAME not in KINEMATICS_STATES:
                        continue
                else:
                    if condition.children[0].STATE_NAME not in KINEMATICS_STATES:
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
                            # Always skip non-kinematic state sampling. Only do so after the object scope has been finalized
                            if condition.STATE_NAME not in KINEMATICS_STATES:
                                continue
                            # Only sample conditions that involve this object
                            if scene_obj not in condition.body:
                                continue
                            success = condition.sample(binary_state=positive)
                            log_msg = " ".join(
                                [
                                    "initial condition sampling",
                                    room_type,
                                    scene_obj,
                                    room_inst,
                                    obj.name,
                                    condition.STATE_NAME,
                                    str(condition.body),
                                    str(success),
                                ]
                            )
                            logging.warning((log_msg))
                            init_sampling_log.append(log_msg)

                            if not success:
                                break

                        if not success:
                            continue

                        if room_inst not in scene_object_scope_filtered[room_type][scene_obj]:
                            scene_object_scope_filtered[room_type][scene_obj][room_inst] = []
                        scene_object_scope_filtered[room_type][scene_obj][room_inst].append(obj)

        for room_type in scene_object_scope_filtered:
            # For each room_type, filter in room_inst that has successful
            # sampling options for all obj_inst in this room_type
            room_inst_satisfied = set.intersection(
                *[
                    set(scene_object_scope_filtered[room_type][obj_inst].keys())
                    for obj_inst in scene_object_scope_filtered[room_type]
                ]
            )

            if len(room_inst_satisfied) == 0:
                error_msg = "Room type [{}] of scene [{}] cannot sample all the objects needed.\nThe following are the possible room instances for each object, the intersection of which is an empty set.\n".format(
                    room_type, self.scene.scene_id
                )
                for obj_inst in scene_object_scope_filtered[room_type]:
                    error_msg += (
                        "{}: ".format(obj_inst)
                        + ", ".join(scene_object_scope_filtered[room_type][obj_inst].keys())
                        + "\n"
                    )
                error_msg += "The following are the initial condition sampling history:\n"
                error_msg += "\n".join(init_sampling_log)
                logging.warning(error_msg)
                feedback["init_success"] = "no"
                feedback["goal_success"] = "untested"
                feedback["init_feedback"] = error_msg

                if self.should_debug_sampling:
                    self.debug_sampling(scene_object_scope_filtered, non_sampleable_obj_conditions)
                return False, feedback

            for obj_inst in scene_object_scope_filtered[room_type]:
                scene_object_scope_filtered[room_type][obj_inst] = {
                    key: val
                    for key, val in scene_object_scope_filtered[room_type][obj_inst].items()
                    if key in room_inst_satisfied
                }

        # For each room instance, perform maximum bipartite matching between object instance in scope to simulator objects
        # Left nodes: a list of object instance in scope
        # Right nodes: a list of simulator objects
        # Edges: if the simulator object can support the sampling requirement of ths object instance
        for room_type in scene_object_scope_filtered:
            # The same room instances will be shared across all scene obj in a given room type
            some_obj = list(scene_object_scope_filtered[room_type].keys())[0]
            room_insts = list(scene_object_scope_filtered[room_type][some_obj].keys())
            success = False
            init_mbm_log = []
            # Loop through each room instance
            for room_inst in room_insts:
                graph = nx.Graph()
                # For this given room instance, gether mapping from obj instance to a list of simulator obj
                obj_inst_to_obj_per_room_inst = {}
                for obj_inst in scene_object_scope_filtered[room_type]:
                    obj_inst_to_obj_per_room_inst[obj_inst] = scene_object_scope_filtered[room_type][obj_inst][
                        room_inst
                    ]
                top_nodes = []
                log_msg = "MBM for room instance [{}]".format(room_inst)
                logging.warning((log_msg))
                init_mbm_log.append(log_msg)
                for obj_inst in obj_inst_to_obj_per_room_inst:
                    for obj in obj_inst_to_obj_per_room_inst[obj_inst]:
                        # Create an edge between obj instance and each of the simulator obj that supports sampling
                        graph.add_edge(obj_inst, obj)
                        log_msg = "Adding edge: {} <-> {}".format(obj_inst, obj.name)
                        logging.warning((log_msg))
                        init_mbm_log.append(log_msg)
                        top_nodes.append(obj_inst)
                # Need to provide top_nodes that contain all nodes in one bipartite node set
                # The matches will have two items for each match (e.g. A -> B, B -> A)
                matches = nx.bipartite.maximum_matching(graph, top_nodes=top_nodes)
                if len(matches) == 2 * len(obj_inst_to_obj_per_room_inst):
                    logging.warning(("Object scope finalized:"))
                    for obj_inst, obj in matches.items():
                        if obj_inst in obj_inst_to_obj_per_room_inst:
                            self.object_scope[obj_inst] = obj
                            logging.warning((obj_inst, obj.name))
                    success = True
                    break
            if not success:
                error_msg = "Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n".format(
                    room_type, self.scene.scene_id
                )
                error_msg += "The following are the initial condition matching history:\n"
                error_msg += "\n".join(init_mbm_log)
                logging.warning(error_msg)
                feedback["init_success"] = "no"
                feedback["goal_success"] = "untested"
                feedback["init_feedback"] = error_msg
                return False, feedback

        np.random.shuffle(self.ground_goal_state_options)
        logging.warning(("number of ground_goal_state_options", len(self.ground_goal_state_options)))
        num_goal_condition_set_to_test = 10

        goal_sampling_error_msgs = []
        # Next, try to fulfill different set of ground goal conditions (maximum num_goal_condition_set_to_test)
        for goal_condition_set in self.ground_goal_state_options[:num_goal_condition_set_to_test]:
            goal_condition_set_success = True
            goal_sampling_log = []
            # Try to fulfill the current set of ground goal conditions
            scene_object_scope_filtered_goal_cond = {}
            for room_type in scene_object_scope_filtered:
                scene_object_scope_filtered_goal_cond[room_type] = {}
                for scene_obj in scene_object_scope_filtered[room_type]:
                    scene_object_scope_filtered_goal_cond[room_type][scene_obj] = {}
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
                                if goal_condition.STATE_NAME not in KINEMATICS_STATES:
                                    continue
                                if scene_obj not in goal_condition.body:
                                    continue
                                success = goal_condition.sample(binary_state=True)
                                log_msg = " ".join(
                                    [
                                        "goal condition sampling",
                                        room_type,
                                        scene_obj,
                                        room_inst,
                                        obj.name,
                                        goal_condition.STATE_NAME,
                                        str(goal_condition.body),
                                        str(success),
                                    ]
                                )
                                logging.warning((log_msg))
                                goal_sampling_log.append(log_msg)
                                if not success:
                                    break
                            if not success:
                                continue

                            if room_inst not in scene_object_scope_filtered_goal_cond[room_type][scene_obj]:
                                scene_object_scope_filtered_goal_cond[room_type][scene_obj][room_inst] = []
                            scene_object_scope_filtered_goal_cond[room_type][scene_obj][room_inst].append(obj)

            for room_type in scene_object_scope_filtered_goal_cond:
                # For each room_type, filter in room_inst that has successful
                # sampling options for all obj_inst in this room_type
                room_inst_satisfied = set.intersection(
                    *[
                        set(scene_object_scope_filtered_goal_cond[room_type][obj_inst].keys())
                        for obj_inst in scene_object_scope_filtered_goal_cond[room_type]
                    ]
                )

                if len(room_inst_satisfied) == 0:
                    error_msg = "Room type [{}] of scene [{}] cannot sample all the objects needed.\nThe following are the possible room instances for each object, the intersection of which is an empty set.\n".format(
                        room_type, self.scene.scene_id
                    )
                    for obj_inst in scene_object_scope_filtered_goal_cond[room_type]:
                        error_msg += (
                            "{}: ".format(obj_inst)
                            + ", ".join(scene_object_scope_filtered_goal_cond[room_type][obj_inst].keys())
                            + "\n"
                        )
                    error_msg += "The following are the goal condition sampling history:\n"
                    error_msg += "\n".join(goal_sampling_log)
                    logging.warning(error_msg)
                    goal_sampling_error_msgs.append(error_msg)
                    if self.should_debug_sampling:
                        self.debug_sampling(
                            scene_object_scope_filtered_goal_cond, non_sampleable_obj_conditions, goal_condition_set
                        )
                    goal_condition_set_success = False
                    break

                for obj_inst in scene_object_scope_filtered_goal_cond[room_type]:
                    scene_object_scope_filtered_goal_cond[room_type][obj_inst] = {
                        key: val
                        for key, val in scene_object_scope_filtered_goal_cond[room_type][obj_inst].items()
                        if key in room_inst_satisfied
                    }

            if not goal_condition_set_success:
                continue
            # For each room instance, perform maximum bipartite matching between object instance in scope to simulator objects
            # Left nodes: a list of object instance in scope
            # Right nodes: a list of simulator objects
            # Edges: if the simulator object can support the sampling requirement of ths object instance
            for room_type in scene_object_scope_filtered_goal_cond:
                # The same room instances will be shared across all scene obj in a given room type
                some_obj = list(scene_object_scope_filtered_goal_cond[room_type].keys())[0]
                room_insts = list(scene_object_scope_filtered_goal_cond[room_type][some_obj].keys())
                success = False
                goal_mbm_log = []
                # Loop through each room instance
                for room_inst in room_insts:
                    graph = nx.Graph()
                    # For this given room instance, gether mapping from obj instance to a list of simulator obj
                    obj_inst_to_obj_per_room_inst = {}
                    for obj_inst in scene_object_scope_filtered_goal_cond[room_type]:
                        obj_inst_to_obj_per_room_inst[obj_inst] = scene_object_scope_filtered_goal_cond[room_type][
                            obj_inst
                        ][room_inst]
                    top_nodes = []
                    log_msg = "MBM for room instance [{}]".format(room_inst)
                    logging.warning((log_msg))
                    goal_mbm_log.append(log_msg)
                    for obj_inst in obj_inst_to_obj_per_room_inst:
                        for obj in obj_inst_to_obj_per_room_inst[obj_inst]:
                            # Create an edge between obj instance and each of the simulator obj that supports sampling
                            graph.add_edge(obj_inst, obj)
                            log_msg = "Adding edge: {} <-> {}".format(obj_inst, obj.name)
                            logging.warning((log_msg))
                            goal_mbm_log.append(log_msg)
                            top_nodes.append(obj_inst)
                    # Need to provide top_nodes that contain all nodes in one bipartite node set
                    # The matches will have two items for each match (e.g. A -> B, B -> A)
                    matches = nx.bipartite.maximum_matching(graph, top_nodes=top_nodes)
                    if len(matches) == 2 * len(obj_inst_to_obj_per_room_inst):
                        logging.warning(("Object scope finalized:"))
                        for obj_inst, obj in matches.items():
                            if obj_inst in obj_inst_to_obj_per_room_inst:
                                self.object_scope[obj_inst] = obj
                                logging.warning((obj_inst, obj.name))
                        success = True
                        break
                if not success:
                    error_msg = "Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n".format(
                        room_type, self.scene.scene_id
                    )
                    error_msg += "The following are the goal condition matching history:\n"
                    error_msg += "\n".join(goal_mbm_log)
                    logging.warning(error_msg)
                    goal_sampling_error_msgs.append(error_msg)
                    goal_condition_set_success = False
                    break

            if not goal_condition_set_success:
                continue

            # if one set of goal conditions (and initial conditions) are satisfied, sampling is successful
            break

        if not goal_condition_set_success:
            goal_sampling_error_msg_compiled = ""
            for i, log_msg in enumerate(goal_sampling_error_msgs):
                goal_sampling_error_msg_compiled += "-" * 30 + "\n"
                goal_sampling_error_msg_compiled += "Ground condition set #{}/{}:\n".format(
                    i + 1, len(goal_sampling_error_msgs)
                )
                goal_sampling_error_msg_compiled += log_msg + "\n"
            feedback["goal_success"] = "no"
            feedback["goal_feedback"] = goal_sampling_error_msg_compiled
            return False, feedback

        # Do sampling again using the object instance -> simulator object mapping from maximum bipartite matching
        for condition, positive in non_sampleable_obj_conditions:
            num_trials = 10
            for _ in range(num_trials):
                success = condition.sample(binary_state=positive)
                if success:
                    break
            if not success:
                logging.warning(
                    "Non-sampleable object conditions failed even after successful matching: {}".format(condition.body)
                )
                feedback["init_success"] = "no"
                feedback["init_feedback"] = "Please run test sampling again."
                return False, feedback

        # Use ray casting for ontop and inside sampling for non-sampleable objects
        for condition, positive in sampleable_obj_conditions:
            if condition.STATE_NAME in ["inside", "ontop"]:
                condition.kwargs["use_ray_casting_method"] = True

        if len(self.sampling_orders) > 0:
            # Pop non-sampleable objects
            self.sampling_orders.pop(0)
            for cur_batch in self.sampling_orders:
                # First sample non-sliced conditions
                for condition, positive in sampleable_obj_conditions:
                    if condition.STATE_NAME == "sliced":
                        continue
                    # Sample conditions that involve the current batch of objects
                    if condition.body[0] in cur_batch:
                        num_trials = 100
                        for _ in range(num_trials):
                            success = condition.sample(binary_state=positive)
                            if success:
                                break
                        if not success:
                            error_msg = "Sampleable object conditions failed: {} {}".format(
                                condition.STATE_NAME, condition.body
                            )
                            logging.warning(error_msg)
                            feedback["init_success"] = "no"
                            feedback["init_feedback"] = error_msg
                            return False, feedback

                # Then sample non-sliced conditions
                for condition, positive in sampleable_obj_conditions:
                    if condition.STATE_NAME != "sliced":
                        continue
                    # Sample conditions that involve the current batch of objects
                    if condition.body[0] in cur_batch:
                        success = condition.sample(binary_state=positive)
                        if not success:
                            error_msg = "Sampleable object conditions failed: {}".format(condition.body)
                            logging.warning(error_msg)
                            feedback["init_success"] = "no"
                            feedback["init_feedback"] = error_msg
                            return False, feedback

        return True, feedback

    def debug_sampling(self, scene_object_scope_filtered, non_sampleable_obj_conditions, goal_condition_set=None):
        igibson.debug_sampling = True
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
                            print(
                                "debug initial condition sampling",
                                room_type,
                                scene_obj,
                                room_inst,
                                obj.name,
                                condition.STATE_NAME,
                                condition.body,
                            )
                            obj_pos = obj.get_position()
                            # Set the pybullet camera to have a bird's eye view
                            # of the sampling process
                            p.resetDebugVisualizerCamera(
                                cameraDistance=3.0, cameraYaw=0, cameraPitch=-89.99999, cameraTargetPosition=obj_pos
                            )
                            success = condition.sample(binary_state=positive)
                            print("success", success)

                        if goal_condition_set is None:
                            continue

                        for goal_condition in goal_condition_set:
                            goal_condition = goal_condition.children[0]
                            if isinstance(goal_condition, Negation):
                                continue
                            if goal_condition.STATE_NAME not in KINEMATICS_STATES:
                                continue
                            if scene_obj not in goal_condition.body:
                                continue
                            print(
                                "goal condition sampling",
                                room_type,
                                scene_obj,
                                room_inst,
                                obj.name,
                                goal_condition.STATE_NAME,
                                goal_condition.body,
                            )
                            obj_pos = obj.get_position()
                            # Set the pybullet camera to have a bird's eye view
                            # of the sampling process
                            p.resetDebugVisualizerCamera(
                                cameraDistance=3.0, cameraYaw=0, cameraPitch=-89.99999, cameraTargetPosition=obj_pos
                            )
                            success = goal_condition.sample(binary_state=True)
                            print("success", success)

    def clutter_scene(self):
        if not self.load_clutter:
            return

        scene_id = self.scene.scene_id
        clutter_ids = [""] + list(range(2, 5))
        clutter_id = np.random.choice(clutter_ids)
        clutter_scene = InteractiveIndoorScene(scene_id, "{}_clutter{}".format(scene_id, clutter_id))
        existing_objects = [value for key, value in self.object_scope.items() if "floor.n.01" not in key]
        self.simulator.import_non_colliding_objects(
            objects=clutter_scene.objects_by_name, existing_objects=existing_objects, min_distance=0.5
        )

    def get_task_obs(self, env):
        state = OrderedDict()
        task_obs = np.zeros((self.task_obs_dim))
        state["robot_pos"] = np.array(env.robots[0].get_position())
        state["robot_orn"] = np.array(env.robots[0].get_rpy())

        i = 0
        for _, v in self.object_scope.items():
            if isinstance(v, URDFObject):
                state["obj_{}_valid".format(i)] = 1.0
                state["obj_{}_pos".format(i)] = np.array(v.get_position())
                state["obj_{}_orn".format(i)] = np.array(p.getEulerFromQuaternion(v.get_orientation()))
                state["obj_{}_in_left_hand".format(i)] = float(
                    env.robots[0].parts["left_hand"].object_in_hand == v.get_body_id()
                )
                state["obj_{}_in_right_hand".format(i)] = float(
                    env.robots[0].parts["right_hand"].object_in_hand == v.get_body_id()
                )
                i += 1

        state_list = []
        for k, v in state.items():
            if isinstance(v, list):
                state_list.extend(v)
            elif isinstance(v, tuple):
                state_list.extend(list(v))
            elif isinstance(v, np.ndarray):
                state_list.extend(list(v))
            elif isinstance(v, (float, int)):
                state_list.append(v)
            else:
                raise ValueError("cannot serialize task obs")

        assert len(state_list) <= len(task_obs)
        task_obs[: len(state_list)] = state_list

        return task_obs


def main():
    igbhvr_act_inst = iGBEHAVIORActivityInstance("assembling_gift_baskets", 0)
    igbhvr_act_inst.initialize_simulator(mode="headless", scene_id="Rs_int")

    for i in range(500):
        igbhvr_act_inst.simulator.step()
    success, failed_conditions = igbhvr_act_inst.check_success()
    print("ACTIVITY SUCCESS:", success)
    if not success:
        print("FAILED CONDITIONS:", failed_conditions)
    igbhvr_act_inst.simulator.disconnect()


if __name__ == "__main__":
    main()
