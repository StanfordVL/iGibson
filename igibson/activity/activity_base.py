import logging
from collections import OrderedDict

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
from igibson.robots.fetch_gripper_robot import FetchGripper
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
from igibson.utils.utils import restoreState

KINEMATICS_STATES = frozenset({"inside", "ontop", "under", "onfloor"})


class iGBEHAVIORActivityInstance(BEHAVIORActivityInstance):
    def __init__(
        self,
        behavior_activity,
        activity_definition=0,
        predefined_problem=None,
        robot_type=BehaviorRobot,
        robot_config={},
    ):
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
        self.robot_type = robot_type
        self.robot_config = robot_config

    def initialize_simulator(
        self,
        simulator=None,
        mode="headless",
        scene_id=None,
        scene_kwargs=None,
        load_clutter=False,
        online_sampling=True,
        debug_obj_inst=None,
    ):
        """
        Get scene populated with objects such that scene satisfies initial conditions
        :param simulator: Simulator class, populated simulator that should completely
                                   replace this function. Use if you would like to bypass internal
                                   Simulator instantiation and population based on initial conditions
                                   and use your own. Warning that if you use this option, we cannot
                                   guarantee that the final conditions will be reachable.
        """
        if simulator is None:
            self.simulator = Simulator(mode=mode, image_width=960, image_height=720, device_idx=0)
        else:
            self.simulator = simulator
        self.load_clutter = load_clutter
        self.debug_obj_inst = debug_obj_inst
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
        restoreState(snapshot_id)
        load_internal_states(self.simulator, self.state_history[snapshot_id])

    def parse_non_sampleable_object_room_assignment(self):
        self.room_type_to_obj_inst = {}
        self.non_sampleable_object_inst = set()
        for cond in self.parsed_initial_conditions:
            if cond[0] == "inroom":
                obj_inst, room_type = cond[1], cond[2]
                obj_cat = self.obj_inst_to_obj_cat[obj_inst]
                if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                    # Invalid room assignment
                    return "You have assigned room type for [{}], but [{}] is sampleable. Only non-sampleable objects can have room assignment.".format(
                        obj_cat, obj_cat
                    )
                if room_type not in self.scene.room_sem_name_to_ins_name:
                    # Missing room type
                    return "Room type [{}] missing in scene [{}].".format(room_type, self.scene.scene_id)
                if room_type not in self.room_type_to_obj_inst:
                    self.room_type_to_obj_inst[room_type] = []
                self.room_type_to_obj_inst[room_type].append(obj_inst)

                if obj_inst in self.non_sampleable_object_inst:
                    # Duplicate room assignment
                    return "Object [{}] has more than one room assignment".format(obj_inst)

                self.non_sampleable_object_inst.add(obj_inst)

        for obj_cat in self.objects:
            if obj_cat not in NON_SAMPLEABLE_OBJECTS:
                continue
            for obj_inst in self.objects[obj_cat]:
                if obj_inst not in self.non_sampleable_object_inst:
                    # Missing room assignment
                    return "All non-sampleable objects should have room assignment. [{}] does not have one.".format(
                        obj_inst
                    )

    def build_sampling_order(self):
        """
        Sampling orders is a list of lists: [[batch_1_inst_1, ... batch_1_inst_N], [batch_2_inst_1, batch_2_inst_M], ...]
        Sampling should happen for batch 1 first, then batch 2, so on and so forth
        Example: OnTop(plate, table) should belong to batch 1, and OnTop(apple, plate) should belong to batch 2
        """
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
            return "Some objects do not have any kinematic condition defined for them in the initial conditions: {}".format(
                ", ".join(remaining_objs)
            )

    def build_non_sampleable_object_scope(self):
        """
        Store simulator object options for non-sampleable objects in self.non_sampleable_object_scope
        {
            "table1": {
                "living_room_0": [URDFObject, URDFObject, URDFObject],
                "living_room_1": [URDFObject]
            },
            "table2": {
                "living_room_0": [URDFObject, URDFObject],
                "living_room_1": [URDFObject, URDFObject]
            },
            "chair1": {
                "living_room_0": [URDFObject],
                "living_room_1": [URDFObject]
            },
        }
        """
        room_type_to_scene_objs = {}
        for room_type in self.room_type_to_obj_inst:
            room_type_to_scene_objs[room_type] = {}
            for obj_inst in self.room_type_to_obj_inst[room_type]:
                room_type_to_scene_objs[room_type][obj_inst] = {}
                obj_cat = self.obj_inst_to_obj_cat[obj_inst]

                # We allow burners to be used as if they are stoves
                categories = self.object_taxonomy.get_subtree_igibson_categories(obj_cat)
                if obj_cat == "stove.n.01":
                    categories += self.object_taxonomy.get_subtree_igibson_categories("burner.n.02")

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
                        # A list of scene objects that satisfy the requested categories
                        scene_objs = [obj for obj in room_objs if obj.category in categories]

                    if len(scene_objs) != 0:
                        room_type_to_scene_objs[room_type][obj_inst][room_inst] = scene_objs

        error_msg = self.consolidate_room_instance(room_type_to_scene_objs, "initial_pre-sampling")
        if error_msg:
            return error_msg
        self.non_sampleable_object_scope = room_type_to_scene_objs

    def import_sampleable_objects(self):
        self.newly_added_objects = set()
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

            # for sliceable objects, exclude the half_XYZ categories
            if is_sliceable:
                categories = [cat for cat in categories if "half_" not in cat]

            for obj_inst in self.objects[obj_cat]:
                category = np.random.choice(categories)
                category_path = get_ig_category_path(category)
                model_choices = os.listdir(category_path)

                # Filter object models if the object category is openable
                synset = self.object_taxonomy.get_class_name_from_igibson_category(category)
                if self.object_taxonomy.has_ability(synset, "openable"):
                    # Always use the articulated version of a certain object if its category is openable
                    # E.g. backpack, jar, etc
                    model_choices = [m for m in model_choices if "articulated_" in m]
                    if len(model_choices) == 0:
                        return "{} is Openable, but does not have articulated models.".format(category)

                # Randomly select an object model
                model = np.random.choice(model_choices)

                # TODO: temporary hack
                # for "collecting aluminum cans", we need pop cans (not bottles)
                if category == "pop" and self.behavior_activity in ["collecting_aluminum_cans"]:
                    model = np.random.choice([str(i) for i in range(40, 46)])
                if category == "spoon" and self.behavior_activity in ["polishing_silver"]:
                    model = np.random.choice([str(i) for i in [2, 5, 6]])

                model_path = get_ig_model_path(category, model)
                filename = os.path.join(model_path, model + ".urdf")
                obj_name = "{}_{}".format(category, len(self.scene.objects_by_name))
                # create the object and set the initial position to be far-away
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
                        # create the object part (or half object) and set the initial position to be far-away
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

                # Load the object into the simulator
                if not self.scene.loaded:
                    self.scene.add_object(simulator_obj)
                else:
                    self.simulator.import_object(simulator_obj)
                self.newly_added_objects.add(simulator_obj)
                self.object_scope[obj_inst] = simulator_obj

    def check_scene(self):
        error_msg = self.parse_non_sampleable_object_room_assignment()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        error_msg = self.build_sampling_order()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        error_msg = self.build_non_sampleable_object_scope()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        error_msg = self.import_sampleable_objects()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        return True, None

    def import_agent(self):
        cached_initial_pose = not self.online_sampling and self.scene.agent != {}
        if self.robot_type == BehaviorRobot:
            agent = BehaviorRobot(self.simulator)
            self.simulator.import_behavior_robot(agent)
            agent.set_position_orientation([300, 300, 300], [0, 0, 0, 1])
            self.object_scope["agent.n.01_1"] = agent.parts["body"]
            if cached_initial_pose:
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
        elif self.robot_type == FetchGripper:
            agent = FetchGripper(self.simulator, self.robot_config)
            self.simulator.import_robot(agent)
            agent.set_position_orientation([300, 300, 300], [0, 0, 0, 1])
            self.object_scope["agent.n.01_1"] = agent
            if cached_initial_pose:
                assert "fetch_gripper_robot_1" in self.scene.agent, "fetch gripper missing from scene cache"
                agent.set_position_orientation(
                    self.scene.agent["fetch_gripper_robot_1"]["xyz"],
                    quat_from_euler(self.scene.agent["fetch_gripper_robot_1"]["rpy"]),
                )
        else:
            Exception("Only BehaviorRobot and FetchGripper are supported")

        agent.robot_specific_reset()
        self.simulator.robots = [agent]
        assert len(self.simulator.robots) == 1, "Error, multiple agents is not currently supported"

    def move_agent(self):
        """
        Backwards compatibility, to be deprecated
        """
        pass

    def import_scene(self):
        self.simulator.reload()
        self.simulator.import_ig_scene(self.scene)

        # Assign object_scope based on a cached scene
        if not self.online_sampling:
            for obj_inst in self.object_scope:
                matched_sim_obj = None
                # TODO: remove after split floors
                if "floor.n.01" in obj_inst:
                    floor_obj = self.scene.objects_by_name["floors"]
                    bddl_object_scope = floor_obj.bddl_object_scope.split(",")
                    bddl_object_scope = {item.split(":")[0]: item.split(":")[1] for item in bddl_object_scope}
                    assert obj_inst in bddl_object_scope
                    room_inst = bddl_object_scope[obj_inst].replace("room_floor_", "")
                    matched_sim_obj = RoomFloor(
                        category="room_floor",
                        name=bddl_object_scope[obj_inst],
                        scene=self.scene,
                        room_instance=room_inst,
                        floor_obj=floor_obj,
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

    def process_single_condition(self, condition):
        if not isinstance(condition.children[0], Negation) and not isinstance(condition.children[0], AtomicFormula):
            logging.warning(("Skipping over sampling of predicate that is not a negation or an atomic formula"))
            return None, None

        if isinstance(condition.children[0], Negation):
            condition = condition.children[0].children[0]
            positive = False
        else:
            condition = condition.children[0]
            positive = True

        return condition, positive

    def group_initial_conditions(self):
        self.non_sampleable_obj_conditions = []
        self.sampleable_obj_conditions = []

        # TODO: currently we assume self.initial_conditions is a list of
        # bddl.condition_evaluation.HEAD, each with one child.
        # This chid is either a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate or
        # a Negation of a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate
        for condition in self.initial_conditions:
            condition, positive = self.process_single_condition(condition)
            if condition is None:
                continue

            if condition.STATE_NAME in KINEMATICS_STATES and not positive:
                return "Initial condition has negative kinematic conditions: {}".format(condition.body)

            condition_body = set(condition.body)
            if len(self.non_sampleable_object_inst.intersection(condition_body)) > 0:
                self.non_sampleable_obj_conditions.append((condition, positive))
            else:
                self.sampleable_obj_conditions.append((condition, positive))

    def filter_object_scope(self, input_object_scope, conditions, condition_type):
        filtered_object_scope = {}
        for room_type in input_object_scope:
            filtered_object_scope[room_type] = {}
            for scene_obj in input_object_scope[room_type]:
                filtered_object_scope[room_type][scene_obj] = {}
                for room_inst in input_object_scope[room_type][scene_obj]:
                    # These are a list of candidate simulator objects that need sampling test
                    for obj in input_object_scope[room_type][scene_obj][room_inst]:
                        # Temporarily set object_scope to point to this candidate object
                        self.object_scope[scene_obj] = obj

                        success = True
                        # If this candidate object is not involved in any conditions,
                        # success will be True by default and this object will qualify
                        for condition, positive in conditions:
                            # Sample positive kinematic conditions that involve this candidate object
                            if condition.STATE_NAME in KINEMATICS_STATES and positive and scene_obj in condition.body:
                                # Use pybullet GUI for debugging
                                if self.debug_obj_inst is not None and self.debug_obj_inst == condition.body[0]:
                                    igibson.debug_sampling = True
                                    obj_pos = obj.get_position()
                                    # Set the camera to have a bird's eye view of the sampling process
                                    p.resetDebugVisualizerCamera(
                                        cameraDistance=3.0,
                                        cameraYaw=0,
                                        cameraPitch=-89.99999,
                                        cameraTargetPosition=obj_pos,
                                    )

                                success = condition.sample(binary_state=positive)
                                log_msg = " ".join(
                                    [
                                        "{} condition sampling".format(condition_type),
                                        room_type,
                                        scene_obj,
                                        room_inst,
                                        obj.name,
                                        condition.STATE_NAME,
                                        str(condition.body),
                                        str(success),
                                    ]
                                )
                                logging.warning(log_msg)

                                # If any condition fails for this candidate object, skip
                                if not success:
                                    break

                        # If this candidate object fails, move on to the next candidate object
                        if not success:
                            continue

                        if room_inst not in filtered_object_scope[room_type][scene_obj]:
                            filtered_object_scope[room_type][scene_obj][room_inst] = []
                        filtered_object_scope[room_type][scene_obj][room_inst].append(obj)

        return filtered_object_scope

    def consolidate_room_instance(self, filtered_object_scope, condition_type):
        for room_type in filtered_object_scope:
            # For each room_type, filter in room_inst that has successful
            # sampling options for all obj_inst in this room_type
            room_inst_satisfied = set.intersection(
                *[
                    set(filtered_object_scope[room_type][obj_inst].keys())
                    for obj_inst in filtered_object_scope[room_type]
                ]
            )

            if len(room_inst_satisfied) == 0:
                error_msg = "{}: Room type [{}] of scene [{}] do not contain or cannot sample all the objects needed.\nThe following are the possible room instances for each object, the intersection of which is an empty set.\n".format(
                    condition_type, room_type, self.scene.scene_id
                )
                for obj_inst in filtered_object_scope[room_type]:
                    error_msg += (
                        "{}: ".format(obj_inst) + ", ".join(filtered_object_scope[room_type][obj_inst].keys()) + "\n"
                    )

                return error_msg

            for obj_inst in filtered_object_scope[room_type]:
                filtered_object_scope[room_type][obj_inst] = {
                    key: val
                    for key, val in filtered_object_scope[room_type][obj_inst].items()
                    if key in room_inst_satisfied
                }

    def maximum_bipartite_matching(self, filtered_object_scope, condition_type):
        # For each room instance, perform maximum bipartite matching between object instance in scope to simulator objects
        # Left nodes: a list of object instance in scope
        # Right nodes: a list of simulator objects
        # Edges: if the simulator object can support the sampling requirement of ths object instance
        for room_type in filtered_object_scope:
            # The same room instances will be shared across all scene obj in a given room type
            some_obj = list(filtered_object_scope[room_type].keys())[0]
            room_insts = list(filtered_object_scope[room_type][some_obj].keys())
            success = False
            # Loop through each room instance
            for room_inst in room_insts:
                graph = nx.Graph()
                # For this given room instance, gether mapping from obj instance to a list of simulator obj
                obj_inst_to_obj_per_room_inst = {}
                for obj_inst in filtered_object_scope[room_type]:
                    obj_inst_to_obj_per_room_inst[obj_inst] = filtered_object_scope[room_type][obj_inst][room_inst]
                top_nodes = []
                log_msg = "MBM for room instance [{}]".format(room_inst)
                logging.warning((log_msg))
                for obj_inst in obj_inst_to_obj_per_room_inst:
                    for obj in obj_inst_to_obj_per_room_inst[obj_inst]:
                        # Create an edge between obj instance and each of the simulator obj that supports sampling
                        graph.add_edge(obj_inst, obj)
                        log_msg = "Adding edge: {} <-> {}".format(obj_inst, obj.name)
                        logging.warning((log_msg))
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
                return "{}: Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n".format(
                    condition_type, room_type, self.scene.scene_id
                )

    def sample_conditions(self, input_object_scope, conditions, condition_type):
        filtered_object_scope = self.filter_object_scope(input_object_scope, conditions, condition_type)
        error_msg = self.consolidate_room_instance(filtered_object_scope, condition_type)
        if error_msg:
            return error_msg, None
        return self.maximum_bipartite_matching(filtered_object_scope, condition_type), filtered_object_scope

    def sample_initial_conditions(self):
        error_msg, self.non_sampleable_object_scope_filtered_initial = self.sample_conditions(
            self.non_sampleable_object_scope, self.non_sampleable_obj_conditions, "initial"
        )
        return error_msg

    def sample_goal_conditions(self):
        np.random.shuffle(self.ground_goal_state_options)
        logging.warning(("number of ground_goal_state_options", len(self.ground_goal_state_options)))
        num_goal_condition_set_to_test = 10

        goal_condition_success = False
        # Try to fulfill different set of ground goal conditions (maximum num_goal_condition_set_to_test)
        for goal_condition_set in self.ground_goal_state_options[:num_goal_condition_set_to_test]:
            goal_condition_processed = []
            for condition in goal_condition_set:
                condition, positive = self.process_single_condition(condition)
                if condition is None:
                    continue
                goal_condition_processed.append((condition, positive))

            error_msg, _ = self.sample_conditions(
                self.non_sampleable_object_scope_filtered_initial, goal_condition_processed, "goal"
            )
            if not error_msg:
                # if one set of goal conditions (and initial conditions) are satisfied, sampling is successful
                goal_condition_success = True
                break

        if not goal_condition_success:
            return error_msg

    def sample_initial_conditions_final(self):
        # Do the final round of sampling with object scope fixed
        for condition, positive in self.non_sampleable_obj_conditions:
            num_trials = 10
            for _ in range(num_trials):
                success = condition.sample(binary_state=positive)
                if success:
                    break
            if not success:
                error_msg = "Non-sampleable object conditions failed even after successful matching: {}".format(
                    condition.body
                )
                return error_msg

        # Use ray casting for ontop and inside sampling for sampleable objects
        for condition, positive in self.sampleable_obj_conditions:
            if condition.STATE_NAME in ["inside", "ontop"]:
                condition.kwargs["use_ray_casting_method"] = True

        if len(self.sampling_orders) > 0:
            # Pop non-sampleable objects
            self.sampling_orders.pop(0)
            for cur_batch in self.sampling_orders:
                # First sample non-sliced conditions
                for condition, positive in self.sampleable_obj_conditions:
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
                            return "Sampleable object conditions failed: {} {}".format(
                                condition.STATE_NAME, condition.body
                            )

                # Then sample non-sliced conditions
                for condition, positive in self.sampleable_obj_conditions:
                    if condition.STATE_NAME != "sliced":
                        continue
                    # Sample conditions that involve the current batch of objects
                    if condition.body[0] in cur_batch:
                        success = condition.sample(binary_state=positive)
                        if not success:
                            return "Sampleable object conditions failed: {}".format(condition.body)

    def sample(self, validate_goal=True):
        error_msg = self.group_initial_conditions()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        error_msg = self.sample_initial_conditions()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        if validate_goal:
            error_msg = self.sample_goal_conditions()
            if error_msg:
                logging.warning(error_msg)
                return False, error_msg

        error_msg = self.sample_initial_conditions_final()
        if error_msg:
            logging.warning(error_msg)
            return False, error_msg

        return True, None

    def clutter_scene(self):
        """
        Load clutter objects into the scene from a random clutter scene
        """
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
                grasping_objects = env.robots[0].is_grasping(v.get_body_id())
                for grasp_idx, grasping in enumerate(grasping_objects):
                    state["obj_{}_pos_in_gripper_{}".format(i, grasp_idx)] = float(grasping)
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
