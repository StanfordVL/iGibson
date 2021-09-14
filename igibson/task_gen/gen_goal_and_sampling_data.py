"""
python igibson/task_gen/gen_goal_and_sampling_data.py 2> logs/gen_goal_and_sampling_data.err > logs/gen_goal_and_sampling_data.out
"""

from collections import OrderedDict
import glob
import json
import logging; log = logging.getLogger(__name__)
import multiprocessing
import numpy as np
import os
import random
import re
import sys
from termcolor import colored
from tqdm import tqdm
import traceback
from typing import Any, Callable, Dict, List, Tuple, Optional

import bddl
from bddl.condition_evaluation import create_scope, compile_state, Negation
# from bddl.config import get_definition_filename 
from bddl.parsing import parse_domain, parse_problem
from IPython import embed

import igibson
from igibson import object_states
from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.objects.articulated_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.task_gen.util import set_seed, load_file, save_file, write_to_file, load_object, step, pcolored
import pybullet as p

from igibson.object_states.object_state_base import BaseObjectState, AbsoluteObjectState
from igibson.object_states.room_states import IsInRoomTemplate
IsInBathroom = type("IsInBathroom", (IsInRoomTemplate,), {"ROOM_TYPE": "bathroom"})
IsInBedroom = type("IsInBedroom", (IsInRoomTemplate,), {"ROOM_TYPE": "bedroom"})
IsInChildsRoom = type("IsInChildsRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "childs_room"})
IsInCloset = type("IsInCloset", (IsInRoomTemplate,), {"ROOM_TYPE": "closet"})
IsInCorridor = type("IsInCorridor", (IsInRoomTemplate,), {"ROOM_TYPE": "corridor"})
IsInDiningRoom = type("IsInDiningRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "dining_room"})
IsInEmptyRoom = type("IsInEmptyRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "empty_room"})
IsInExerciseRoom = type("IsInExerciseRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "exercise_room"})
IsInGarage = type("IsInGarage", (IsInRoomTemplate,), {"ROOM_TYPE": "garage"})
IsInHomeOffice = type("IsInHomeOffice", (IsInRoomTemplate,), {"ROOM_TYPE": "home_office"})
IsInKitchen = type("IsInKitchen", (IsInRoomTemplate,), {"ROOM_TYPE": "kitchen"})
IsInLivingRoom = type("IsInLivingRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "living_room"})
IsInLobby = type("IsInLobby", (IsInRoomTemplate,), {"ROOM_TYPE": "lobby"})
IsInPantryRoom = type("IsInPantryRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "pantry_room"})
IsInPlayroom = type("IsInPlayroom", (IsInRoomTemplate,), {"ROOM_TYPE": "playroom"})
IsInStaircase = type("IsInStaircase", (IsInRoomTemplate,), {"ROOM_TYPE": "staircase"})
IsInStorageRoom = type("IsInStorageRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "storage_room"})
IsInTelevisionRoom = type("IsInTelevisionRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "television_room"})
IsInUtilityRoom = type("IsInUtilityRoom", (IsInRoomTemplate,), {"ROOM_TYPE": "utility_room"})
IsInBalcony = type("IsInBalcony", (IsInRoomTemplate,), {"ROOM_TYPE": "balcony"})
IsInLibrary = type("IsInLibrary", (IsInRoomTemplate,), {"ROOM_TYPE": "library"})
IsInAuditorium = type("IsInAuditorium", (IsInRoomTemplate,), {"ROOM_TYPE": "auditorium"})
IsInUndefined = type("IsInUndefined", (IsInRoomTemplate,), {"ROOM_TYPE": "undefined"})


bddl.set_backend("iGibson")

activity_to_scenes = load_file(os.path.join(os.path.dirname(bddl.__file__), "..", "utils", "activity_to_preselected_scenes.json"))
defined_activities = os.listdir(os.path.join(os.path.dirname(bddl.__file__), 'activity_definitions'))
activity_to_scenes = {k: v for k, v in activity_to_scenes.items() if k in defined_activities}
# try:
# 	os.remove('logs/gen_goal_and_sampling_data.out')
# except:
# 	pass
# try:
# 	os.remove('logs/gen_goal_and_sampling_data.err')
# except:
# 	pass

KINEMATICS_STATES = frozenset({"inside", "ontop", "under", "onfloor"})

def sample_goal_state_for_activity_scene(activity_name, activity_id, scene_id, sample_idx=0):
	inst = iGBEHAVIORActivityInstance(activity_name, activity_definition=activity_id) # TODO should I create a new one each time?
	scene_kwargs = {
		"not_load_object_categories": ["ceilings"],
	}
	settings = MeshRendererSettings(texture_scale=1)
	simulator = Simulator(mode="headless", image_width=960, image_height=720, rendering_settings=settings)
	init_success = inst.initialize_simulator(
		scene_id=scene_id, 
		simulator=simulator, 
		load_clutter=False, 
		should_debug_sampling=True, 
		scene_kwargs=scene_kwargs,
		online_sampling=True,
		do_sample=False,
	)
	if not init_success:
		write_to_file('logs/gen_goal_and_sampling_data.err',
			f'Failed to initialize activity instance for activity={activity_name}, activity_id={activity_id}, scene={scene_id}, sample_idx={sample_idx}')
		return None
	success, goal_condition_set_success, goal_sampling_error_msgs = inst.assign_scope_for_goal_conditions()
	if not success:
		write_to_file('logs/gen_goal_and_sampling_data.err',
			f"""Failed to assign goal scope for activity={activity_name}, activity_id={activity_id}, scene={scene_id}, sample_idx={sample_idx}.
Goal conditions: {goal_condition_set_success}
Goal error messages:
{goal_sampling_error_msgs}
---
""")
		return None
	success, feedback = inst.sample_goal_conditions()
	if not success:
		write_to_file('logs/gen_goal_and_sampling_data.err',
			f"""Failed to sample goal conditions for activity={activity_name}, activity_id={activity_id}, scene={scene_id}, sample_idx={sample_idx}.
Feedback:
{feedback}
---
""")
		return None
	success, sorted_conditions = inst.check_success()
	if not success:
		write_to_file('logs/gen_goal_and_sampling_data.err',
			f"""Goal conditions not satisfied for activity={activity_name}, activity_id={activity_id}, scene={scene_id}, sample_idx={sample_idx}.
Sorted conditions:
{sorted_conditions}
---
""")
		return None
	return inst


def traverse_bddl(bddl_cond, func, reduce_func=None):
	if isinstance(bddl_cond, list):
		res = []
		for c in bddl_cond:
			res.extend(traverse_bddl(c, func, reduce_func))
		if reduce_func is not None:
			return [reduce_func(res)]
		return res
	res = [func(bddl_cond)]
	for c in bddl_cond.children:
		res.extend(traverse_bddl(c, func, reduce_func))
	if reduce_func is not None:
		return [reduce_func(res)]
	return res

def has_negated_kinematic(bddl_cond, inside_negation=False):
	if isinstance(bddl_cond, list):
		for c in bddl_cond:
			if has_negated_kinematic(c, inside_negation=inside_negation):
				return True
		return False
	if isinstance(bddl_cond, Negation):
		inside_negation = not inside_negation
	elif inside_negation and hasattr(bddl_cond, 'STATE_NAME') and bddl_cond.STATE_NAME in KINEMATICS_STATES:
		return True
	for c in bddl_cond.children:
		if has_negated_kinematic(c, inside_negation=inside_negation):
			return True
	return False

def has_kinematic(bddl_cond):
	if isinstance(bddl_cond, list):
		for c in bddl_cond:
			if has_kinematic(c):
				return True
		return False
	if hasattr(bddl_cond, 'STATE_NAME') and bddl_cond.STATE_NAME in KINEMATICS_STATES:
		return True
	for c in bddl_cond.children:
		if has_kinematic(c):
			return True
	return False

def check_goal_sampleable(activity_name, activity_id=0):
	"""Filters out any tasks that have NextTo, which is not implemented.
	sorting_groceries -> False
	preserving_food -> True
	"""
	# print(activity_name)
	inst = iGBEHAVIORActivityInstance(activity_name, activity_definition=activity_id)
	def is_implemented(cond):
		try:
			cond.STATE_CLASS._set_value(None, None, None)
		except NotImplementedError:
			return False
		except Exception as e:
			# print(e)
			pass
		return True
	if has_negated_kinematic(inst.goal_conditions):
		return False
	if not has_kinematic(inst.goal_conditions):
		return False
	return traverse_bddl(inst.goal_conditions, is_implemented, all)[0]


def process_activities():
	processed_files = glob.glob(os.path.join(
						os.path.dirname(os.path.dirname(igibson.ig_dataset_path)),
						'derived_data',
						'scenes',
						'*',
						'*',
						'*.p'
					))
	activities = list(sorted(set([fname.split('/')[-2] for fname in processed_files])))
	set_seed()
	np.random.shuffle(activities)
	print(activities)
	print(len(activities))
	for activity_name in (activities): # reversed(sorted(activity_to_scenes.keys())):
		# print(activity_name)
		# if activity_name != 'packing_food_for_work':
		# if activity_name != 'preserving_food':
		# 	continue
		for activity_id in range(1):
			if not check_goal_sampleable(activity_name, activity_id):
				write_to_file('logs/gen_goal_and_sampling_data.out',
					f'Skipping activity={activity_name}, activity_id={activity_id}; not sampleable')
				continue
			scene_choices = activity_to_scenes[activity_name]
			for scene_id in scene_choices:
				set_seed()
				for sample_idx in range(10):

					# "{}_task_{}_{}_{}".format(scene_id, task, task_id, num_init)
					filename = os.path.join(
						os.path.dirname(os.path.dirname(igibson.ig_dataset_path)),
						'derived_data',
						'scenes',
						scene_id,
						activity_name,
						f'{scene_id}_{activity_name}_{activity_id}_{sample_idx}_scene_graph_v0.p'
					)
					urdf_file = filename.replace('_scene_graph_v0.p', '_v0.urdf')
					if not os.path.isdir(os.path.dirname(filename)):
						os.makedirs(os.path.dirname(filename), exist_ok=True)
					if os.path.isfile(filename) and os.path.isfile(urdf_file):
						continue

					write_to_file('logs/gen_goal_and_sampling_data.out',
						f'Processing activity={activity_name}, activity_id={activity_id}, scene={scene_id}, sample_idx={sample_idx}')
					try:
						inst = sample_goal_state_for_activity_scene(activity_name, activity_id, scene_id)
					except Exception as e:
						write_to_file('logs/gen_goal_and_sampling_data.err', str(e))
						traceback.print_exc()
						# embed()
						continue
					if inst is None:
						continue
					try:
						scene_graph = inst.get_scene_graph()
						
						# with open(filename, 'w') as f:
						# 	# json.dump(scene_graph, f)
						# with open('data.pkl', 'wb') as f:
						# 	pickle.dump(scene_graph, f)
						save_file(filename, scene_graph)
						write_to_file('logs/gen_goal_and_sampling_data.out',
							f'Wrote file at {filename}')
						sim_obj_to_bddl_obj = {
							value.name: {"object_scope": key} for key, value in inst.object_scope.items()
						}
						inst.scene.save_modified_urdf(urdf_file, sim_obj_to_bddl_obj)
						write_to_file('logs/gen_goal_and_sampling_data.out',
							f'Wrote file at {urdf_file}')
						# logging.warning(("Saved:", urdf_file))
					except Exception as e:
						write_to_file('logs/gen_goal_and_sampling_data.err', str(e))
						traceback.print_exc()
						# embed()
						continue
				# break

def load_scene(scene_id, activity_name, activity_id, sample_idx):
    inst = iGBEHAVIORActivityInstance(activity_name, activity_definition=activity_id)
    urdf_file = os.path.join(
        os.path.dirname(os.path.dirname(igibson.ig_dataset_path)),
        'derived_data',
        'scenes',
        scene_id,
        activity_name,
        f'{scene_id}_{activity_name}_{activity_id}_{sample_idx}_v0.urdf.urdf'  # TODO rename ext
    )
    scene_kwargs = {
        "not_load_object_categories": ["ceilings"],
        "urdf_file": urdf_file,
    }
    settings = MeshRendererSettings(texture_scale=1)
    simulator = Simulator(mode="iggui", image_width=960, image_height=720, rendering_settings=settings)
    init_success = inst.initialize_simulator(
        scene_id=scene_id, 
        simulator=simulator, 
        load_clutter=False, 
        should_debug_sampling=False, 
        scene_kwargs=scene_kwargs,
        online_sampling=False,
        do_sample=False,
    )
    while True:
        simulator.step()
        success, sorted_conditions = inst.check_success()
        print("TASK SUCCESS:", success)
        if not success:
            print("FAILED CONDITIONS:", sorted_conditions["unsatisfied"])
        else:
            pass


def load_empty_scene(scene_id, activity_name, activity_id, sample_idx):
	simulator = Simulator(mode="pbgui", image_width=960, image_height=720)
	scene = EmptyScene()
	simulator.import_scene(scene)
	p.setGravity(0, 0, 0)

	sg_file = os.path.join(
		os.path.dirname(os.path.dirname(igibson.ig_dataset_path)),
		'derived_data',
		'scenes',
		scene_id,
		activity_name,
		f'{scene_id}_{activity_name}_{activity_id}_{sample_idx}_scene_graph_v0.p'
	)
	scene_graph = load_file(sg_file)
	for obj_inst in scene_graph:
		try:
			pcolored(obj_inst)
			obj_dict = scene_graph[obj_inst]
			category = obj_dict['category']
			model_id = obj_dict['filename'].split('/')[-2] # TODO check
			pos = obj_dict['pos']
			orn = obj_dict['orn']
			scale = obj_dict['scale']
			if scale is None:
				bbox = obj_dict['bounding_box']
			else:
				bbox = None
			obj = load_object(simulator, category, model_id, position=pos, orientation=orn, bounding_box=bbox, scale=scale)
			if 'states' in obj_dict:
				states = obj_dict['states']
				for state, value in states.items():
					obj[state].set_value(value)
			# embed()
		except Exception:
			traceback.print_exc()
	p.setGravity(0, 0, -9.8)
	pcolored(sg_file)
	while True:
		step(simulator, nstep=1)

if __name__ == '__main__':
	processed_files = glob.glob(os.path.join(
						os.path.dirname(os.path.dirname(igibson.ig_dataset_path)),
						'derived_data',
						'scenes',
						'*',
						'*',
						'*.p'
					))
	activities = list(filter(check_goal_sampleable, sorted(set([fname.split('/')[-2] for fname in processed_files]))))
	print(activities)
	print(len(activities))
	# load_scene('Pomaria_1_int', 'washing_pots_and_pans', 0, 0)
	# load_empty_scene('Pomaria_1_int', 'washing_pots_and_pans', 0, 0)
	# load_empty_scene('Benevolence_2_int', 'brushing_lint_off_clothing', 0, 0)
	# load_empty_scene('Merom_1_int', 'collect_misplaced_items', 0, 0)
	# load_empty_scene('Ihlen_1_int', 'unpacking_suitcase', 0, 0)
	# sample_goal_state_for_activity_scene('sorting_books', 0, 'Rs_int', sample_idx=0)
	process_activities()

# ['bringing_in_wood', 'brushing_lint_off_clothing', 'cleaning_shoes', 'cleaning_the_hot_tub', 'cleaning_the_pool', 'cleanin
# g_toilet', 'collect_misplaced_items', 'organizing_file_cabinet', 'polishing_furniture', 'putting_dishes_away_after_cleanin
# g', 'putting_leftovers_away', 're-shelving_library_books', 'sorting_books', 'storing_food', 'unpacking_suitcase', 'washing
# _pots_and_pans']
