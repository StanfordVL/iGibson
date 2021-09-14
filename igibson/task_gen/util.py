from collections import OrderedDict
import json
import logging; log = logging.getLogger(__name__)
import multiprocessing
import numpy as np
import os
import pickle
import random
import re
import sys
from termcolor import colored
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple, Optional

import igibson
from igibson.objects.articulated_object import URDFObject


def load_file(filename):
	if filename.endswith('.txt'):
		with open(filename) as f:
			lines = list(map(str.rstrip, f.readlines()))
			return lines
	if filename.endswith('.json'):
		with open(filename) as f:
			obj = json.load(f)
		return obj
		# return json.loads(filename)
	# if filename.endswith('.json.gz'):
	# 	with gzip.open(filename, 'rt') as f:
	# 		return json.load(f)
	if filename.endswith('.pkl') or filename.endswith('.p'):
		with open(filename, 'rb') as f:
			obj = pickle.load(f)
		return obj
	raise Exception(f'File name {filename} ends with unrecognized extension.')


def save_file(filename, obj):
	if filename.endswith('.txt'):
		with open(filename, 'w') as f:
			for line in obj:
				f.write(line + '\n')
	elif filename.endswith('.json'):
		with open(filename, 'w') as f:
			json.dump(obj, f)
	elif filename.endswith('.pkl') or filename.endswith('.p'):
		with open(filename, 'wb') as f:
			pickle.dump(obj, f, protocol=-1)  # highest protocol
	else:
		raise Exception(f'File name {filename} ends with unrecognized extension.')


def write_to_file(filename, s):
	print(colored(s, 'green'))
	with open(filename, 'a') as f:
		f.write(s + '\n')


def set_seed(seed: int = 0):
	seed = seed % 2147483647 # seed must be <= 2**32-1 # largest prime under 2**31
	random.seed(seed)
	np.random.seed(seed)
	# tf.random.set_seed(seed)
	# torch.manual_seed(seed)


def run_parallel(func, dataset, N_PARALLEL=8):
	with multiprocessing.Pool(N_PARALLEL) as p:
		list(tqdm(p.imap(func, dataset), total=len(dataset)))


def get_category_dir(category):
    return os.path.join(igibson.ig_dataset_path, "objects", category)


def load_object(simulator, category, model_id, position=None, orientation=None, bounding_box=None, scale=None, scene_object=None, relation=None, nstep=100):
    fname = os.path.join(igibson.ig_dataset_path, "objects", category, model_id, model_id + ".urdf")
    obj = URDFObject(filename=fname, category=category, model_path=os.path.dirname(fname), bounding_box=bounding_box, scale=scale)
    simulator.import_object(obj)
    if position is not None:
        obj.set_position(position)
    if orientation is not None:
        obj.set_orientation(orientation)
    elif relation is not None:
        obj.states[relation].set_value(scene_object, True, use_ray_casting_method=True)
    step(simulator, nstep)
    pcolored(f"Loaded {category}/{model_id}")
    return obj


def step(simulator, nstep=100):
    for _ in range(100):
        simulator.step()


def pcolored(text, color="green"):
    print(colored(text, color))