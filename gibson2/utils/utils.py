import numpy as np
import tensorflow as tf  # pylint: ignore-module
import builtins
import functools
import copy
import os
import collections
import yaml

def make_gpu_session(num_gpu=1):
    if num_gpu == 1:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.Session()
    return sess


def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f)
    return config_data
