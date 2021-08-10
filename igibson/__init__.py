import logging
import os

import yaml

__version__ = "2.0.0"

logging.getLogger().setLevel(logging.INFO)

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "global_config.yaml")) as f:
    global_config = yaml.load(f, Loader=yaml.FullLoader)

# can override assets_path and dataset_path from environment variable
if "GIBSON_ASSETS_PATH" in os.environ:
    assets_path = os.environ["GIBSON_ASSETS_PATH"]
else:
    assets_path = global_config["assets_path"]
assets_path = os.path.expanduser(assets_path)

if "GIBSON_DATASET_PATH" in os.environ:
    g_dataset_path = os.environ["GIBSON_DATASET_PATH"]
else:
    g_dataset_path = global_config["g_dataset_path"]
g_dataset_path = os.path.expanduser(g_dataset_path)

if "IGIBSON_DATASET_PATH" in os.environ:
    ig_dataset_path = os.environ["IGIBSON_DATASET_PATH"]
else:
    ig_dataset_path = global_config["ig_dataset_path"]
ig_dataset_path = os.path.expanduser(ig_dataset_path)

if "3DFRONT_DATASET_PATH" in os.environ:
    threedfront_dataset_path = os.environ["3DFRONT_DATASET_PATH"]
else:
    threedfront_dataset_path = global_config["threedfront_dataset_path"]
threedfront_dataset_path = os.path.expanduser(threedfront_dataset_path)

if "CUBICASA_DATASET_PATH" in os.environ:
    cubicasa_dataset_path = os.environ["CUBICASA_DATASET_PATH"]
else:
    cubicasa_dataset_path = global_config["cubicasa_dataset_path"]
cubicasa_dataset_path = os.path.expanduser(cubicasa_dataset_path)

if "KEY_PATH" in os.environ:
    key_path = os.environ["KEY_PATH"]
else:
    key_path = global_config["key_path"]
key_path = os.path.expanduser(key_path)

root_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.isabs(assets_path):
    assets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), assets_path)
if not os.path.isabs(g_dataset_path):
    g_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), g_dataset_path)
if not os.path.isabs(ig_dataset_path):
    ig_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ig_dataset_path)
if not os.path.isabs(threedfront_dataset_path):
    threedfront_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), threedfront_dataset_path)
if not os.path.isabs(cubicasa_dataset_path):
    cubicasa_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cubicasa_dataset_path)
if not os.path.isabs(key_path):
    key_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), key_path)

logging.info("Importing iGibson (igibson module)")
logging.info("Assets path: {}".format(assets_path))
logging.info("Gibson Dataset path: {}".format(g_dataset_path))
logging.info("iG Dataset path: {}".format(ig_dataset_path))
logging.info("3D-FRONT Dataset path: {}".format(threedfront_dataset_path))
logging.info("CubiCasa5K Dataset path: {}".format(cubicasa_dataset_path))
logging.info("iGibson Key path: {}".format(key_path))

example_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples")
example_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples", "configs")

logging.info("Example path: {}".format(example_path))
logging.info("Example config path: {}".format(example_config_path))

debug_sampling = False
