import logging
import os

import packaging.version
import yaml

__version__ = "2.2.2"

__logo__ = r"""
 _   _____  _  _
(_) / ____|(_)| |
 _ | |  __  _ | |__   ___   ___   _ __
| || | |_ || || '_ \ / __| / _ \ | '_ \
| || |__| || || |_) |\__ \| (_) || | | |
|_| \_____||_||_.__/ |___/ \___/ |_| |_|
"""

log = logging.getLogger(__name__)
_LOG_LEVEL = os.environ.get("IG_LOG_LEVEL", "INFO").upper()
log.setLevel(level=_LOG_LEVEL)

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

if log.isEnabledFor(logging.INFO):
    print(__logo__)

log.debug("Importing iGibson (igibson module)")
log.debug("Assets path: {}".format(assets_path))
log.debug("Gibson Dataset path: {}".format(g_dataset_path))
log.debug("iG Dataset path: {}".format(ig_dataset_path))
log.debug("3D-FRONT Dataset path: {}".format(threedfront_dataset_path))
log.debug("CubiCasa5K Dataset path: {}".format(cubicasa_dataset_path))
log.debug("iGibson Key path: {}".format(key_path))


def get_version(dataset_path):
    try:
        version_filename = os.path.join(dataset_path, "VERSION")
        with open(version_filename, "r") as version_file:
            return packaging.version.Version(version_file.read())
    except (IOError, packaging.version.InvalidVersion):
        raise ValueError("Could not read version file at %s - please update your assets and ig_dataset.", dataset_path)


_PARSED_VERSION = packaging.version.Version(__version__)

# Backwards compatible up to this version:
MIN_ASSETS_VERSION_INCL = "2.0.6"
# Compatible with releases for same major/minor/patch:
MAX_ASSETS_VERSION_EXCL = "%d.%d.%d" % (_PARSED_VERSION.major, _PARSED_VERSION.minor, _PARSED_VERSION.micro + 1)
if os.path.exists(assets_path):
    _assets_version = get_version(assets_path)
    assert (
        packaging.version.Version(MIN_ASSETS_VERSION_INCL)
        <= _assets_version
        < packaging.version.Version(MAX_ASSETS_VERSION_EXCL)
    ), "ig_assets version %s incompatible. Needs to be in range [%s, %s)" % (
        str(_assets_version),
        str(MIN_ASSETS_VERSION_INCL),
        str(MAX_ASSETS_VERSION_EXCL),
    )

# Backwards compatible up to this version:
MIN_DATASET_VERSION_INCL = "2.0.6"
# Compatible with releases for same major/minor/patch:
MAX_DATASET_VERSION_EXCL = "%d.%d.%d" % (_PARSED_VERSION.major, _PARSED_VERSION.minor, _PARSED_VERSION.micro + 1)
if os.path.exists(ig_dataset_path):
    _ig_dataset_version = get_version(ig_dataset_path)
    assert (
        packaging.version.Version(MIN_DATASET_VERSION_INCL)
        <= _ig_dataset_version
        < packaging.version.Version(MAX_DATASET_VERSION_EXCL)
    ), "ig_dataset version %s incompatible. Needs to be in range [%s, %s)" % (
        str(_ig_dataset_version),
        str(MIN_DATASET_VERSION_INCL),
        str(MAX_DATASET_VERSION_EXCL),
    )

examples_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples")
configs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

log.debug("Example path: {}".format(examples_path))
log.debug("Example config path: {}".format(configs_path))

# whether to enable debugging mode for object sampling
debug_sampling = False

# whether to ignore visual shape when importing to pybullet
ignore_visual_shape = True
