import logging
import os

import semver
import yaml

__version__ = "2.0.6"

__logo__ = """
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
            return semver.VersionInfo.parse(version_file.read())
    except (IOError, ValueError):
        raise ValueError("Could not read version file at %s - please update your assets and ig_dataset.", dataset_path)


# Assert that the assets and dataset versions are both sub-versions of the iGibson version.
_parsed_version = semver.VersionInfo.parse(__version__)

if os.path.exists(assets_path):
    _assets_version = get_version(assets_path)
    assert (
        # The version numbers should be same at the major/minor/patch level but can differ at the 4th level.
        _parsed_version
        <= _assets_version
        < _parsed_version.bump_patch()
    ), "ig_assets version %s does not match iGibson version %s (need %s.*)" % (
        str(_assets_version),
        str(_parsed_version),
        str(_assets_version),
    )

if os.path.exists(ig_dataset_path):
    _ig_dataset_version = get_version(ig_dataset_path)
    assert (
        # The version numbers should be same at the major/minor/patch level but can differ at the 4th level.
        _parsed_version
        <= _ig_dataset_version
        < _parsed_version.bump_patch()
    ), "ig_dataset version %s does not match iGibson version %s (need %s.*)" % (
        str(_ig_dataset_version),
        str(_parsed_version),
        str(_ig_dataset_version),
    )

example_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples")
example_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

log.debug("Example path: {}".format(example_path))
log.debug("Example config path: {}".format(example_config_path))

# whether to enable debugging mode for object sampling
debug_sampling = False

# whether to ignore visual shape when importing to pybullet
ignore_visual_shape = True
