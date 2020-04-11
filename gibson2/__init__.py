import yaml
import os

__version__ = "0.0.4"

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'global_config.yaml')) as f:
    global_config = yaml.load(f, Loader=yaml.FullLoader)

assets_path = global_config['assets_path']
dataset_path = global_config['dataset_path']
root_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.isabs(assets_path):
    assets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), assets_path)
if not os.path.isabs(dataset_path):
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset_path)

print('using asset: path {} dataset path: {}'.format(assets_path, dataset_path))
