import yaml
import os


with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'global_config.yaml')) as f:
    global_config = yaml.load(f)

assets_path = global_config['assets_path']
if not os.path.isabs(assets_path):
    assets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               assets_path)

print('using asset path {}'.format(assets_path))
