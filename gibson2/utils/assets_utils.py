import gibson2
import os
def download_data():
    if not os.path.exists(gibson2.assets_path):
        os.system('wget https://storage.googleapis.com/gibsonassets/assets_gibson_v2.tar.gz -O /tmp/assets_gibson_v2.tar.gz')
        os.system('tar -zxf /tmp/assets_gibson_v2.tar.gz --directory {}'.format(os.path.dirname(gibson2.assets_path)))

    if not os.path.exists(gibson2.dataset_path):
        os.makedirs(gibson2.dataset_path)
    if not os.path.exists(os.path.join(gibson2.dataset_path, 'Rs')):
        os.system('wget https://storage.googleapis.com/gibson_scenes/Rs.tar.gz -O /tmp/Rs.tar.gz')
        os.system('tar -zxf /tmp/Rs.tar.gz --directory {}'.format(gibson2.dataset_path))
    if not os.path.exists(os.path.join(gibson2.assets_path, 'turtlebot_p2p_nav_house.yaml')):
        os.system('wget https://storage.googleapis.com/gibson_scenes/turtlebot_p2p_nav_house.yaml \
        -O {}/turtlebot_p2p_nav_house.yaml'.format(gibson2.assets_path))
