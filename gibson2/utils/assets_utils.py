import gibson2
import os
import argparse
import random
import subprocess


def get_ig_scene_path(scene_name):
    ig_dataset_path = gibson2.ig_dataset_path
    ig_scenes_path = ig_dataset_path + "/scenes"
    assert scene_name in os.listdir(ig_scenes_path), "Scene {} does not exist".format(scene_name)
    return os.path.join(ig_scenes_path, scene_name)

def get_ig_category_path(category_name):
    ig_dataset_path = gibson2.ig_dataset_path
    ig_categories_path = ig_dataset_path + "/objects"
    assert category_name in os.listdir(ig_categories_path), "Category {} does not exist".format(category_name)
    return os.path.join(ig_categories_path, category_name)  

def get_ig_model_path(category_name, model_name):
    ig_category_path = get_ig_category_path(category_name)
    assert model_name in os.listdir(ig_category_path), "Model {} from category {} does not exist".format(model_name, category_name)
    return os.path.join(ig_category_path, model_name)  

def get_ig_assets_version():
    process = subprocess.Popen(['git',  '-C', gibson2.ig_dataset_path, 'rev-parse', 'HEAD'],
                               shell=False, stdout=subprocess.PIPE)
    git_head_hash = str(process.communicate()[0].strip())
    return "{}".format(git_head_hash)

def get_scene_path(scene_id):
    data_path = gibson2.dataset_path
    assert scene_id in os.listdir(data_path) or scene_id == 'stadium', "Scene {} does not exist".format(scene_id)
    return os.path.join(data_path, scene_id)

def get_texture_file(mesh_file):
    model_dir = os.path.dirname(mesh_file)
    with open(mesh_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if 'mtllib' in line]
        if len(lines) == 0:
            return
        mtl_file = lines[0].split()[1]
        mtl_file = os.path.join(model_dir, mtl_file)

    with open(mtl_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if 'map_Kd' in line]
        if len(lines) == 0:
            return
        texture_file = lines[0].split()[1]
        texture_file = os.path.join(model_dir, texture_file)

    return texture_file

def download_assets():
    if not os.path.exists(gibson2.assets_path):
        os.system('wget https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz -O /tmp/assets_igibson.tar.gz')
        os.system('tar -zxf /tmp/assets_igibson.tar.gz --directory {}'.format(os.path.dirname(gibson2.assets_path)))


def download_demo_data():
    if not os.path.exists(gibson2.dataset_path):
        os.makedirs(gibson2.dataset_path)

    if not os.path.exists(os.path.join(gibson2.dataset_path, 'Rs')):
        os.system('wget https://storage.googleapis.com/gibson_scenes/Rs.tar.gz -O /tmp/Rs.tar.gz')
        os.system('tar -zxf /tmp/Rs.tar.gz --directory {}'.format(gibson2.dataset_path))

    # if not os.path.exists(os.path.join(gibson2.dataset_path, 'Rs_interactive')):
    #     os.system('wget https://storage.googleapis.com/gibson_scenes/Rs_interactive.tar.gz -O /tmp/Rs_interactive.tar.gz')
    #     os.system('tar -zxf /tmp/Rs_interactive.tar.gz --directory {}'.format(gibson2.dataset_path))

def download_dataset(url):
    if not os.path.exists(gibson2.dataset_path):
        os.makedirs(gibson2.dataset_path)

    file_name = url.split('/')[-1]
    os.system('wget {} -O /tmp/{}'.format(url, file_name))
    os.system('tar -zxf /tmp/{} --strip-components=1 --directory {}'.format(file_name, gibson2.dataset_path))
    # these datasets comes as folders, in these folder there are scenes, so --strip-components are needed. 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_assets', action='store_true', help='download assets file')
    parser.add_argument('--download_demo_data', action='store_true', help='download demo data Rs and Rs_interactive')
    parser.add_argument('--download_dataset', type=str, help='download dataset file given an URL')

    args = parser.parse_args()

    if args.download_assets:
        download_assets()
    elif args.download_demo_data:
        download_demo_data()
    elif args.download_dataset is not None:
        download_dataset(args.download_dataset)
