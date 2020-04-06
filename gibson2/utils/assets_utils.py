import gibson2
import os
import argparse

def get_model_path(model_id):
    data_path = gibson2.dataset_path
    assert model_id in os.listdir(data_path) or model_id == 'stadium', "Model {} does not exist".format(model_id)
    return os.path.join(data_path, model_id)

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

    if not os.path.exists(os.path.join(gibson2.dataset_path, 'Rs_interactive')):
        os.system('wget https://storage.googleapis.com/gibson_scenes/Rs_interactive.tar.gz -O /tmp/Rs_interactive.tar.gz')
        os.system('tar -zxf /tmp/Rs_interactive.tar.gz --directory {}'.format(gibson2.dataset_path))

def download_dataset(url):
    if not os.path.exists(gibson2.dataset_path):
        os.makedirs(gibson2.dataset_path)

    file_name = url.split('/')[-1]
    os.system('wget {} -O /tmp/{}'.format(url, file_name))
    os.system('tar -zxf /tmp/{} --directory {}'.format(file_name, gibson2.dataset_path))
    os.system('mv {}/{}/* {}'.format(gibson2.dataset_path, os.path.splitext(file_name)[0], gibson2.dataset_path))

if __name__ == "__main__":
    #download_data()
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
