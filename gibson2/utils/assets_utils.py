import gibson2
import os

def get_model_path(model_id):
    data_path = gibson2.dataset_path
    assert model_id in os.listdir(data_path) or model_id == 'stadium', "Model {} does not exist".format(model_id)
    return os.path.join(data_path, model_id)

def download_data():
    if not os.path.exists(gibson2.assets_path):
        os.system('wget https://storage.googleapis.com/gibsonassets/assets_igibson.tar.gz -O /tmp/assets_igibson.tar.gz')
        os.system('tar -zxf /tmp/assets_igibson.tar.gz --directory {}'.format(os.path.dirname(gibson2.assets_path)))

    if not os.path.exists(gibson2.dataset_path):
        os.makedirs(gibson2.dataset_path)

    if not os.path.exists(os.path.join(gibson2.dataset_path, 'Rs')):
        os.system('wget https://storage.googleapis.com/gibson_scenes/Rs.tar.gz -O /tmp/Rs.tar.gz')
        os.system('tar -zxf /tmp/Rs.tar.gz --directory {}'.format(gibson2.dataset_path))

if __name__ == "__main__":
    download_data()
