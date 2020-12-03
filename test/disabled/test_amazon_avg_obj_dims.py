import os
import json
import numpy as np
from IPython import embed
from mws import mws
import urllib.request
import time
import sys


# More docuemntation can be found in here: https://docs.google.com/document/d/1sTkbLHr-tNhkGnGwpl1SWxjzFHy87PLzu0FmwPXlqKQ/edit?usp=sharing
# The handpicked ASINs and the object dims for NOT_IN_AMAZON can be found here: https://docs.google.com/spreadsheets/d/1D5LJDqY0hzyCuMSSZS34nP0H7Vi6Kq2RxQWon2Sg_mg/edit?usp=sharing

# These object classes DO exist Amazon and we can retrieve a list of object
# instances that are close to our object models through a direct keyword search
DIRECT_QUERY = [
    'basket',
    'bathtub',
    'bench',
    'bottom_cabinet',
    'bottom_cabinet_no_top',
    'carpet',
    'chair',
    'chest',
    'coffee_machine',
    'coffee_table',
    'console_table',
    'cooktop',
    'crib',
    'cushion',
    'dishwasher',
    'dryer',
    'fence',
    'floor_lamp',
    'fridge',
    'grandfather_clock',
    'guitar',
    'heater',
    'laptop',
    'loudspeaker',
    'microwave',
    'mirror',
    'monitor',
    'office_chair',
    'oven',
    'picture',
    'plant',
    'pool_table',
    'range_hood',
    'shelf',
    'shower',
    'sink',
    'sofa',
    'sofa_chair',
    'speaker_system',
    'standing_tv',
    'stool',
    'stove',
    'table',
    'table_lamp',
    'toilet',
    'top_cabinet',
    'towel_rack',
    'trash_can',
    'treadmill',
    'wall_clock',
    'wall_mounted_tv',
    'washer',
]

DIRECT_QUERY_SUB = {
    'bathtub': 'freestanding bathtub',
    'bottom cabinet': 'base cabinet',
    'bottom cabinet_no_top': 'base cabinet',
    'carpet': 'rug',
    'cushion': 'pillow',
    'dryer': 'front load dryer',
    'fridge': 'top freezer fridge',
    'heater': 'wall heater',
    'mirror': 'wall mirror',
    'oven': 'wall oven',
    'picture': 'canvas art',
    'range hood': 'wall mount range hood',
    'shelf': 'book shelf',
    'shower': 'corner steam shower',
    'sink': 'bathroom sink',
    'sofa chair': 'arm chair',
    'speaker system': 'home speaker system',
    'standing tv': 'tv',
    'stove': 'gas range',
    'table': 'dining table',
    'top cabinet': 'wall mounted cabinet',
    'wall_mounted tv': 'tv',
    'washer': 'front load washing machine',
}

# These object classes DO NOT exist on Amazon. Need to find info manually.
NOT_IN_AMAZON = [
    'bed',
    'counter',
    'door',
    'piano',
    'window',
]


def get_product_api():
    access_key = 'AKIAJHPQJIW6QR5LWITQ'
    seller_id = 'A1Q8GZYLKNKOGP'
    secret_key = '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'
    marketplace_usa = 'ATVPDKIKX0DER'
    products_api = mws.Products(access_key, secret_key, seller_id, region='US')
    return products_api, marketplace_usa


def query_amazon(products_api, marketplace_usa):
    root_dir = '/cvgl2/u/chengshu/ig_dataset'
    obj_dir = os.path.join(root_dir, 'objects')
    obj_dim_dir = os.path.join(root_dir, 'object_dims')

    for obj_class in sorted(os.listdir(obj_dir)):
        obj_class_query = obj_class.replace('_', ' ')
        if obj_class not in ['top_cabinet', 'bottom_cabinet']:
            continue
        if obj_class not in DIRECT_QUERY:
            continue
        if obj_class_query in DIRECT_QUERY_SUB:
            obj_class_query = DIRECT_QUERY_SUB[obj_class_query]
        obj_class_dir = os.path.join(obj_dim_dir, obj_class)
        os.makedirs(obj_class_dir, exist_ok=True)
        products = products_api.list_matching_products(
            marketplaceid=marketplace_usa, query=obj_class_query)
        assert products.response.status_code == 200, 'API failed'

        valid_item = 0
        dims = []
        for product in products.parsed.Products.Product:
            # no valid dimension, skip
            if 'ItemDimensions' not in product.AttributeSets.ItemAttributes:
                continue

            item_dim = product.AttributeSets.ItemAttributes.ItemDimensions
            item_dim_keys = item_dim.keys()
            has_valid_dim = 'Width' in item_dim_keys and 'Length' in item_dim_keys \
                and 'Height' in item_dim_keys and 'Weight' in item_dim_keys
            # no valid dimension, skip
            if not has_valid_dim:
                continue

            # no thumbnail image, skip
            if 'SmallImage' not in product.AttributeSets.ItemAttributes:
                continue

            ASIN = product.Identifiers.MarketplaceASIN.ASIN

            item_dim_array = [
                item_dim.Width.value,
                item_dim.Length.value,
                item_dim.Height.value,
                item_dim.Weight.value,
            ]
            item_dim_array = [float(elem) for elem in item_dim_array]
            if obj_class in [
                'bathtub', 'bench', 'chest', 'coffee_table',
                'console_table', 'cooktop', 'crib', 'cushion', 'fence',
                'heater', 'laptop', 'microwave', 'monitor',
                'pool_table', 'range_hood', 'shelf', 'shower', 'sofa',
                'table', 'towel_rack', 'standing_tv', 'wall_mounted_tv',
            ]:
                short_side = min(item_dim_array[:2])
                long_side = max(item_dim_array[:2])
                item_dim_array[0] = long_side
                item_dim_array[1] = short_side
            elif obj_class in ['table_lamp']:
                short_side = min(item_dim_array[:2])
                long_side = max(item_dim_array[:2])
                item_dim_array[0] = short_side
                item_dim_array[1] = long_side
            else:
                short_side, long_side, longest_side = sorted(
                    item_dim_array[:3])
                if obj_class in ['guitar', 'mirror', 'picture', 'wall_clock']:
                    item_dim_array[0] = long_side
                    item_dim_array[1] = short_side
                    item_dim_array[2] = longest_side
                elif obj_class in ['carpet']:
                    item_dim_array[0] = longest_side
                    item_dim_array[1] = long_side
                    item_dim_array[2] = short_side

            img_url = product.AttributeSets.ItemAttributes.SmallImage.URL

            dim_txt = os.path.join(obj_class_dir, ASIN + '.txt')
            with open(dim_txt, 'w+') as f:
                f.write(' '.join([str(elem) for elem in item_dim_array]))
            img_path = os.path.join(obj_class_dir, ASIN + '.jpg')
            urllib.request.urlretrieve(img_url, img_path)
            valid_item += 1
            dims.append(item_dim_array)

        dims = np.array(dims)
        # inch to cm
        dims[:, :3] *= 2.54
        # pound to kg
        dims[:, 3] *= 0.453592
        print(f"{obj_class} {obj_class_query}: {valid_item} valid items.")
        print('avg W x L x H (in cm), weight (in kg): {} {} {} {}'.format(
            np.median(dims[:, 0]),
            np.median(dims[:, 1]),
            np.median(dims[:, 2]),
            np.median(dims[:, 3])
        ))
        # Request quota is once every 5 seconds.
        time.sleep(6.0)


def check_weight():
    root_dir = '/cvgl2/u/chengshu/ig_dataset'
    obj_dim_dir = os.path.join(root_dir, 'object_dims')
    non_amazon_csv = os.path.join(obj_dim_dir, 'non_amazon.csv')
    assert os.path.isfile(non_amazon_csv), \
        f'please download non-amazon dimensions and put it in this path: {non_amazon_csv}'

    obj_dim_dict = {}
    for obj_class in sorted(os.listdir(obj_dim_dir)):
        obj_dir = os.path.join(obj_dim_dir, obj_class)
        if not os.path.isdir(obj_dir):
            continue
        dims = []
        for txt_file in os.listdir(obj_dir):
            if not txt_file.endswith('.txt'):
                continue
            txt_file = os.path.join(obj_dir, txt_file)
            with open(txt_file) as f:
                dims.append([float(item)
                             for item in f.readline().strip().split()])
        dims = np.array(dims)
        # inch to cm
        dims[:, :3] *= 2.54
        # pound to kg
        dims[:, 3] *= 0.453592
        print(f"{obj_class}: {len(dims)} valid items.")
        print('avg W x L x H (in cm), weight (in kg): {} {} {} {}'.format(
            np.median(dims[:, 0]),
            np.median(dims[:, 1]),
            np.median(dims[:, 2]),
            np.median(dims[:, 3])
        ))
        obj_dim_dict[obj_class] = {
            'size': [np.median(dims[:, 0] * 0.01),
                     np.median(dims[:, 1] * 0.01),
                     np.median(dims[:, 2] * 0.01)],
            'mass': np.median(dims[:, 3])
        }

    with open(non_amazon_csv) as f:
        # skip first row
        f.readline()
        for line in f.readlines():
            line = line.strip().split(',')
            obj_class, width, length, height, weight = line
            width, length, height, weight = \
                float(width) * 2.54, float(length) * 2.54, \
                float(height) * 2.54, float(weight) * 0.453592
            print(f"{obj_class}:")
            print('avg W x L x H (in cm), weight (in kg): {} {} {} {}'.format(
                width, length, height, weight
            ))
            obj_dim_dict[obj_class] = {
                'size': [width * 0.01,
                         length * 0.01,
                         height * 0.01],
                'mass': weight
            }

    for obj_class in obj_dim_dict:
        volume = np.prod(obj_dim_dict[obj_class]['size'])
        obj_dim_dict[obj_class]['density'] = obj_dim_dict[obj_class]['mass'] / volume

    avg_obj_dims = os.path.join(obj_dim_dir, 'avg_obj_dims.json')
    with open(avg_obj_dims, 'w+') as f:
        json.dump(obj_dim_dict, f)


# For testing purposes
def query_single_product(products_api, marketplace_usa, obj_class_query):
    products = products_api.list_matching_products(
        marketplaceid=marketplace_usa, query=obj_class_query)
    assert products.response.status_code == 200, 'API failed'

    dims = []
    for product in products.parsed.Products.Product:
        # no valid dimension, skip
        if 'ItemDimensions' not in product.AttributeSets.ItemAttributes:
            continue

        item_dim = product.AttributeSets.ItemAttributes.ItemDimensions
        item_dim_keys = item_dim.keys()
        has_valid_dim = 'Width' in item_dim_keys and 'Length' in item_dim_keys \
            and 'Height' in item_dim_keys and 'Weight' in item_dim_keys
        # no valid dimension, skip
        if not has_valid_dim:
            continue

        # no thumbnail image, skip
        if 'SmallImage' not in product.AttributeSets.ItemAttributes:
            continue

        item_dim_array = [
            item_dim.Width.value,
            item_dim.Length.value,
            item_dim.Height.value,
            item_dim.Weight.value,
        ]
        dims.append([float(item) for item in item_dim_array])
        ASIN = product.Identifiers.MarketplaceASIN.ASIN
        print(ASIN)

    dims = np.array(dims)
    # inch to cm
    dims[:, :3] *= 2.54
    # pound to kg
    dims[:, 3] *= 0.453592
    print(dims)
    print('{} avg W x L x H (in cm), weight (in kg): {} {} {} {}'.format(
        obj_class_query,
        np.median(dims[:, 0]),
        np.median(dims[:, 1]),
        np.median(dims[:, 2]),
        np.median(dims[:, 3])
    ))


def main():
    products_api, marketplace_usa = get_product_api()
    # query_single_product(products_api, marketplace_usa,
    #                      'base cabinet')
    query_amazon(products_api, marketplace_usa)
    # manually download csv file from non-Amazon products from
    # https://docs.google.com/spreadsheets/d/1D5LJDqY0hzyCuMSSZS34nP0H7Vi6Kq2RxQWon2Sg_mg/edit#gid=0
    check_weight()


if __name__ == '__main__':
    main()
