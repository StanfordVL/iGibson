import json
import os
import math
import numpy as np

def get_binary_label(image_path):
    img_dir = os.path.dirname(image_path)
    img_fname = os.path.splitext(os.path.basename(image_path))[0]
    img_id = img_fname.split('_')[1]
    json_path = os.path.join(img_dir, 'info_{}.json'.format(img_id))
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    if data['hit'] is None:
        return False
    joint_pre = data['interaction_pre']['joint']
    joint_post = data['interaction_post']['joint']

    link_pre = data['interaction_pre']['link']
    link_post = data['interaction_post']['link']
    
    link_pos_delta = np.linalg.norm(np.array(link_pre[0]) 
                                  - np.array(link_post[0]))
    if link_pos_delta > 0.2:
        return True

    if joint_pre is not None:
        if joint_pre['type'] == 4:
            return False
        if joint_pre['type'] == 0:
            return math.abs(joint_post['pos'] - joint_pre['pos']) > 0.1
        if joint_pre['type'] == 1:
            return math.abs(joint_post['pos'] - joint_pre['pos']) > 0.1

    return link_pos_delta > 0.1
            

