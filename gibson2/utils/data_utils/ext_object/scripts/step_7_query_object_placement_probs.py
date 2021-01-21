from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
import math
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Configure which surfaces and containers in a scene an object might go in.')
    parser.add_argument('object_name', type=str)
    parser.add_argument('scene_name', type=str)
    args = parser.parse_args()

    support_objs_json = 'data/ig_dataset/scenes/%s/misc/all_support_objs.json'%args.scene_name
    scene_misc_dir = os.path.dirname(support_objs_json)
    if not os.path.isdir(scene_misc_dir):
        print('%s is not a valid scene name'%args.scene_name)
        return

    obj_json_path = 'data/ig_dataset/objects/%s/blender_fixed/misc/placement_probs.json'%(args.object_name)
    obj_misc_dir = os.path.dirname(obj_json_path)
    if not os.path.isdir(obj_misc_dir):
        print('%s is not a valid object name'%args.object_name)
        return

    if os.path.isfile(support_objs_json):
        with open(support_objs_json, 'r') as f:
            support_obj_dicts = json.load(f)
    else:
        settings = MeshRendererSettings(enable_shadow=False, msaa=False, enable_pbr=False)
        s = Simulator(mode='headless', image_width=800,
                      image_height=800, rendering_settings=settings)
        simulator = s
        scene = InteractiveIndoorScene(args.scene_name, texture_randomization=False, object_randomization=False)
        s.import_ig_scene(scene)

        support_obj_dicts = []
        for obj_name in scene.objects_by_name:
            obj = scene.objects_by_name[obj_name]
            if not obj.supporting_surfaces:
                continue
            info_dict = {}
            info_dict['name'] = obj_name
            info_dict['category'] = obj.category
            info_dict['room'] = obj.in_rooms[0]
            info_dict['supporting_surface_types'] = list(obj.supporting_surfaces.keys())
            support_obj_dicts.append(info_dict)

        with open(support_objs_json, 'w') as f:
            json.dump(support_obj_dicts, f)

        s.disconnect()

    print('Total number of supporting objects in scene: %d'%len(support_obj_dicts))

    support_obj_type_tuples = set()
    unique_categories = set()
    for support_obj_dict in support_obj_dicts:
        obj_category = support_obj_dict['category']
        unique_categories.add(obj_category)

    unique_categories = list(unique_categories)
    category_rooms = {category: set() for category in unique_categories}
    room_category_support_types = {}
    for support_obj_dict in support_obj_dicts:
        obj_category = support_obj_dict['category']
        obj_room = support_obj_dict['room'][:-2]
        category_rooms[obj_category].add(obj_room)
        room_category_support_types[(obj_category,obj_room)] = support_obj_dict['supporting_surface_types']

    for category in category_rooms:
        category_rooms[category] = list(category_rooms[category])

    obj_probs = {}
    done = False
    total_prob = 0.0
    while not done:
        for i,category in enumerate(unique_categories):
            print('Object option %d: %s'%(i,category))

        user_input = input('Input obj category number or name to assign probability, or done to finish\n')
        if user_input == 'done':
            break

        try:
            obj_num = int(user_input)
            obj_category = unique_categories[obj_num]
        except:
            obj_category = user_input
            if obj_category not in unique_categories:
                print('Input %s is not valid, try again'%obj_category)
                continue

        rooms = category_rooms[obj_category]
        if len(rooms) == 1:
            room_type = rooms[0]
        else:
            for i,room in enumerate(rooms):
                print('Room option %d: %s'%(i,room))
            room_choice = input('Input obj room number or name to assign probability\n')
            try:
                room_type = rooms[int(room_choice)]
            except:
                room_type = room_choice
                if room_type not in rooms:
                    print('%s is not a valid room, try again'%room_type)
                    continue

        support_types = room_category_support_types[(obj_category, room_type)]
        if len(support_types) == 1:
            support_type = support_types[0]
        else:
            for i,support_type in enumerate(support_types):
                print('Support type option %d: %s'%(i,support_type))
            type_choice = input('Input support type number or name to assign probability\n')
            try:
                support_type = support_types[int(type_choice)]
            except:
                support_type = type_choice
                if support_type not in support_types:
                    print('%s is not a valid support type, try again'%support_type)
                    continue

        prob = float(input('Enter probability for object %s being %s %s in room %s\n'%\
                           (args.object_name, support_type, obj_category, room_type)))
        obj_probs['%s-%s-%s'%(obj_category,room_type,support_type)] = prob
        total_prob+=prob

    for key in obj_probs:
        obj_probs[key]/=total_prob

    with open(obj_json_path, 'w') as f:
        json.dump(obj_probs, f)

if __name__ == '__main__':
    main()
