import bpy
import os
import sys
import glob
import xml.etree.ElementTree as ET
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from material_util import clean_nodes,build_pbr_textured_nodes_from_name,create_empty_image
from math import *
from mathutils import *
from collections import defaultdict
import json
from os import close, dup, O_WRONLY


# https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def look_at(obj_camera, point):
    if not isinstance(point, Vector):
        point = Vector(point)
    loc_camera = obj_camera.location
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

class redirect_output(): 
    def __init__(self): 
        logfile = '/tmp/blender_command.log'
        with open(logfile, 'a') as f:
            f.close()
        self.old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
          
    def __enter__(self): 
        return self
      
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        os.close(1)
        os.dup(self.old)
        os.close(self.old)

def setup_cycles(samples=2048):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.ao_bounces = 0
    bpy.context.scene.cycles.ao_bounces_render = 0
    bpy.context.scene.cycles.diffuse_bounces = 3
    bpy.context.scene.cycles.glossy_bounces = 4
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.min_light_bounces = 0
    bpy.context.scene.cycles.min_transparent_bounces = 0
    bpy.context.scene.cycles.transmission_bounces = 4
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.volume_bounces = 0
    bpy.context.scene.render.tile_x = 200
    bpy.context.scene.render.tile_y = 200
    bpy.context.scene.cycles.samples = samples 

def setup_resolution(x=1920,y=1080):
    bpy.context.scene.render.resolution_x = x
    bpy.context.scene.render.resolution_y = y

def bake_model(mat_dir, channels, 
               objects=None, overwrite=False,
               set_default_samples=True):

    if objects is None:
        for on in bpy.context.scene.objects.keys():
            obj = bpy.context.scene.objects[on]
            obj.select_set(True)
        bpy.ops.object.select_all(action='SELECT')
    else:
        for obj in objects:
            obj.select_set(True)

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    if set_default_samples:
        setup_cycles()

    c = 'COMBINED'
    if c in channels:
        print('baking combined...')
        if overwrite or not os.path.isfile('{}/{}.png'.format(mat_dir, c)):
        # if not os.path.isfile('{}/{}.png'.format(mat_dir, c)):
            if c in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[c])
            res,margin = channels[c]
            for mat in bpy.data.materials:
                node = create_empty_image(mat.node_tree, c, False, dim=(res,res))
                node.select = True
                mat.node_tree.nodes.active = node
            with redirect_output():
                bpy.ops.object.bake(type=c, 
                                pass_filter={'AO', 'EMIT', 'DIRECT', 
                                             'INDIRECT', 'COLOR', 'DIFFUSE', 
                                             'GLOSSY', 'TRANSMISSION'}, 
                                margin=margin) 
            bpy.data.images[c].filepath_raw = '{}/{}.png'.format(mat_dir, c)
            bpy.data.images[c].file_format = 'PNG'
            bpy.data.images[c].save()

    c = 'ROUGHNESS'
    if c in channels:
        print('baking roughness...')
        if overwrite or not os.path.isfile('{}/{}.png'.format(mat_dir, c)):
        # if not os.path.isfile('{}/{}.png'.format(mat_dir, c)):
            if c in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[c])
            res,margin = channels[c]
            for mat in bpy.data.materials:
                node = create_empty_image(mat.node_tree, c, False, dim=(res,res))
                node.select = True
                mat.node_tree.nodes.active = node
            with redirect_output():
                bpy.ops.object.bake(type=c, margin=margin) 
            bpy.data.images[c].filepath_raw = '{}/{}.png'.format(mat_dir, c)
            bpy.data.images[c].file_format = 'PNG'
            bpy.data.images[c].save()

    #######################################
    # Re-wire metallic for baking
    #######################################
    c = 'METALLIC'
    if c in channels:
        print('baking metallic...')
        if overwrite or not os.path.isfile('{}/{}.png'.format(mat_dir, c)):
            if c in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[c])
            res,margin = channels[c]
            for mat in bpy.data.materials:
                principle_bsdf = mat.node_tree.nodes['Principled BSDF']
                metallic_port = principle_bsdf.inputs['Metallic']
                if len(metallic_port.links) == 0:
                    metallic_node = None
                    metallic_val = metallic_port.default_value
                else:
                    metallic_node= metallic_port.links[0].from_node
                    output = metallic_node.outputs['Color']
                    for l in output.links :
                        mat.node_tree.links.remove(l)
                roughness_port = principle_bsdf.inputs['Roughness']
                if len(roughness_port.links) == 0:
                    roughness_node = None
                    roughness_val = roughness_port.default_value
                else:
                    roughness_node = roughness_port.links[0].from_node
                    output = roughness_node.outputs['Color']
                    for l in output.links :
                        mat.node_tree.links.remove(l)
                if metallic_node is None:
                    roughness_port.default_value = metallic_val
                else:
                    mat.node_tree.links.new(metallic_node.outputs['Color'], roughness_port)

            for mat in bpy.data.materials:
                node = create_empty_image(mat.node_tree, c, False, dim=(res,res))
                node.select = True
                mat.node_tree.nodes.active = node
            with redirect_output():
                bpy.ops.object.bake(type='ROUGHNESS', margin=margin) 
            bpy.data.images[c].filepath_raw = '{}/{}.png'.format(mat_dir, c)
            bpy.data.images[c].file_format = 'PNG'
            bpy.data.images[c].save()

    #######################################
    # Dis-connect Metallic for baking
    #######################################
    if 'DIFFUSE' in channels or 'NORMAL' in channels:
        for mat in bpy.data.materials:
            if 'Principled BSDF' not in mat.node_tree.nodes:
                continue
            principle_bsdf = mat.node_tree.nodes['Principled BSDF']
            metallic_port = principle_bsdf.inputs['Metallic']
            if len(metallic_port.links) != 0:
                metallic_node= metallic_port.links[0].from_node
                output = metallic_node.outputs['Color']
                for l in output.links :
                    mat.node_tree.links.remove(l)
            metallic_port.default_value = 0

            roughness_port = principle_bsdf.inputs['Roughness']
            if len(roughness_port.links) != 0:
                roughness_node = roughness_port.links[0].from_node
                output = roughness_node.outputs['Color']
                for l in output.links :
                    mat.node_tree.links.remove(l)
            roughness_port.default_value = 0.5


    c = 'DIFFUSE'
    if c in channels:
        print('baking diffuse...')
        if overwrite or not os.path.isfile('{}/{}.png'.format(mat_dir, c)):
            if c in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[c])
            res,margin = channels[c]
            for mat in bpy.data.materials:
                node = create_empty_image(mat.node_tree, c, True, dim=(res,res))
                node.select = True
                mat.node_tree.nodes.active = node
            with redirect_output():
                bpy.ops.object.bake(type=c, pass_filter={'COLOR'}, 
                                margin=margin)
            bpy.data.images[c].filepath_raw = '{}/{}.png'.format(mat_dir, c)
            bpy.data.images[c].file_format = 'PNG'
            bpy.data.images[c].save()

    #######################################
    # Re-wire normal for baking
    #######################################
    c = 'NORMAL'
    if c in channels:
        print('baking normal...')
        if overwrite or not os.path.isfile('{}/{}.png'.format(mat_dir, c)):
            if c in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[c])
            res,margin = channels[c]
            for mat in bpy.data.materials:
                if 'Principled BSDF' not in mat.node_tree.nodes:
                    continue
                principle_bsdf = mat.node_tree.nodes['Principled BSDF']
                diffuse_port = principle_bsdf.inputs['Base Color']
                for l in diffuse_port.links :
                    mat.node_tree.links.remove(l)
                normal_port = principle_bsdf.inputs['Normal']
                if len(normal_port.links) == 0:
                    normal_node= None
                    normal_val = (0.212, 0.212, 1., 1.)
                else:
                    normal_node = normal_port.links[0].from_node.inputs['Color'].links[0].from_node
                    normal_node.image.colorspace_settings.name = 'sRGB'
                    output = normal_node.outputs['Color']
                    for l in output.links :
                        mat.node_tree.links.remove(l)
                if normal_node is None:
                    diffuse_port.default_value = normal_val 
                else:
                    mat.node_tree.links.new(normal_node.outputs['Color'], diffuse_port)
            for mat in bpy.data.materials:
                node = create_empty_image(mat.node_tree, c, True, 
                                          dim=(res,res))
                node.select = True
                mat.node_tree.nodes.active = node
            with redirect_output():
                bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, 
                                margin=margin)
            bpy.data.images[c].filepath_raw = '{}/{}.png'.format(mat_dir, c)
            bpy.data.images[c].file_format = 'PNG'
            bpy.data.images[c].save()


def import_ig_scene_structure(scene_dir, import_mat=False):
    obj_dir = os.path.join(scene_dir, 
                                'shape', 'visual')
    for f in os.listdir(obj_dir):
        if not f.endswith('.obj'):
            continue
        element_name = '_'.join(os.path.splitext(f)[0].split('_')[:-1])
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_scene.obj(
            filepath = os.path.join(obj_dir, f),
            axis_up='Z', axis_forward='X',
            use_edges = True,
            use_smooth_groups = True, 
            use_split_objects = False,
            use_split_groups = False,
            use_groups_as_vgroups = False,
            use_image_search = False,
            split_mode = 'OFF')
        imported = bpy.context.selected_objects[:]
        for obj in imported:
            if obj.hide_get():
                obj.hide_set(False)
            ms = obj.data.materials
            for _ in range(len(ms)):
                ms.pop()
        if import_mat:
            mat = build_pbr_textured_nodes_from_name(element_name, 
                        get_ig_scene_texture_paths(scene_dir, element_name),
                        is_cc0=False)
            for obj in imported:
                obj.data.materials.append(mat)
        clean_unused()


def import_obj_folder(object_name, obj_dir,up='Z',forward='X'):
    scene_collection = bpy.context.view_layer.layer_collection
    bpy.context.view_layer.active_layer_collection = scene_collection
    bpy.ops.object.select_all(action='DESELECT')
    collection = bpy.data.collections.new(object_name) 
    bpy.context.scene.collection.children.link(collection)
    with redirect_output():
        for o in os.listdir(obj_dir):
            if os.path.splitext(o)[-1] != '.obj':
                continue
            bpy.ops.import_scene.obj(
                                filepath = os.path.join(obj_dir, o),
                                axis_up = up, 
                                axis_forward = forward,
                                use_edges = True,
                                use_smooth_groups = True, 
                                use_split_objects = False,
                                use_split_groups = False,
                                use_groups_as_vgroups = False,
                                use_image_search = False,
                                split_mode = 'OFF')
            imported = bpy.context.selected_objects[:]
            for obj in imported:
                collection.objects.link(obj)
                bpy.context.scene.collection.objects.unlink(obj)
    return collection

def import_ig_object(object_root, 
                     import_mat=False, 
                     scale=(1.,1.,1.),
                     loc=(0.,0.,0.), orn=(0.,0.,0.)):
    scene_collection = bpy.context.view_layer.layer_collection
    bpy.context.view_layer.active_layer_collection = scene_collection
    object_root = os.path.normpath(object_root)
    object_name = '_'.join(object_root.split('_')[-2:])
    obj_dir = os.path.join(object_root, 'shape', 'visual')
    collection = import_obj_folder(object_name, obj_dir)
    for obj in collection.objects:
        obj.scale = scale
        obj.rotation_euler = orn
        obj.location = loc
        if obj.hide_get():
            obj.hide_set(False)
        ms = obj.data.materials
        for _ in range(len(ms)):
            ms.pop()
        # obj.data.materials.append(mat)
    clean_unused()
    if not import_mat:
        return
    mat = build_pbr_textured_nodes_from_name('obj_mat', 
                get_ig_texture_paths(object_root),
                is_cc0=False)
    # for obj in bpy.context.selected_objects[:]:
    for obj in collection.objects:
        # obj = bpy.context.scene.objects[on]
        obj.data.materials.append(mat)
    bpy.ops.object.select_all(action='DESELECT')

def export_obj_folder(save_dir, skip_empty=True, save_material=False,
                      up='Z', forward='X'):
    with redirect_output():
        for on in bpy.context.scene.objects.keys():
            obj = bpy.context.scene.objects[on]
            if obj.type != 'MESH':
                continue
            if obj.hide_get():
                obj.hide_set(False)
            if skip_empty:
                me = obj.data
                v,e,f = len(me.vertices),len(me.edges),len(me.polygons)
                if v == 0 or e == 0 or f ==0:
                    continue
            bpy.context.view_layer.objects.active = obj 
            obj.select_set(True)
            save_path = os.path.join(save_dir, "{}.obj".format(on))
            bpy.ops.export_scene.obj(filepath=save_path, 
                                     use_selection=True, 
                                     axis_up=up, axis_forward=forward, 
                                     use_materials=save_material,
                                     use_triangles=True,
                                     path_mode="COPY")
            obj.select_set(False)

def export_ig_object(model_root, 
                     save_dir='shape/visual', 
                     skip_empty=False,
                     save_material=False):
    obj_dir= os.path.join(model_root, save_dir)
    os.makedirs(obj_dir, exist_ok=True)
    export_obj_folder(obj_dir, skip_empty=skip_empty, 
                      save_material=save_material)

def get_ig_scene_texture_paths(scene_root, element_name):
    def get_file_path(type):
        files = glob.glob(glob.escape(scene_root) + 
                          "/material/{}/*_{}.*".format(element_name, type))
        return files[0] if files else ""

    texture_paths = {}
    texture_paths["color"] = get_file_path("col")
    texture_paths["metallic"] = get_file_path("met")
    texture_paths["roughness"] = get_file_path("rgh")
    texture_paths["normal"] = get_file_path("nrm")
    texture_paths["displacement"] = get_file_path("disp")
    texture_paths["ambient_occlusion"] = get_file_path("AO")
    return texture_paths

def get_ig_texture_paths(object_root):
    def get_file_path(type):
        return os.path.join(object_root, 'material', '{}.png'.format(type))
    texture_paths = {}
    texture_paths["color"] = get_file_path("DIFFUSE")
    texture_paths["metallic"] = get_file_path("METALLIC")
    texture_paths["roughness"] = get_file_path("ROUGHNESS")
    texture_paths["normal"] = get_file_path("NORMAL")
    texture_paths["displacement"] = ""
    texture_paths["ambient_occlusion"] = ""
    return texture_paths


def clean_unused():
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def generate_light(location, energy, name):
    bpy.ops.mesh.primitive_torus_add()
    shape = bpy.context.active_object
    shape.location = location
    shape.scale = (0.1,0.1,0.03)
    new_material = bpy.data.materials.new("rim_light")
    new_material.use_nodes = True
    clean_nodes(new_material.node_tree.nodes)
    node_tree = new_material.node_tree
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfTranslucent')
    principled_node.name = 'BSDF'
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    principled_node.inputs['Color'].default_value = (1.0,1.0,1.0,1.0)
    ms = shape.data.materials
    for _ in range(len(ms)):
        ms.pop()
    shape.data.materials.append(new_material)

    # light_data = bpy.data.lights.new(name="Sample", type='AREA')
    type = 'POINT'
    light_data = bpy.data.lights.new(name="Sample", type=type)
    light_data.energy = energy 
    if type == 'POINT':
        light_data.shadow_soft_size = 0.15 
    elif type == 'AREA':
        light_data.size = 0.2 
        light_data.shape = 'DISK'

    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    light_object.location = location

    dg = bpy.context.evaluated_depsgraph_get() 
    dg.update()

def set_up_pano_cam():
    setup_cycles(samples=512)

    # create the camera object
    cam = bpy.data.cameras.new("Pano Cam")
    cam_obj = bpy.data.objects.new("Pano Cam", cam)
    cam_obj.data.type = 'PANO'
    cam_obj.data.cycles.panorama_type = 'EQUIRECTANGULAR'
    cam_z = 1.5 
    cam_obj.location = (0.,0.,cam_z)
    cam_rot = (pi / 2.,0.,0.)
    cam_obj.rotation_euler = cam_rot

    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    bpy.context.scene.render.resolution_x = 2048
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.resolution_percentage = 100 
    bpy.context.scene.render.image_settings.file_format = 'HDR'
    return cam_obj

def render_light_probe(cam, xy_loc, save_to):
    # pngpath = "{}.png".format(os.path.splitext(blend_path)[0])
    x,y = xy_loc 
    z = cam.location[-1]
    cam.location = (x,y,z) 
    bpy.data.scenes['Scene'].render.filepath = save_to
    with redirect_output():
        bpy.ops.render.render( write_still=True)

def set_up_birds_eye_ortho_cam(scene_length):
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.data.scenes['Scene'].display.shading.light = 'FLAT'
    bpy.data.scenes['Scene'].display.render_aa="OFF"

    # create the camera object
    cam1 = bpy.data.cameras.new("Bird-Eye Cam")
    cam1.lens = 16
    cam_obj1 = bpy.data.objects.new("Bird-Eye Cam", cam1)
    cam_obj1.data.type = 'ORTHO'
    cam_z = 10.
    cam_obj1.location = (0.,0.,cam_z)
    cam_rot = (0.,0.,0.)
    cam_obj1.rotation_euler = cam_rot

    bpy.context.scene.collection.objects.link(cam_obj1)
    bpy.context.scene.camera = cam_obj1

    cam_obj1.data.ortho_scale = scene_length 
    bpy.context.scene.render.resolution_x = 100 * scene_length 
    bpy.context.scene.render.resolution_y = 100 * scene_length
    bpy.context.scene.render.resolution_percentage = 100 
    bpy.context.scene.render.dither_intensity = 0.
    return cam_obj1

def update_range(cam, floor_range):
    cam_z = cam.location[-1]
    cam.data.clip_start = cam_z - floor_range[1]
    cam.data.clip_end = cam_z - floor_range[0]

def import_gibson_v1_rooms(room_dir):
    objs = [l for l in os.listdir(room_dir) 
              if os.path.splitext(l)[-1] == '.obj']
    for o in objs:
        bpy.ops.import_scene.obj(
            filepath = os.path.join(room_dir, o),
            use_edges = True,
            use_smooth_groups = True, 
            use_split_objects = False,
            use_split_groups = False,
            use_groups_as_vgroups = False,
            use_image_search = False,
            split_mode = 'OFF')


def import_ig_scene_wall(scene_dir):
    wall_obj_path = os.path.join(scene_dir, 
                                'shape', 'visual', 'wall_vm.obj')
    bpy.ops.import_scene.obj(
            filepath = wall_obj_path,
            axis_up='Z', axis_forward='Y',
            use_edges = True,
            use_smooth_groups = True, 
            use_split_objects = False,
            use_split_groups = False,
            use_groups_as_vgroups = False,
            use_image_search = False,
            split_mode = 'OFF')

def set_up_render_with_background_color(color=(1.0,1.0,1.0)):
    bpy.context.scene.render.film_transparent = True

    # switch on nodes and get reference
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # create input image node
    in_node = tree.nodes.new(type='CompositorNodeRLayers')
    in_node.location = -400,0

    #create mix node
    mix_node = tree.nodes.new(type='CompositorNodeMixRGB')
    mix_node.location = 400,0
    mix_node.inputs[1].default_value = (*color, 1.)

    # create output node
    out_node = tree.nodes.new('CompositorNodeComposite')   
    out_node.location = 800,0

    # link nodes
    tree.links.new(in_node.outputs[0], mix_node.inputs[2])
    tree.links.new(mix_node.outputs[0], out_node.inputs[0])
    tree.links.new(in_node.outputs[1], mix_node.inputs[0])

    # with redirect_output():
        # bpy.ops.render.render( write_still=True)
