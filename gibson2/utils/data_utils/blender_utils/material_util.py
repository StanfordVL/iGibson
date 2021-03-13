import bpy
import glob
import os
from typing import Tuple
import sys
import math

################################################################################
# glob Utility 
################################################################################

def insensitive_glob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))

################################################################################
# Node Utility 
################################################################################

def create_texture_node(node_tree: bpy.types.NodeTree, 
                        name: str, path: str, 
                        is_color_data: bool) -> bpy.types.Node:
    # Instantiate a new texture image node
    texture_node = node_tree.nodes.new(type='ShaderNodeTexImage')
    texture_node.name = name
    # print(path)
    # Open an image and set it to the node
    texture_node.image = bpy.data.images.load(path)

    # Set other parameters
    if bpy.app.version >= (2, 80, 0):
        texture_node.image.colorspace_settings.is_data = False if is_color_data else True
    else:
        texture_node.color_space = 'COLOR' if is_color_data else 'NONE'

    # Return the node
    return texture_node

def create_empty_image(node_tree: bpy.types.NodeTree, 
                       name: str, is_color_data: bool,
                       dim: Tuple[int, int] = (2048, 2048)) -> bpy.types.Node:
    # Instantiate a new texture image node
    texture_node = node_tree.nodes.new(type='ShaderNodeTexImage')
    # Open an image and set it to the node
    width,height = dim
    if name in bpy.data.images:
        texture_node.image = bpy.data.images[name]
    else:
        texture_node.image = bpy.data.images.new(name=name, 
                                width=width, height=height)
    # Set other parameters
    if bpy.app.version >= (2, 80, 0):
        texture_node.image.colorspace_settings.is_data = False if is_color_data else True
    else:
        texture_node.color_space = 'COLOR' if is_color_data else 'NONE'

    # Return the node
    return texture_node

def clean_nodes(nodes: bpy.types.Nodes) -> None:
    for node in nodes:
        nodes.remove(node)

################################################################################
# PBR Utility 
################################################################################

def build_pbr_textured_nodes(node_tree: bpy.types.NodeTree,
                             color_texture_path: str = "",
                             metallic_texture_path: str = "",
                             roughness_texture_path: str = "",
                             normal_texture_path: str = "",
                             displacement_texture_path: str = "",
                             ambient_occlusion_texture_path: str = "",
                             scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                             is_cc0: bool = True) -> None:
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    # principled_node.name = 'BSDF'
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    coord_node = node_tree.nodes.new(type='ShaderNodeTexCoord')
    mapping_node = node_tree.nodes.new(type='ShaderNodeMapping')
    mapping_node.vector_type = 'POINT'
    if bpy.app.version >= (2, 81, 0):
        mapping_node.inputs["Scale"].default_value = scale
    else:
        mapping_node.scale = scale
    node_tree.links.new(coord_node.outputs['UV'], mapping_node.inputs['Vector'])

    if color_texture_path != "":
        texture_node = create_texture_node(node_tree, 'diffuse', color_texture_path, True)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        if ambient_occlusion_texture_path != "":
            ao_texture_node = create_texture_node(node_tree, 'ambient_occlusion', ambient_occlusion_texture_path, False)
            node_tree.links.new(mapping_node.outputs['Vector'], ao_texture_node.inputs['Vector'])
            mix_node = node_tree.nodes.new(type='ShaderNodeMixRGB')
            mix_node.blend_type = 'MULTIPLY'
            node_tree.links.new(texture_node.outputs['Color'], mix_node.inputs['Color1'])
            node_tree.links.new(ao_texture_node.outputs['Color'], mix_node.inputs['Color2'])
            node_tree.links.new(mix_node.outputs['Color'], principled_node.inputs['Base Color'])
        else:
            node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Base Color'])

    if metallic_texture_path != "":
        texture_node = create_texture_node(node_tree, 'metallic',  metallic_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Metallic'])

    if roughness_texture_path != "":
        texture_node = create_texture_node(node_tree, 'roughness', roughness_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Roughness'])

    if normal_texture_path != "":
        texture_node = create_texture_node(node_tree, 'normal',  normal_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        normal_map_node = node_tree.nodes.new(type='ShaderNodeNormalMap')
        if is_cc0:
            separate_map_node = node_tree.nodes.new(type='ShaderNodeSeparateRGB')
            combine_map_node = node_tree.nodes.new(type='ShaderNodeCombineRGB')
            invert_map_node = node_tree.nodes.new(type='ShaderNodeInvert')
            node_tree.links.new(texture_node.outputs['Color'], separate_map_node.inputs['Image'])
            node_tree.links.new(separate_map_node.outputs['R'], combine_map_node.inputs['R'])
            node_tree.links.new(separate_map_node.outputs['B'], combine_map_node.inputs['B'])
            node_tree.links.new(separate_map_node.outputs['G'], invert_map_node.inputs['Color'])
            node_tree.links.new(invert_map_node.outputs['Color'], combine_map_node.inputs['G'])
            node_tree.links.new(combine_map_node.outputs['Image'], normal_map_node.inputs['Color'])
        else:
            node_tree.links.new(texture_node.outputs['Color'], normal_map_node.inputs['Color'])
        node_tree.links.new(normal_map_node.outputs['Normal'], principled_node.inputs['Normal'])

    if displacement_texture_path != "":
        texture_node = create_texture_node(node_tree, 'displacement', displacement_texture_path, False)
        node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
        node_tree.links.new(texture_node.outputs['Color'], output_node.inputs['Displacement'])

    arrange_nodes(node_tree, use_current_layout_as_initial_guess=False)



def build_pbr_textured_nodes_from_name(material_name: str, 
                        texture_path: dict,
                        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                        is_cc0: bool = True,
                        ) -> bpy.types.Material:
    new_material = bpy.data.materials.new(material_name)
    new_material.use_nodes = True
    clean_nodes(new_material.node_tree.nodes)

    build_pbr_textured_nodes(new_material.node_tree,
                             color_texture_path=texture_path["color"],
                             metallic_texture_path=texture_path["metallic"],
                             roughness_texture_path=texture_path["roughness"],
                             normal_texture_path=texture_path["normal"],
                             displacement_texture_path=texture_path["displacement"],
                             ambient_occlusion_texture_path=texture_path["ambient_occlusion"],
                             scale=scale,
                             is_cc0=is_cc0)
    
    return new_material


################################################################################
# Node Arrange 
################################################################################

def arrange_nodes(node_tree: bpy.types.NodeTree,
                  use_current_layout_as_initial_guess: bool = False,
                  fix_horizontal_location: bool = True,
                  fix_vertical_location: bool = True,
                  fix_overlaps: bool = True,
                  verbose: bool = False) -> None:
    max_num_iters = 2000
    epsilon = 1e-05
    target_space = 50.0

    second_stage = False

    if not use_current_layout_as_initial_guess:
        for node in node_tree.nodes:
            node.location = (0.0, 0.0)

    if verbose:
        print("-----------------")
        print("Target nodes:")
        for node in node_tree.nodes:
            print("- " + node.name)

    # In the first stage, expand nodes overly
    target_space *= 2.0

    # Gauss-Seidel-style iterations
    previous_squared_deltas_sum = sys.float_info.max
    for i in range(max_num_iters):
        squared_deltas_sum = 0.0

        if fix_horizontal_location:
            for link in node_tree.links:
                k = 0.9 if not second_stage else 0.5
                threshold_factor = 2.0

                x_from = link.from_node.location[0]
                x_to = link.to_node.location[0]
                w_from = link.from_node.width
                signed_space = x_to - x_from - w_from
                C = signed_space - target_space
                grad_C_x_from = -1.0
                grad_C_x_to = 1.0

                # Skip if the distance is sufficiently large
                if C >= target_space * threshold_factor:
                    continue

                lagrange = C / (grad_C_x_from * grad_C_x_from + grad_C_x_to * grad_C_x_to)
                delta_x_from = -lagrange * grad_C_x_from
                delta_x_to = -lagrange * grad_C_x_to

                link.from_node.location[0] += k * delta_x_from
                link.to_node.location[0] += k * delta_x_to

                squared_deltas_sum += k * k * (delta_x_from * delta_x_from + delta_x_to * delta_x_to)

        if fix_vertical_location:
            k = 0.5 if not second_stage else 0.05
            socket_offset = 20.0

            def get_from_socket_index(node, node_socket):
                for i in range(len(node.outputs)):
                    if node.outputs[i] == node_socket:
                        return i
                assert False

            def get_to_socket_index(node, node_socket):
                for i in range(len(node.inputs)):
                    if node.inputs[i] == node_socket:
                        return i
                assert False

            for link in node_tree.links:
                from_socket_index = get_from_socket_index(link.from_node, link.from_socket)
                to_socket_index = get_to_socket_index(link.to_node, link.to_socket)
                y_from = link.from_node.location[1] - socket_offset * from_socket_index
                y_to = link.to_node.location[1] - socket_offset * to_socket_index
                C = y_from - y_to
                grad_C_y_from = 1.0
                grad_C_y_to = -1.0
                lagrange = C / (grad_C_y_from * grad_C_y_from + grad_C_y_to * grad_C_y_to)
                delta_y_from = -lagrange * grad_C_y_from
                delta_y_to = -lagrange * grad_C_y_to

                link.from_node.location[1] += k * delta_y_from
                link.to_node.location[1] += k * delta_y_to

                squared_deltas_sum += k * k * (delta_y_from * delta_y_from + delta_y_to * delta_y_to)

        if fix_overlaps and second_stage:
            k = 0.9
            margin = 0.5 * target_space

            # Examine all node pairs
            for node_1 in node_tree.nodes:
                for node_2 in node_tree.nodes:
                    if node_1 == node_2:
                        continue

                    x_1 = node_1.location[0]
                    x_2 = node_2.location[0]
                    w_1 = node_1.width
                    w_2 = node_2.width
                    cx_1 = x_1 + 0.5 * w_1
                    cx_2 = x_2 + 0.5 * w_2
                    rx_1 = 0.5 * w_1 + margin
                    rx_2 = 0.5 * w_2 + margin

                    # Note: "dimensions" and "height" may not be correct depending on the situation
                    def get_height(node):
                        if node.dimensions.y > epsilon:
                            # Note: node.dimensions.y seems to store twice the value of node.height
                            return node.dimensions.y / 2.0
                        elif math.fabs(node.height - 100.0) > epsilon:
                            return node.height
                        else:
                            return 200.0

                    y_1 = node_1.location[1]
                    y_2 = node_2.location[1]
                    h_1 = get_height(node_1)
                    h_2 = get_height(node_2)
                    cy_1 = y_1 - 0.5 * h_1
                    cy_2 = y_2 - 0.5 * h_2
                    ry_1 = 0.5 * h_1 + margin
                    ry_2 = 0.5 * h_2 + margin

                    C_x = math.fabs(cx_1 - cx_2) - (rx_1 + rx_2)
                    C_y = math.fabs(cy_1 - cy_2) - (ry_1 + ry_2)

                    # If no collision, just skip
                    if C_x >= 0.0 or C_y >= 0.0:
                        continue

                    # Solve collision for the "easier" direction
                    if C_x > C_y:
                        grad_C_x_1 = 1.0 if cx_1 - cx_2 >= 0.0 else -1.0
                        grad_C_x_2 = -1.0 if cx_1 - cx_2 >= 0.0 else 1.0
                        lagrange = C_x / (grad_C_x_1 * grad_C_x_1 + grad_C_x_2 * grad_C_x_2)
                        delta_x_1 = -lagrange * grad_C_x_1
                        delta_x_2 = -lagrange * grad_C_x_2

                        node_1.location[0] += k * delta_x_1
                        node_2.location[0] += k * delta_x_2

                        squared_deltas_sum += k * k * (delta_x_1 * delta_x_1 + delta_x_2 * delta_x_2)
                    else:
                        grad_C_y_1 = 1.0 if cy_1 - cy_2 >= 0.0 else -1.0
                        grad_C_y_2 = -1.0 if cy_1 - cy_2 >= 0.0 else 1.0
                        lagrange = C_y / (grad_C_y_1 * grad_C_y_1 + grad_C_y_2 * grad_C_y_2)
                        delta_y_1 = -lagrange * grad_C_y_1
                        delta_y_2 = -lagrange * grad_C_y_2

                        node_1.location[1] += k * delta_y_1
                        node_2.location[1] += k * delta_y_2

                        squared_deltas_sum += k * k * (delta_y_1 * delta_y_1 + delta_y_2 * delta_y_2)

        if verbose:
            print("Iteration #" + str(i) + ": " + str(previous_squared_deltas_sum - squared_deltas_sum))

        # Check the termination conditiion
        if math.fabs(previous_squared_deltas_sum - squared_deltas_sum) < epsilon:
            if second_stage:
                break
            else:
                target_space = 0.5 * target_space
                second_stage = True

        previous_squared_deltas_sum = squared_deltas_sum


