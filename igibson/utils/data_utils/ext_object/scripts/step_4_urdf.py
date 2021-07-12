import os
import subprocess
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser("Generate URDF for rigid_body")
parser.add_argument('--input_dir', dest='input_dir')
parser.add_argument('--mass', dest='mass', type=float, default=10.)

visual_component_template = '''<visual name="{obj_name}">
      <origin xyz="0. 0. 0."/>
      <geometry>
        <mesh filename="{viz_mesh}"/>
      </geometry>
    </visual>
'''

collision_component_template = '''<collision>
      <origin xyz="0. 0. 0."/>
      <geometry>
        <mesh filename="{collision_mesh}"/>
      </geometry>
    </collision>
'''

def get_viz_string(obj_name, viz_mesh):
    return visual_component_template.format(
            obj_name=obj_name, viz_mesh=viz_mesh)

def get_col_string(obj_name, col_mesh):
    return collision_component_template.format(
            obj_name=obj_name, collision_mesh=col_mesh)

structure_base_template = '''<?xml version="1.0" ?>
<robot name="{}">
  <link name="base_link">
    <inertial>
      <origin xyz="0. 0. 0." />
      <mass value="{}" />
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001" />
    </inertial>
    {{}}
  </link>
</robot>
'''

def convert_to_urdf_rigid(input_dir, mass=10.):
    # get base urdf
    input_dir = os.path.normpath(input_dir)

    vm_dir = os.path.join(input_dir, 'shape', 'visual')
    cm_dir = os.path.join(input_dir, 'shape', 'collision')

    model_id = os.path.basename(input_dir)
    urdf_path = os.path.join(input_dir, '{}.urdf'.format(model_id))

    base_urdf = structure_base_template.format(model_id, mass, '')
    # add all visual mesh components
    components = ''
    vms = [f for f in os.listdir(vm_dir) 
             if os.path.splitext(f)[1] == '.obj']
    for m in vms:
        components += get_viz_string(
                os.path.splitext(m)[0], 
                os.path.join('shape', 'visual', m)) 
    # add all collision mesh components
    cms = [f for f in os.listdir(cm_dir) 
             if os.path.splitext(f)[1] == '.obj']
    for m in cms:
        components += get_col_string(
                os.path.splitext(m)[0], 
                os.path.join('shape', 'collision', m)) 

    with open(urdf_path, 'w') as fp:
        fp.write( base_urdf.format(components) )

if __name__ == '__main__':
    args = parser.parse_args()
    convert_to_urdf_rigid(args.input_dir, args.mass)
