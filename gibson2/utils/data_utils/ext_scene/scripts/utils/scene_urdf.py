import os
import subprocess
import argparse

parser = argparse.ArgumentParser("Generate URDF for scene structure")
parser.add_argument('--model_dir', dest='model_dir')

viz_mesh= '''<visual name="{obj_name}">
                    <origin xyz="0. 0. 0."/>
                    <geometry>
                            <mesh filename="{viz_mesh}"/>
                    </geometry>
            </visual>
            '''
col_mesh = '''<collision>
                    <origin xyz="0. 0. 0."/>
                    <geometry>
                            <mesh filename="{collision_mesh}"/>
                    </geometry>
            </collision>
            '''
def get_component_string(obj_name, viz_mesh, collision_mesh,
                         shift=(0.,0.,0.), mass='5000'):
    shift_x,shift_y,shift_z = shift
    return structure_component_template.format(
            obj_name=obj_name, viz_mesh=viz_mesh, collision_mesh=collision_mesh,
            shift_x=shift_x, shift_y=shift_y, shift_z=shift_z, mass=mass)

orig_urdf='''<?xml version="1.0" ?>
<robot name="igibson_scene">
	<link name="world"/>
	<link category="walls" model="{model_name}" name="walls"/>
	<link category="floors" model="{model_name}" name="floors"/>
	<link category="ceilings" model="{model_name}" name="ceilings"/>
	<joint name="fix_to_world_walls" type="fixed">
		<origin rpy="0 0 0" xyz="0 0 0"/>
		<child link="walls"/>
		<parent link="world"/>
	</joint>
	<joint name="fix_to_world_floors" type="fixed">
		<origin rpy="0 0 0" xyz="0 0 0"/>
		<child link="floors"/>
		<parent link="world"/>
	</joint>
	<joint name="fix_to_world_ceilings" type="fixed">
		<origin rpy="0 0 0" xyz="0 0 0"/>
		<child link="ceilings"/>
		<parent link="world"/>
	</joint>
</robot>
'''

orig_urdf_with_cabinet='''<?xml version="1.0" ?>
<robot name="igibson_scene">
	<link name="world"/>
	<link category="walls" model="{model_name}" name="walls"/>
	<link category="floors" model="{model_name}" name="floors"/>
	<link category="ceilings" model="{model_name}" name="ceilings"/>
	<link category="{save_name}" model="{model_name}" name="{save_name}"/>
	<joint name="fix_to_world_walls" type="fixed">
		<origin rpy="0 0 0" xyz="0 0 0"/>
		<child link="walls"/>
		<parent link="world"/>
	</joint>
	<joint name="fix_to_world_floors" type="fixed">
		<origin rpy="0 0 0" xyz="0 0 0"/>
		<child link="floors"/>
		<parent link="world"/>
	</joint>
	<joint name="fix_to_world_ceilings" type="fixed">
		<origin rpy="0 0 0" xyz="0 0 0"/>
		<child link="ceilings"/>
		<parent link="world"/>
	</joint>
	<joint name="fix_to_world_{save_name}" type="fixed">
		<origin rpy="0 0 0" xyz="0 0 0"/>
		<child link="{save_name}"/>
		<parent link="world"/>
	</joint>
</robot>
'''


structure_base_template = '''<?xml version="1.0" ?>
<robot name="igibson_scene_{}">
	<link name="base_link">
            <inertial>
                <origin xyz="0. 0. 0." />
                <mass value="10000" />
                <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001" />
            </inertial>
        {{}}
	</link>
</robot>
'''

def gen_scene_urdf(model_dir, model_name,component=None):
    # get base urdf
    model_dir = os.path.normpath(model_dir)
    model_id = os.path.basename(model_dir)
    base_urdf = structure_base_template.format(model_id)
    # add all components
    viz_meshdir = os.path.join(model_dir, 'shape', 'visual')
    col_meshdir = os.path.join(model_dir, 'shape', 'collision')

    meshes = os.listdir(viz_meshdir)
    components = ''
    for m in meshes:
        if os.path.splitext(m)[-1] != '.obj':
            continue
        if component is not None and component not in m:
            continue
        components += viz_mesh.format(
                obj_name=os.path.splitext(m)[0],
                viz_mesh='shape/visual/{}'.format(m)) 
    meshes = os.listdir(col_meshdir)
    for m in meshes:
        if os.path.splitext(m)[-1] != '.obj':
            continue
        if component is not None and component not in m:
            continue
        components += col_mesh.format(
                collision_mesh='shape/collision/{}'.format(m)) 

    return base_urdf.format(components)

def gen_orig_urdf(model_name):
    return orig_urdf.format(model_name=model_name)

def gen_orig_urdf_with_cabinet(model_name,save_name):
    return orig_urdf_with_cabinet.format(
                            model_name=model_name,
                            save_name=save_name)

