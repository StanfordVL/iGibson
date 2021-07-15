import argparse
import os
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser("Generate URDF for rigid_body")
parser.add_argument("--input_dir", dest="input_dir")
parser.add_argument("--mass", dest="mass", type=float, default=10.0)
parser.add_argument("--urdf", dest="urdf")

visual_component_template = """<visual name="{obj_name}">
      <origin xyz="0. 0. 0."/>
      <geometry>
        <mesh filename="{viz_mesh}"/>
      </geometry>
    </visual>
"""

collision_component_template = """<collision>
      <origin xyz="0. 0. 0."/>
      <geometry>
        <mesh filename="{collision_mesh}"/>
      </geometry>
    </collision>
"""


def get_viz_string(obj_name, viz_mesh):
    return visual_component_template.format(obj_name=obj_name, viz_mesh=viz_mesh)


def get_col_string(obj_name, col_mesh):
    return collision_component_template.format(obj_name=obj_name, collision_mesh=col_mesh)


structure_base_template = """<?xml version="1.0" ?>
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
"""


def convert_to_urdf_rigid(input_dir, mass):
    # Build a naive URDF from scratch for rigid objects
    # get base urdf
    input_dir = os.path.normpath(input_dir)

    vm_dir = os.path.join(input_dir, "shape", "visual")
    cm_dir = os.path.join(input_dir, "shape", "collision")

    model_id = os.path.basename(input_dir)
    urdf_path = os.path.join(input_dir, "{}.urdf".format(model_id))

    base_urdf = structure_base_template.format(model_id, mass, "")
    # add all visual mesh components
    components = ""
    vms = [f for f in os.listdir(vm_dir) if os.path.splitext(f)[1] == ".obj"]
    for m in vms:
        components += get_viz_string(os.path.splitext(m)[0], os.path.join("shape", "visual", m))
    # add all collision mesh components
    cms = [f for f in os.listdir(cm_dir) if os.path.splitext(f)[1] == ".obj"]
    for m in cms:
        components += get_col_string(os.path.splitext(m)[0], os.path.join("shape", "collision", m))

    with open(urdf_path, "w") as fp:
        fp.write(base_urdf.format(components))


def convert_to_urdf_articulated(input_dir, mass, urdf):
    # Build upon an existing URDF for articulated objects
    vm_dir = os.path.join("shape", "visual")
    cm_dir = os.path.join("shape", "collision")

    tree = ET.parse(urdf)
    root = tree.getroot()
    for link in tree.findall("link"):
        link_name = link.attrib["name"]
        vms = link.findall("visual")
        if len(vms) == 0:
            # Remove links that do not have visual meshes and also their joints
            fake_base_link = tree.find('link[@name="{}"]'.format(link_name))
            root.remove(fake_base_link)
            fake_base_joint = None
            for joint in root.findall("joint"):
                if joint.find("parent").attrib["link"] == link_name:
                    fake_base_joint = joint
                    break
            assert fake_base_joint is not None
            root.remove(fake_base_joint)
            continue

        for vm in vms:
            old_filename = vm.find("geometry/mesh").attrib["filename"]
            new_filename = os.path.join(vm_dir, os.path.basename(old_filename))
            vm.find("geometry/mesh").attrib["filename"] = new_filename

        cms = link.findall("collision")
        assert len(cms) > 0
        # Make sure all collision meshes have the same origins
        all_origin_atrribs = [cm.find("origin").attrib for cm in cms]
        assert all_origin_atrribs.count(all_origin_atrribs[0]) == len(
            all_origin_atrribs
        ), "collision meshes within the same link has different origins"

        # Overwrite filename for the first collision mesh
        new_filename = os.path.join(cm_dir, "{}_cm.obj".format(link_name))
        cms[0].find("geometry/mesh").attrib["filename"] = new_filename

        # Remove other collision meshes (if exist)
        for i in range(1, len(cms)):
            link.remove(cms[i])

    model_id = os.path.basename(input_dir)
    urdf_path = os.path.join(input_dir, "{}.urdf".format(model_id))
    with open(urdf_path, "wb+") as f:
        tree.write(f)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.urdf is None:
        convert_to_urdf_rigid(args.input_dir, args.mass)
    else:
        convert_to_urdf_articulated(args.input_dir, args.mass, args.urdf)
