""" This VR hand dexterity benchmark allows the user to interact with many types of objects
and interactive objects, and provides a good way to qualitatively measure the dexterity of a VR hand.

You can use the left and right controllers to start/stop/reset the timer,
as well as show/hide its display. The "overlay toggle" action and its
corresponding button index mapping can be found in the vr_config.yaml file in the gibson2 folder.
"""
import os
import pybullet as p
import pybullet_data
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrAgent
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_utils import VrTimer
from gibson2 import assets_path

# Objects in the benchmark - corresponds to Rs kitchen environment, for range of items and
# transferability to the real world
# Note: the scene will automatically load in walls/ceilings/floors in addition to these objects
benchmark_names = [
    'door_54',
    'trash_can_25',
    'counter_26',
    'bottom_cabinet_39',
    'fridge_40',
    'bottom_cabinet_41',
    'sink_42',
    'microwave_43',
    'dishwasher_44',
    'oven_45',
    'bottom_cabinet_46',
    'top_cabinet_47',
    'top_cabinet_48',
    'top_cabinet_49',
    'top_cabinet_50',
    'top_cabinet_51'
]

# Set to true to print Simulator step() statistics
PRINT_STATS = True
# Set to true to use gripper instead of VR hands
USE_GRIPPER = False
# Set to true to print object information
PRINT_OBJECT_INFO = True

# HDR files for PBR rendering
hdr_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

def main():
    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(optimized=True,
                                                fullscreen=False,
                                                env_texture_filename=hdr_texture,
                                                env_texture_filename2=hdr_texture2,
                                                env_texture_filename3=background_texture,
                                                light_modulation_map_filename=light_modulation_map_filename,
                                                enable_shadow=True, 
                                                enable_pbr=True,
                                                msaa=True,
                                                light_dimming_factor=1.0)
    s = Simulator(mode='vr', 
                use_fixed_fps = True,
                rendering_settings=vr_rendering_settings)

    scene = InteractiveIndoorScene('Rs_int')
    scene._set_obj_names_to_load(benchmark_names)
    s.import_ig_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    vr_agent = VrAgent(s, use_gripper=USE_GRIPPER)
    # Move VR agent to the middle of the kitchen
    s.set_vr_start_pos(start_pos=[0,2.1,0], vr_height_offset=-0.02)

    # Mass values to use for each object type - len(masses) objects will be created of each type
    masses = [1, 5, 10]

    # List of objects to load with name: filename, type, scale, base orientation, start position, spacing vector and spacing value
    obj_to_load = {
        'mustard': ('006_mustard_bottle', 'ycb', 1, (0.0, 0.0, 0.0, 1.0), (0.0, 1.6, 1.18), (-1, 0, 0), 0.15),
        'marker': ('040_large_marker', 'ycb', 1, (0.0, 0.0, 0.0, 1.0), (1.5, 2.6, 0.92), (0, -1, 0), 0.15),
        'can': ('005_tomato_soup_can', 'ycb', 1, (0.0, 0.0, 0.0, 1.0), (1.7, 2.6, 0.95), (0, -1, 0), 0.15),
        'drill': ('035_power_drill', 'ycb', 1, (0.0, 0.0, 0.0, 1.0), (1.5, 2.2, 1.15), (0, -1, 0), 0.2),
        'small_jenga': ('jenga/jenga.urdf', 'pb', 1, (0.000000, 0.707107, 0.000000, 0.707107), (-0.9, 1.6, 1.18), (-1, 0, 0), 0.1),
        'large_jenga': ('jenga/jenga.urdf', 'pb', 2, (0.000000, 0.707107, 0.000000, 0.707107), (-1.3, 1.6, 1.31), (-1, 0, 0), 0.15),
        'small_duck': ('duck_vhacd.urdf', 'pb', 1, (0.000000, 0.000000, 0.707107, 0.707107), (-1.8, 1.95, 1.12), (1, 0, 0), 0.15),
        'large_duck': ('duck_vhacd.urdf', 'pb', 2, (0.000000, 0.000000, 0.707107, 0.707107), (-1.95, 2.2, 1.2), (1, 0, 0), 0.2),
        'small_sphere': ('sphere_small.urdf', 'pb', 1, (0.000000, 0.000000, 0.707107, 0.707107), (-0.5, 1.63, 1.15), (-1, 0, 0), 0.15),
        'large_sphere': ('sphere_small.urdf', 'pb', 2, (0.000000, 0.000000, 0.707107, 0.707107), (-0.5, 1.47, 1.15), (-1, 0, 0), 0.15)
    }

    for name in obj_to_load:
        fpath, obj_type, scale, orn, pos, space_vec, space_val = obj_to_load[name]
        for i in range(len(masses)):
            if obj_type == 'ycb':
                handle = YCBObject(fpath, scale=scale)
            elif obj_type == 'pb':
                handle = ArticulatedObject(fpath, scale=scale)
            
            s.import_object(handle, use_pbr=False, use_pbr_mapping=False)
            # Calculate new position along spacing vector
            new_pos = (pos[0] + space_vec[0] * space_val * i, pos[1] + space_vec[1] * space_val * i, pos[2] + space_vec[2] * space_val * i)
            handle.set_position(new_pos)
            handle.set_orientation(orn)
            p.changeDynamics(handle.body_id, -1, mass=masses[i])

        print('Information for object: {}'.format(name))
        print('Masses: {}'.format(masses))
        aabbMin, aabbMax = p.getAABB(handle.body_id)
        x = aabbMax[0] - aabbMin[0]
        y = aabbMax[1] - aabbMin[1]
        z = aabbMax[2] - aabbMin[2]
        print('Dimensions - x: {}, y: {}, z: {}'.format(x, y, z))

    # Time how long demo takes
    time_text = s.add_vr_overlay_text(text_data='Current time: NOT STARTED', font_size=100, font_style='Bold', 
                            color=[0,0,0], pos=[100, 100])
    timer = VrTimer()

    # Main simulation loop
    while True:
        s.step(print_stats=PRINT_STATS)

        # Events that manage timer functionality
        r_toggle = s.query_vr_event('right_controller', 'overlay_toggle')
        l_toggle = s.query_vr_event('left_controller', 'overlay_toggle')
        # Overlay toggle action on right controller is used to start/stop timer
        if r_toggle and not l_toggle:
            if timer.is_timer_running():
                timer.stop_timer()
            else:
                timer.start_timer()
        # Overlay toggle action on left controller is used to show/hide timer
        elif l_toggle and not r_toggle:
            time_text.set_show_state(not time_text.get_show_state())
        # Reset timer if both toggle buttons are pressed at once
        elif r_toggle and l_toggle:
            timer.refresh_timer()

        # Update timer value
        time_text.set_text('Current time: {}'.format(round(timer.get_timer_val(), 1)))

        # Update VR agent
        vr_agent.update()

    s.disconnect()


if __name__ == '__main__':
    main()