import os

import gibson2
from gibson2 import object_states
from gibson2.objects.ycb_object import YCBObject
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.robots.behavior_robot import BehaviorRobot
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.simulator import Simulator


def main():
    # HDR files for PBR rendering
    hdr_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
    hdr_texture2 = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
    light_modulation_map_filename = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
    background_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        enable_pbr=True,
        msaa=False,
        light_dimming_factor=1.0
    )

    # VR system settings
    s = Simulator(mode='vr', rendering_settings=vr_rendering_settings)
    # scene = InteractiveIndoorScene('Rs_int')
    scene = EmptyScene()
    s.import_scene(scene)

    # Create a BEHAVIOR Robot and it will handle all initialization and importing under-the-hood
    vr_agent = BehaviorRobot(s)
    s.set_vr_start_pos([0, 0, 0], vr_height_offset=-0.1)

    # Objects to interact with
    mustard_start = [-1, 1.55, 1.2]
    mustard = YCBObject('006_mustard_bottle')
    s.import_object(mustard, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
    mustard.set_position(mustard_start)

    # Main simulation loop
    while True:
        s.step()

        # Update VR objects
        vr_agent.update()

        in_hand = mustard.states[object_states.InHandOfRobot].get_value()
        in_reach = mustard.states[object_states.InReachOfRobot].get_value()
        in_same_room = mustard.states[object_states.InSameRoomAsRobot].get_value()

        print("Mustard in hand: %r, in reach: %r, in same room: %r" % (in_hand, in_reach, in_same_room))

    s.disconnect()


if __name__ == '__main__':
    main()
