import numpy as np
import os

import gibson2
from gibson2 import object_states
from gibson2.simulator import Simulator
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import URDFObject
from gibson2.utils.assets_utils import get_ig_model_path
from gibson2.object_states.factory import prepare_object_states


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

    s = Simulator(mode='gui')

    scene = EmptyScene()
    s.import_scene(scene)

    model_path = os.path.join(get_ig_model_path('sink', 'sink_1'), 'sink_1.urdf')

    sink = URDFObject(filename=model_path,
                      category='sink',
                      name='sink_1',
                      scale=np.array([0.8, 0.8, 0.8]),
                      abilities={'toggleable': {}, 'water_source': {}}
                      )

    s.import_object(sink)
    sink.set_position([1, 1, 0.8])
    sink.states[object_states.ToggledOn].set_value(True)

    block = YCBObject(name='036_wood_block')
    s.import_object(block)
    block.set_position([1, 1, 1.8])
    block.abilities = ["soakable", "cleaning_tool"]
    prepare_object_states(block, abilities={"soakable": {}, "cleaning_tool": {}})
    # assume block can soak water

    model_path = os.path.join(get_ig_model_path('breakfast_table', '19203'), '19203.urdf')
    desk = URDFObject(filename=model_path,
                      category='table',
                      name='19898',
                      scale=np.array([0.8, 0.8, 0.8]),
                      abilities={'dustyable': {}}
                      )

    print(desk.states.keys())
    desk.states[object_states.Dirty].set_value(True)
    s.import_object(desk)
    desk.set_position([1, -2, 0.4])

    # Main simulation loop
    try:
        while True:
            s.step()
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
