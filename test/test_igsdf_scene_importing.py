#!/usr/bin/env python

from gibson2.simulator import Simulator
from gibson2.scenes.igsdf_scene import iGSDFScene
from gibson2.utils.utils import parse_config
import os
import gibson2

import time

config = parse_config(os.path.join(gibson2.root_path, '../test/test.yaml'))


def test_import_igsdf():
    scene = iGSDFScene('Rs')
    s = Simulator(mode='iggui', image_width=512,
                  image_height=512, )
    s.import_ig_scene(scene)

    # turtlebot1 = Turtlebot(config)
    # s.import_robot(turtlebot1)
    # turtlebot1.set_position([0.5, 0, 3.5])

    start = time.time()
    end = time.time()
    s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
    for i in range(150000000):
        # import pdb
        # pdb.set_trace()
        s.step()
        end = time.time()
        print("Elapsed time: ", end - start)
        print("Frequency: ", 1 / (end - start))
        start = end

    s.disconnect()
    print("end")


def main():
    test_import_igsdf()


if __name__ == "__main__":
    main()
