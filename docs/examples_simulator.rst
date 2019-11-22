Examples of simulator
=======================

Create a simple simulated environment
---------------------------------------

Here we show how to craete a simple simulated environment with one mesh and one robot. Use the config file in `gibsonv2/examples/configs/turtlebot_p2p_nav.yaml`.

.. code-block:: python

    import yaml
    from gibson2.core.physics.robot_locomotors import Turtlebot
    from gibson2.core.simulator import Simulator
    from gibson2.core.physics.scene import BuildingScene, StadiumScene
    from gibson2.utils.utils import parse_config
    import pytest
    import pybullet as p
    import numpy as np

    config = parse_config('turtlebot_p2p_nav.yaml')

    s = Simulator(mode='headless')
    scene = BuildingScene('Ohopee')
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)
    for i in range(100):
        s.step()

    s.disconnect()



Simulate a robot and many objects
---------------------------------------

Here we show how to create a simulated environment with many objects


.. code-block:: python

    import yaml
    from gibson2.core.physics.robot_locomotors import Turtlebot
    from gibson2.core.simulator import Simulator
    from gibson2.core.physics.scene import BuildingScene, StadiumScene
    from gibson2.utils.utils import parse_config
    import pytest
    import pybullet as p
    import numpy as np

    config = parse_config('turtlebot_p2p_nav.yaml')

    s = Simulator(mode='headless')
    scene = BuildingScene('Ohopee')
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)
    for i in range(100):
        s.step()

        
    for i in range(30):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)

    s.disconnect()
