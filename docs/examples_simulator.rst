(Deprecated) Examples of simulator
==================================

Create a simple simulated environment
---------------------------------------

Here we show how to craete a simple simulated environment with one mesh and one robot. Use the config file in `gibsonv2/examples/configs/turtlebot_p2p_nav.yaml`. This example can be found at `examples/demo/simulator_example.py`. 

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

    s = Simulator(mode='gui', resolution=512)
    scene = BuildingScene('Ohopee')
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)
    for i in range(100):
        s.step()

    s.disconnect()

This example should show something like below, a turtlebot in a house.


.. image:: images/simulator_example.png
    :width: 600

Adding a camera to this example:

.. code-block:: python

    for i in range(100):
        turtlebot.apply_action([0.1,0.1])
        s.step()
        rgb = s.renderer.render_robot_cameras(modes=('rgb'))

With full physical simulation and rendering at 512x512, it runs at about 180 fps (don't forget to switch to headless mode).

If we switch to render to pytorch tensor like below, it can run full physical simulation and rendering at 512x512 at 460 fps.

.. code-block:: python

    s = Simulator(mode='headless', resolution=512, render_to_tensor=False)

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

    for i in range(30):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)

    for i in range(100):
        s.step()

    s.disconnect()


Simulate articulated objects / Complex scenes
------------------------------------------------

.. code-block:: python

    import yaml
    from gibson2.core.physics.robot_locomotors import Turtlebot, JR2_Kinova, Fetch
    from gibson2.core.simulator import Simulator
    from gibson2.core.physics.scene import EmptyScene
    from gibson2.core.physics.interactive_objects import InteractiveObj, BoxShape, YCBObject
    from gibson2.utils.utils import parse_config
    import pytest
    import pybullet as p
    import numpy as np

    config = parse_config('../configs/jr_interactive_nav.yaml')
    s = Simulator(mode='gui')
    scene = EmptyScene()
    s.import_scene(scene)
    jr = JR2_Kinova(config)
    s.import_robot(jr)
    jr.robot_body.reset_position([0,0,0])
    jr.robot_body.reset_orientation([0,0,1,0])
    fetch = Fetch(config)
    s.import_robot(fetch)
    fetch.robot_body.reset_position([0,1,0])
    fetch.robot_body.reset_orientation([0,0,1,0])
    obj = InteractiveObj(filename='/data4/mdv0/cabinet/0007/part_objs/cabinet_0007.urdf')
    s.import_interactive_object(obj)
    obj.set_position([-2,0,0.5])
    obj = InteractiveObj(filename='/data4/mdv0/cabinet/0007/part_objs/cabinet_0007.urdf')
    s.import_interactive_object(obj)
    obj.set_position([-2,2,0.5])
    obj = InteractiveObj(filename='/data4/mdv0/cabinet/0004/part_objs/cabinet_0004.urdf')
    s.import_interactive_object(obj)
    obj.set_position([-2.1, 1.6, 2])
    obj = InteractiveObj(filename='/data4/mdv0/cabinet/0004/part_objs/cabinet_0004.urdf')
    s.import_interactive_object(obj)
    obj.set_position([-2.1, 0.4, 2])
    obj = BoxShape([-2.05,1,0.5], [0.35,0.6,0.5])
    s.import_interactive_object(obj)
    obj = BoxShape([-2.45,1,1.5], [0.01,2,1.5])
    s.import_interactive_object(obj)
    p.createConstraint(0,-1,obj.body_id, -1, p.JOINT_FIXED, [0,0,1], [-2.55,1,1.5], [0,0,0])
    obj = YCBObject('003_cracker_box')
    s.import_object(obj)
    p.resetBasePositionAndOrientation(obj.body_id, [-2,1,1.2], [0,0,0,1])
    obj = YCBObject('003_cracker_box')
    s.import_object(obj)
    p.resetBasePositionAndOrientation(obj.body_id, [-2,2,1.2], [0,0,0,1])

    for i in range(100):
        s.step()

    s.disconnect()


The resulting scene looks like below:

.. image:: images/cabinets.png
    :width: 600
