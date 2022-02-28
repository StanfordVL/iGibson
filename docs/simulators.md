# Simulator

### Overview

**Simulator** maintains an instance of **Renderer** and **PhysicsEngine** and provides APIs to import **Scene**, **Object** and **Robot** into both of them and keep them synchronized at all time.

Some key functions are the following:
- `load`: initialize PyBullet physics engine and MeshRenderer
- `import_{scene, ig_scene}`: import the scene into PyBullet by calling `scene.load`, and then import it into MeshRenderer by calling `self.renderer.add_instance_group`. If `InteractiveIndoorScene` is imported using `import_scene`, all objects in the scene are also imported.
- `import_{object, articulated_object, robot}`: import the object, articulated object and robot into the simulator in a similar manner
- `sync`: synchronize the poses of the dynamic objects (including the robots) between PyBullet and MeshRenderer. Specifically, it calls `update_position` for each object, in which it retrieve the object's pose in PyBullet, and then update its pose accordingly in MeshRenderer.

If `Simulator` uses `gui` mode, by default it will also maintain a `Viewer`, which essentially is a virtual camera in the scene that can render images. More info about the `Viewer` can be found here: [igibson/render/viewer.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/render/viewer.py). Notably, you can press `h` in the `ExternalView` window to show the help menu for mouse/keyboard control.

Most of the code can be found here: [igibson/simulator.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/simulator.py).

### Examples
In this example, we import a scene (type selected by the user), a `Turtlebot`, and ten `YCBObject` into the simulator.

```{literalinclude} ../igibson/examples/objects/load_objects.py
:language: python
```
