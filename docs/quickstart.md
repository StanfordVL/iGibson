# Quickstart

### iGibson in Action
Let's get our hands dirty and see iGibson in action.

```bash
cd demo/examples
python env_example.py
```
You should see something like this:
![quickstart.png](images/quickstart.png)

The main window shows PyBullet visualization. The robot (TurtleBot) is moving around with random actions in a realistic house (called "Rs", the one you just downloaded!).

On the right hand side, you can see two windows from our mesh renderer. The top one (RobotView) shows the robot's first person view. The bottom one (ExternalView) shows the view of a virtual camera floating in the air.

If you want to have a virtual tour around the house yourself, you can click on the ExternalView window, and then translate the virtual camera to a different location by pressing "WASD" on your keyboard and rotate it to a different angle by dragging your mouse.

That's it!

### Benchmarks

Performance is a big designing focus for iGibson. We provide a few scripts to benchmark the rendering and physics
simulation framerate in iGibson.

```bash
cd gibson2/demo/examples
python benchmark.py
```

You will see output similar to:
```
physics simulation + rendering rgb, resolution 512, render_to_tensor True: 421.12805140080695 fps
Rendering rgb, resolution 512, render_to_tensor True: 778.2959856272473 fps
Rendering 3d, resolution 512, render_to_tensor True: 857.2466839793148 fps
Rendering normal, resolution 512, render_to_tensor True: 878.6977946996199 fps

physics simulation + rendering rgb, resolution 512, render_to_tensor False: 205.68141718250024 fps
Rendering rgb, resolution 512, render_to_tensor False: 265.74379871537326 fps
Rendering 3d, resolution 512, render_to_tensor False: 292.0761459884919 fps
Rendering normal, resolution 512, render_to_tensor False: 265.70666134193806 fps

```
