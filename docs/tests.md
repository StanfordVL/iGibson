# Tests
### Overview
We provide tests in [test](https://github.com/StanfordVL/iGibson/tree/master/test). You can run them by this:
```bash
cd test
pytest
```
It will take a few minutes. If all tests pass, you will see something like this
```bash
=============================== test session starts ================================
platform linux -- Python 3.5.6, pytest-4.6.9, py-1.5.3, pluggy-0.13.1
rootdir: /cvgl2/u/chengshu/gibsonv2
plugins: openfiles-0.3.0, doctestplus-0.1.3, arraydiff-0.2
collected 27 items

test_binding.py ..                                                           [  7% ]
test_navigate_env.py ..                                                      [ 14% ]
test_object.py ....                                                          [ 29% ]
test_render.py ...                                                           [ 40% ]
test_robot.py ..........                                                     [ 77% ]
test_scene_importing.py ....                                                 [ 92% ]
test_simulator.py .                                                          [ 96% ]
test_viewer.py
```
We will further improve our test coverage in the next few weeks.

