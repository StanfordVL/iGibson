(Deprecated) Examples of navigation tasks
=========================================

Point2Point Navigation
---------------------------

.. code-block:: python

    config_filename = os.path.join(os.path.dirname(gibson2.__file__),'../examples/configs/turtlebot_p2p_nav.yaml')
    nav_env = NavigateEnv(config_file=config_filename,
      mode=args.mode,
      action_timestep=1.0 / 10.0,
      physics_timestep=1.0 / 40.0)

    for episode in range(10):
        print('Episode: {}'.format(episode))
        nav_env.reset()
        for step in range(500):  # 500 steps, 50s world time
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)



Interactive Navigation - Door opening
---------------------------------------

.. code-block:: python

    config_filename = os.path.join(os.path.dirname(gibson2.__file__),'../examples/configs/jr2_interactive_nav.yaml')
    nav_env = InteractiveNavigateEnv(config_file=config_filename,
      mode=args.mode,
      action_timestep=1.0 / 10.0,
      physics_timestep=1.0 / 40.0)
                              
    for episode in range(10):
        print('Episode: {}'.format(episode))
        nav_env.reset()
        for step in range(500):  # 500 steps, 50s world time
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)




Interactive Navigation - Among movable objects
-------------------------------------------------

.. code-block:: python

    config_filename = os.path.join(os.path.dirname(gibson2.__file__),'../examples/configs/turtlebot_p2p_nav.yaml')
    nav_env = InteractiveGibsonNavigateEnv(config_file=config_filename,
      mode=args.mode,
      action_timestep=1.0 / 10.0,
      physics_timestep=1.0 / 40.0)
                              
    for episode in range(10):
        print('Episode: {}'.format(episode))
        nav_env.reset()
        for step in range(500):  # 500 steps, 50s world time
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)
