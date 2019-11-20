Environments
=============


BaseEnv
------------

.. autoclass:: gibson2.envs.base_env.BaseEnv

   .. automethod:: __init__
   .. automethod:: reload
   .. automethod:: load
   .. automethod:: clean
   .. automethod:: simulator_step


NavigateEnv
-------------

.. autoclass:: gibson2.envs.locomotor_env.NavigateEnv

	.. automethod:: __init__
	.. automethod:: step
	.. automethod:: reset