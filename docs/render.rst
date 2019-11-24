MeshRenderer
==============

.. automodule:: gibson2.core.render
.. automodule:: gibson2.core.render.mesh_renderer
.. automodule:: gibson2.core.render.mesh_renderer.mesh_renderer_cpu

MeshRenderer
--------------
.. autoclass:: gibson2.core.render.mesh_renderer.mesh_renderer_cpu.MeshRenderer

	.. automethod:: __init__
	.. automethod:: setup_framebuffer
	.. automethod:: load_object
	.. automethod:: add_instance
	.. automethod:: add_instance_group
	.. automethod:: add_robot
	.. automethod:: set_camera
	.. automethod:: set_fov
	.. automethod:: get_intrinsics
	.. automethod:: readbuffer
	.. automethod:: render
	.. automethod:: clean
	.. automethod:: release

MeshRendererG2G
----------------

.. autoclass:: gibson2.core.render.mesh_renderer.mesh_renderer_tensor.MeshRendererG2G

	.. automethod:: __init__
	.. automethod:: render_to_tensor


VisualObject
--------------
.. autoclass:: gibson2.core.render.mesh_renderer.mesh_renderer_cpu.VisualObject

	.. automethod:: __init__


InstanceGroup
--------------
.. autoclass:: gibson2.core.render.mesh_renderer.mesh_renderer_cpu.InstanceGroup

	.. automethod:: __init__
	.. automethod:: render


Instance
--------------
.. autoclass:: gibson2.core.render.mesh_renderer.mesh_renderer_cpu.Instance

	.. automethod:: __init__
	.. automethod:: render
	

