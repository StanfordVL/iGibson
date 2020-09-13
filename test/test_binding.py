import platform
if platform.system() == 'Darwin':
    from gibson2.render.mesh_renderer import GLFWRendererContext as MeshRendererContext
else:
    from gibson2.render.mesh_renderer import EGLRendererContext as MeshRendererContext

from gibson2.render.mesh_renderer.get_available_devices import get_available_devices


def test_device():
    assert len(get_available_devices()) > 0


def test_binding():
    r = MeshRendererContext.MeshRendererContext(256, 256, get_available_devices()[0])
    r.init()
    r.release()
