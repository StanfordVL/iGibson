import platform
from igibson.render.mesh_renderer.get_available_devices import get_available_devices


def test_device():
    assert len(get_available_devices()) > 0


def test_binding():
    if platform.system() == 'Darwin':
        from igibson.render.mesh_renderer import GLFWRendererContext
        r = GLFWRendererContext.GLFWRendererContext(256, 256)
    else:
        from igibson.render.mesh_renderer import EGLRendererContext
        r = EGLRendererContext.EGLRendererContext(256, 256, get_available_devices()[0])
    r.init()
    r.release()
