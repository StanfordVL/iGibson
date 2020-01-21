from gibson2.core.render.mesh_renderer import MeshRendererContext
from gibson2.core.render.mesh_renderer.get_available_devices import get_available_devices


def test_device():
    assert len(get_available_devices()) > 0


def test_binding():
    r = MeshRendererContext.MeshRendererContext(256, 256, get_available_devices()[0])
    r.init()
    r.release()
