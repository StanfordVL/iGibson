from gibson2.core.render.viewer import Viewer


def test_viewer():
    viewer = Viewer()
    for i in range(100):
        viewer.update()
