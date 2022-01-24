from igibson.robots.fetch import Fetch
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene, StaticIndoorScene
from igibson.objects.ycb_object import YCBObject
from igibson.objects import cube
from igibson.utils.utils import parse_config
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from igibson.utils.mesh_util import ortho
import cv2
from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.matterport_acoustic_mesh import getMatterportAcousticMesh
from igibson.utils.mesh_util import lookat, mat2xyz, ortho, perspective, quat2rotmat, safemat2quat, xyz2mat, xyzw2wxyz
from audio_system import AudioSystem

class FakeViewer:
    def __init__(self):
        self.px, self.py, self.pz = 0, 1, 0
        self.phi, self.theta = 0, 0

def main():
    scene_choices = [
        "1pXnuDYAj8r",
        #"1LXtFkjw3qL",
        #"Ihlen_0_int",
    ]
    scene_trav_map_size = {
        "1pXnuDYAj8r": 4000,
        #"1LXtFkjw3qL": 3000,
        #"Ihlen_0_int": 2000
    }
    for scene_id in scene_choices:
        map_size = scene_trav_map_size[scene_id]
        settings = MeshRendererSettings(enable_shadow=False, msaa=False)
        s = Simulator(mode="headless", image_width=map_size, image_height=map_size, rendering_settings=settings)
        scene = StaticIndoorScene(
            scene_id, 
            #trav_map_resolution = 0.1,
        )
        
        s.import_scene(scene)
        renderer = s.renderer
        print(scene.trav_map_original_size)

        points = scene.get_points_grid(5)[0]

        acousticMesh = getMatterportAcousticMesh(s, "/cvgl/group/Gibson/matterport3d-downsized/v2/{}/sem_map.png".format(scene_id))
        fake_viewer = FakeViewer()
        audioSystem = AudioSystem(s, fake_viewer, acousticMesh, is_Viewer=True)
        s.attachAudioSystem(audioSystem)


        #s.step()
        obj1 = cube.Cube(pos=points[0], dim=[0.1,0.1,0.01], visual_only=True, mass=0, color=[1,0,0,1])
        obj2 = cube.Cube(pos=points[-1], dim=[0.1,0.1,0.01], visual_only=True, mass=0, color=[0,0,1,1])
        s.import_object(obj1)
        s.import_object(obj2)
        obj_id = obj1.get_body_id()
        
        # Attach wav file to imported cube obj
        audioSystem.registerSource(obj_id, "440Hz_44100Hz.wav", enabled=True)
        # Ensure source continuously repeats
        audioSystem.setSourceRepeat(obj_id)
        s.step()

        camera_pose = np.array([0, 0, -1.0])
        view_direction = np.array([0, 0, 1.0])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
        # cache original P and recover for robot cameras
        p_range = map_size / 200.0
        renderer.P = ortho(-p_range, p_range, -p_range, p_range, -10, 20.0)
        intrinsics = renderer.get_intrinsics()
        frame, three_d = renderer.render(modes=("rgb", "3d"))
        depth = -three_d[:, :, 2]
        # white bg
        frame[depth == 0] = 1.0
        frame = cv2.flip(frame, 0)
        bg = (frame[:, :, 0:3][:, :, ::-1] * 255).astype(np.uint8)
        print(bg.shape)
        overlay = np.ones_like(bg) * 255
        cv2.imwrite("floorplan/{}.png".format(scene_id), bg)
        print(intrinsics)


        for idx, world_point in enumerate(points):
            #print(world_point)
            #renderer_pt = renderer.P @ renderer.V @ np.append(world_point, 1)
            #print(renderer_pt)
            #camera_coords = renderer.transform_pose(np.append(world_point, [1, 0, 0, 0]))
            camera_coords = renderer.transform_point(world_point)
            renderer_pt = renderer.P @ np.append(camera_coords, 1)
            print("World point: {}, Camera point: {}, renderer_pt: {}".format(world_point, camera_coords, renderer_pt))
            #fake_viewer.px = world_point[0]
            #fake_viewer.py = world_point[1]
            #fake_viewer.pz = world_point[2]
            #s.step()
            #obj = cube.Cube(pos=world_point, dim=[0.01, 0.01, 0.01], visual_only=True, mass=0, color=[1,0,0,0.5])
            #s.import_object(obj)
            py = int(renderer_pt[0] * renderer_pt[3] * map_size) // 2 + map_size // 2
            px = int(renderer_pt[1] * renderer_pt[3] * map_size) // 2 + map_size // 2
            #py = int(-1 * renderer_pt[0])
            #px = int(-1 * renderer_pt[1])
            for i in range(10):
                for j in range(10):
                    if idx == 0:
                        overlay[px + i, py + j] = [0, 0, 255]
                    elif idx == len(points) - 1:
                        overlay[px + i, py + j] = [255, 0, 0]
                    else:
                        overlay[px + i, py + j] = [0, 0, 0]



        cv2.imwrite("floorplan/{}_trav.png".format(scene_id), overlay)
        img = cv2.addWeighted(bg, 0.9, overlay, 0.4, 0)
        cv2.imwrite("floorplan/{}_combined.png".format(scene_id), img)

        s.disconnect()


if __name__ == "__main__":
    main()
