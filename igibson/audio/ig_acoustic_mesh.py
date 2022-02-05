from igibson.audio.audio_system import AcousticMesh
from igibson.audio.acoustic_material_mapping import ResonanceMaterialToId
import numpy as np
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.objects import cube

#We hardcode these mappings because there is no clean way to directly access the materials of the iGibson mesh if they are loaded with
#randomization off
iGibsonToResonanceMaterialMap = {
    "walls": "ConcreteBlockPainted",
    "floors": "WoodPanel",
    "ceilings": "ConcreteBlockPainted",
    "window": "GlassThick"
}

def dumpFromRenderer(renderer, obj_pb_ids):
    instances_vertices = []
    instances_faces = []
    len_v = 0
    for instance in renderer.instances:
        if instance.pybullet_uuid in obj_pb_ids:
            vertex_info, face_info = instance.dump()
            for v, f in zip(vertex_info, face_info):
                instances_vertices.append(v)
                instances_faces.append(f + len_v)
                len_v += len(v)
    instances_vertices = np.concatenate(instances_vertices, axis=0)
    instances_faces = np.concatenate(instances_faces, axis=0)

    return instances_vertices, instances_faces

def getIgAcousticMesh(simulator):
    vert, face, mat = [], [], []
    #For ig_scenes, we only load a mesh of large, static objects for reverb/reflection baking
    for category in ["walls", "floors", "ceilings", "window"]:
        for obj in simulator.scene.objects_by_category[category]:
            for id in obj.get_body_ids():
                obj_verts, obj_faces = dumpFromRenderer(simulator.renderer, [id])
                mat.extend([ResonanceMaterialToId[iGibsonToResonanceMaterialMap[category]]] * len(obj_faces))
                vert.extend(obj_verts)
                face.extend(obj_faces)
   
    vert, face = np.asarray(vert), np.asarray(face)
    vert_flattened = np.empty((vert.size,), dtype=vert.dtype)
    vert_flattened[0::3] = vert[:,0]
    vert_flattened[1::3] = vert[:,1]
    vert_flattened[2::3] = vert[:,2]

    face_flattened = np.empty((face.size,), dtype=face.dtype)
    face_flattened[0::3] = face[:,0]
    face_flattened[1::3] = face[:,1]
    face_flattened[2::3] = face[:,2]

    material_indices = np.asarray(mat)
    mesh = AcousticMesh()
    mesh.faces = face_flattened
    mesh.verts = vert_flattened
    mesh.materials = material_indices
    return mesh

# Main simply exercises code and puts colored cubes to verify material segmentation
if __name__=="__main__":
    s = Simulator(mode='iggui', image_width=512, image_height=512)
    scene = InteractiveIndoorScene('Beechwood_0_int', texture_randomization=False, object_randomization=False)
    s.import_ig_scene(scene)
    mesh = getIgAcousticMesh(s)

    materials = mesh.materials
    verts = mesh.verts
    faces = mesh.faces

    idxs = np.nonzero(materials==ResonanceMaterialToId["WoodPanel"])[0]
    for i, idx in enumerate(idxs):
        if idx % (len(idxs) // 400) == 0:
            vertex =  [verts[3*faces[3*idx]], verts[3*faces[3*idx]+1], verts[3*faces[3*idx]+2]]
            obj = cube.Cube(pos=vertex, dim=[0.01, 0.01, 0.01], visual_only=True, mass=0, color=[1,0,1,1])
            s.import_object(obj)
    
    idxs = np.nonzero(materials==ResonanceMaterialToId["ConcreteBlockPainted"])[0]
    for i, idx in enumerate(idxs):
        if idx % (len(idxs) // 400) == 0:
            vertex =  [verts[3*faces[3*idx]], verts[3*faces[3*idx]+1], verts[3*faces[3*idx]+2]]
            obj = cube.Cube(pos=vertex, dim=[0.01, 0.01, 0.01], visual_only=True, mass=0, color=[1,0,0,1])
            s.import_object(obj)

    for i in range(1000000000):
        s.step()

    s.disconnect()