from igibson.audio.audio_system import AcousticMesh
from igibson.audio.acoustic_material_mapping import ResonanceMaterialToId
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene, StaticIndoorScene
from PIL import Image
from igibson.objects import cube
import csv
import os
import numpy as np

Image.MAX_IMAGE_PIXELS = 1000000000  

MatterportToResonanceMaterialMap = {
    "couch" : "CurtainHeavy",
    "armchair" : 'PlywoodPanel',
    "ceiling" : "ConcreteBlockPainted",
    "wall" : "ConcreteBlockPainted",
    "floor" : "WoodPanel",
    "shower floor" : 'PolishedConcreteOrTile',
    "shower wall" : 'PolishedConcreteOrTile',
    "blinds" : "CurtainHeavy",
    "kitchen shelf" : 'PlywoodPanel',
    "bed" :  "CurtainHeavy",
    "picture" : 'PlywoodPanel',
    "curtain" : "CurtainHeavy",
    "sofa chair" : "CurtainHeavy",
    "door" : "WoodPanel",
    "partition" : "ConcreteBlockPainted",
    "door frame" : "PlywoodPanel",
    "window" : "GlassThick"
}

def buildMatterportCategories():
    matterportSemanticClassToCategory = {}
    with open("category_mapping.tsv") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            if i > 0:
                matterportSemanticClassToCategory[int(row[0])] = row[2]
    return matterportSemanticClassToCategory

def classesToMaterials(classes):
    materials = []
    unknowns = {}
    mapped_faces = 0
    for c in classes:
        if c in MatterportToResonanceMaterialMap:
            res_material = MatterportToResonanceMaterialMap[c]
            materials.append(ResonanceMaterialToId[res_material])
            mapped_faces += 1
        else:
            materials.append(ResonanceMaterialToId["Uniform"])
            if c in unknowns:
                unknowns[c] += 1
            else:
                unknowns[c] = 1
    biggest_unknowns = sorted(unknowns, key=unknowns.get, reverse=True)
    print("Acoustic mapping successful for {}% of mesh".format(mapped_faces * 100 / len(classes)))
    if len(biggest_unknowns) > 0:
        print("Unmapped Matterport classes found, substituting with transparent, printing largest 10")
        for i in range(min(10, len(biggest_unknowns))):
            print("  Class: " + biggest_unknowns[i] + "  Num faces: " + str(unknowns[biggest_unknowns[i]]))
    return np.asarray(materials)

def getMatterportAcousticMesh(s, sem_map_fn):
    sem_map_img = Image.open(sem_map_fn)
    sem_map = np.array(sem_map_img) // 16

    categoryMap = buildMatterportCategories()
    verts, faces, classes = [], [], []

    #iterate over data already loaded by renderer rather than re-loading all of this
    for vertex_data in s.renderer.vertex_data:
        vertex_positions = vertex_data[:,0:3]
        texcoords = vertex_data[:,6:8]
        for i in range(0, texcoords.shape[0], 3):
            classes_for_face = []
            for offset in range(3):
                # Convert uv coordinates to pixel location
                u =  texcoords[i+offset,0]
                v = texcoords[i+offset,1]
                if u < 0 or v < 0 or u > 1 or v > 1:
                    vertex_class = -1
                else:
                    pixel_x = int((1-v) * sem_map.shape[1] - 0.1) 
                    pixel_y = int(u * sem_map.shape[0] - 0.1)
                    # Get pixel values
                    vertex_class_rgb = sem_map[pixel_x, pixel_y]
                    #Decode pixel rgb to class ID
                    vertex_class = int(np.round(vertex_class_rgb[0]) + \
                                        np.round(vertex_class_rgb[1]) * 16.0 + \
                                        np.round(vertex_class_rgb[2] * 16.0 * 16.0))
                classes_for_face.append(vertex_class)
            # majority vote for face class from vertex classes
            face_class = max(set(classes_for_face), key = classes_for_face.count)
            classes.append(categoryMap.get(face_class, "undefined"))
        faces.extend(list(range(len(faces), len(faces) + len(vertex_positions))))
        vert_flattened = np.empty((vertex_positions.size,), dtype=vertex_positions.dtype)
        vert_flattened[0::3] = vertex_positions[:,0]
        vert_flattened[1::3] = vertex_positions[:,1]
        vert_flattened[2::3] = vertex_positions[:,2]
        verts.extend(vert_flattened)
    materials = classesToMaterials(classes)

    mesh = AcousticMesh()
    mesh.verts = np.asarray(verts)
    mesh.faces = np.asarray(faces)
    mesh.materials = materials

    return mesh

# Main simply exercises code and puts colored cubes to verify material segmentation
if __name__=="__main__":
    s = Simulator(mode='iggui', image_width=512, image_height=512)
    scene = StaticIndoorScene('17DRP5sb8fy', pybullet_load_texture=True)
    s.import_scene(scene)
    mesh = getMatterportAcousticMesh(s, "/cvgl/group/Gibson/matterport3d-downsized/v2/17DRP5sb8fy/sem_map.png")

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

