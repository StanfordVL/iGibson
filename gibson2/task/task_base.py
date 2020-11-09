import numpy as np 
import os 

import tasknet as tn 
from tasknet.task_base import TaskNetTask
import gibson2
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings 
from gibson2.objects.articulated_object import URDFObject, ArticulatedObject
from gibson2.external.pybullet_tools.utils import *


class iGTNTask(TaskNetTask):
    def __init__(self, atus_activity):
        super().__init__(atus_activity)

    def initialize_scene(self):             # NOTE can't have the same method name right? Should TaskNetTask.initialize() be a private method so that this can be initialize()?  
        '''
        Get scene populated with objects such that scene satisfies initial conditions 
        '''
        # Set self.scene_name, self.scene, self.sampled_simulator_objects, and self.sampled_dsl_objects
        self.initialize(InteractiveIndoorScene, ArticulatedObject)

        hdr_texture = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
        hdr_texture2 = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
        light_modulation_map_filename = os.path.join(
            gibson2.ig_dataset_path, 'scenes', self.scene_name, 'layout', 'floor_lighttype_0.png')
        background_texture = os.path.join(
            gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

        settings = MeshRendererSettings(env_texture_filename=hdr_texture,
                                        env_texture_filename2=hdr_texture2,
                                        env_texture_filename3=background_texture,
                                        light_modulation_map_filename=light_modulation_map_filename,
                                        enable_shadow=True, msaa=True,
                                        light_dimming_factor=1.0)
        self.simulator = Simulator(mode='iggui', image_width=960, image_height=720, device_idx=0, rendering_settings=settings)
    
        self.simulator.viewer.min_cam_z = 1.0
        self.simulator.import_ig_scene(self.scene)

        # NOTE not making a separate add_objects function since users shouldn't be able to add extra tasks,
        # that could make the final conditions unsatisfiable or otherwise vacuous. Can change if needed.
        for obj, obj_pos, obj_orn in self.sampled_simulator_objects:
            self.simulator.import_object(obj)
            obj.set_position_orientation(obj_pos, obj_orn)
        
        # Match IDs of simulator and DSL objects 
        for sim_obj, dsl_obj in zip(self.sampled_simulator_objects, self.sampled_dsl_objects):
            dsl_obj.body_id = sim_obj.body_id
            print(dsl_obj.body_id)
            print(dsl_obj.category)

        sys.exit()

    # def check_success(self): 

    #### CHECKERS ####
    def onTop(self, objA, objB):
        '''
        Checks if one object is on top of another. TODO does it need to update TN object representation?
                                                        We've been saying no.
        True iff objA TODO 
        :param objA: simulator object
        :param objB: simulator object 
        '''
        center, extent = get_center_extent(objA.body_id) # TODO: approximate_as_prism
        bottom_aabb = get_aabb(objB.body_id)

        base_center = center - np.array([0, 0, extent[2]])/2
        top_z_min = base_center[2]
        bottom_z_max = bottom_aabb[1][2]
        height_correct = (bottom_z_max - abs(below_epsilon)) <= top_z_min <= (bottom_z_max + abs(above_epsilon))
        bbox_contain = (aabb_contains_point(base_center[:2], aabb2d_from_aabb(bottom_aabb)))
        touching = body_collision(objA.body_id, objB.body_id)

        return height_correct and bbox_contain and touching 
        # return is_placement(objA.body_id, objB.body_id) or is_center_stable(objA.body_id, objB.body_id)

    def inside(self, objA, objB):
        '''
        Checks if one object is inside another. 
        True iff the AABB of objA does not extend past the AABB of objB
        :param objA: simulator object
        :param objB: simulator object 
        '''
        return aabb_contains_aabb(objA.body_id, objB.body_id)

    def nextTo(self, objA, objB):
        '''
        Checks if one object is next to another. 
        True iff the distance between the objects is TODO less than 2/3 of the average
                 side length across both objects' AABBs 
        :param objA: simulator object
        :param objB: simulator object 
        '''
        objA_aabb, objB_aabb = get_aabb(objA.body_id), get_aabb(objB.body_id)
        objA_lower, objA_upper = objA_aabb
        objB_lower, objB_upper = objB_aabb
        distance_vec = []
        for dim in range(3):
            glb = max(objA_lower[dim], objB_lower[dim])
            lub = min(objA_upper[dim], objB_upper[dim])
            distance_vec.append(max(0, glb - lub))
        distance = np.linalg.norm(np.array(distance_vec))
        objA_dims = objA_upper - objA_lower
        objB_dims = objB_upper - objB_lower
        avg_aabb_length = np.mean(objA_dims + objB_dims)

        return distance <= (avg_aabb_length * (2./3.))  # TODO better function
        
    def under(self, objA, objB):
        '''
        Checks if one object is underneath another. 
        True iff the (x, y) coordinates of objA's AABB center are within the (x, y) projection
                 of objB's AABB, and the z-coordinate of objA's AABB upper is less than the 
                 z-coordinate of objB's AABB lower.
        :param objA: simulator object
        :param objB: simulator object 
        ''' 
        within = aabb_contains_point(
                                get_aabb_center(get_aabb(objA.body_id))[:2], 
                                aabb2d_from_aabb(get_aabb(objB.body_id)))
        objA_aabb = get_aabb(objA.body_id)
        objB_aabb = get_aabb(objB.body_id)
        
        within = aabb_contains_point(
                                get_aabb_center(objA_aabb)[:2],
                                aabb2d_from_aabb(objB_aabb))
        below = objA_aabb[1][2] <= objB_aabb[0][2]
        return within and below 
        
    def touching(self, objA, objB):

        return body_collision(objA.body_id, objB.body_id)    

    #### SAMPLERS ####
    def sampleOnTop(self, objA, objB):
        pass 
    
    def sampleInside(self, objA, objB):
        pass
    
    def sampleNextTo(self, objA, objB):
        pass

    def sampleUnder(self, objA, objB):
        pass
    
    def sampleTouching(self, objA, objB):
        pass


def main():
    igtn_task = iGTNTask('kinematic_checker_testing')
    igtn_task.initialize_scene()

    for i in range(500):
        igtn_task.simulator.step()
    print('TASK SUCCESS:', igtn_task.check_success())
    igtn_task.simulator.disconnect()
                
    
if __name__ == '__main__':
    main()

    

