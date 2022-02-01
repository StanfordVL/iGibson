import bddl
from bddl.activity import *
from igibson.tasks.bddl_backend import IGibsonBDDLBackend
from igibson.action_generators.motion_primitive_generator import MotionPrimitiveActionGenerator
from igibson.objects.articulated_object import URDFObject
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_model_path

behavior_activity = "re-shelving_library_books_simple"
activity_definition = 0
simulator_name = "igibson"

s = Simulator(mode="gui_non_interactive", image_width=512, image_height=512, device_idx=0, use_pb_gui=True)
scene = InteractiveIndoorScene(
    "Rs_int",
    urdf_file="/home/ziang/Workspace/Github/ig_dataset/scenes/Rs_int/urdf/Rs_int_task_re-shelving_library_books_simple_0_0_fixed_furniture",
    load_object_categories=["walls", "floors", "ceilings", "breakfast_table", "shelf", "hardback"]
)
s.import_scene(scene)

ref = {"book": "hardback",
       "table": "breakfast_table",
       "floor": "floors"}

conds = Conditions(behavior_activity, activity_definition, simulator_name)
scope = get_object_scope(conds)
backend = IGibsonBDDLBackend()                      # TODO pass in backend from iGibson
init = get_initial_conditions(conds, backend, scope)
goal = get_goal_conditions(conds, backend, scope)
for obj_cat in conds.parsed_objects:
    for obj_inst in conds.parsed_objects[obj_cat]:
        print("DEBUG", obj_inst)
        cat = obj_inst.split('.')[0]
        if cat in ref:
            cat = ref[cat]
        scope[obj_inst] = scene.objects_by_category[cat][0]

print(scene.objects_by_name.enumerate())

populated_scope = scope              # TODO populate scope in iGibson, e.g. through sampling
goal = get_goal_conditions(conds, backend, populated_scope)
ground = get_ground_goal_state_options(conds, backend, populated_scope, goal)
# natural_init = get_natural_initial_conditions(conds)
# natural_init = get_natural_goal_conditions(conds)


print("####### Initial #######")
print(init)
print()
print("####### Goal #######")
print(goal)
print()
print("####### Ground #######")
print(ground)
print()
# print("####### Natural language conditions #######")
# print(natural_init)
# print(natural_goal)

print("####### Goal evaluation #######")
for __ in range(100):
    print(evaluate_goal_conditions(ground[0]))
