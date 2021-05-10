import json
import requests
import sys

PRODUCTION_URL = "http://34.123.89.206:8000"
TEST_URL = "http://34.123.89.206:5000"
url = TEST_URL

SCENES = [
        "Wainscott_0_int",
        "Merom_1_int",
        "Beechwood_0_int"
    ]
ACTIVITY_NAME = "cleaning_up_after_a_meal"
OBJECT_LIST = """(:objects
 	bowl.n.01_1 bowl.n.01_2 - bowl.n.01
	table.n.02_1 - table.n.02
	bag.n.01_1 - bag.n.01
	chair.n.01_1 chair.n.01_2 - chair.n.01
	plate.n.04_1 plate.n.04_2 plate.n.04_3 plate.n.04_4 - plate.n.04
	cup.n.01_1 cup.n.01_2 - cup.n.01
	food.n.02_1 food.n.02_2 - food.n.02
	floor.n.01_1 floor.n.01_2 - floor.n.01
	detergent.n.02_1 - detergent.n.02
	agent.n.01_1 - agent.n.01
)
"""
INITIAL_CONDITIONS = "(:init (ontop bowl.n.01_1 table.n.02_1) (ontop bowl.n.01_2 table.n.02_1) (stained bowl.n.01_1) (stained bowl.n.01_2) (ontop bag.n.01_1 chair.n.01_1) (ontop plate.n.04_1 table.n.02_1) (ontop plate.n.04_2 table.n.02_1) (ontop plate.n.04_3 table.n.02_1) (ontop plate.n.04_4 table.n.02_1) (stained plate.n.04_1) (stained plate.n.04_2) (stained plate.n.04_3) (stained plate.n.04_4) (ontop cup.n.01_1 table.n.02_1) (ontop cup.n.01_2 table.n.02_1) (stained cup.n.01_1) (stained cup.n.01_2) (ontop food.n.02_1 chair.n.01_2) (onfloor food.n.02_2 floor.n.01_1) (onfloor detergent.n.02_1 floor.n.01_1) (stained chair.n.01_1) (stained chair.n.01_2) (stained table.n.02_1) (inroom floor.n.01_1 dining_room) (inroom floor.n.01_2 kitchen) (inroom table.n.02_1 dining_room) (inroom chair.n.01_1 dining_room) (inroom chair.n.01_2 dining_room) (onfloor agent.n.01_1 floor.n.01_2))"
GOAL_CONDITIONS = "(:goal (and (and (forall (?bowl.n.01 - bowl.n.01) (not (stained ?bowl.n.01))) (forall (?plate.n.04 - plate.n.04) (not (stained ?plate.n.04))) (forall (?cup.n.01 - cup.n.01) (not (stained ?cup.n.01)))) (forall (?food.n.02 - food.n.02) (inside ?food.n.02 ?bag.n.01_1)) (onfloor ?bag.n.01_1 ?floor.n.01_1) (and (not (stained ?chair.n.01_2)) (not (stained ?table.n.02_1)))))"


def run_setup(scenes=None):
    scenes = scenes if scenes is not None else SCENES 
    res = requests.post(url + "/setup", data=json.dumps(scenes), verify=False)
    scenes_ids = json.loads(res.text)["scenes_ids"]
    return scenes_ids

def run_check_sampling(scenes_ids, 
                       activity_name=None, 
                       initial_conditions=None,
                       goal_conditions=None,
                       object_list=None):

    activity_name = activity_name if activity_name is not None else ACTIVITY_NAME
    initial_conditions = initial_conditions if initial_conditions is not None else INITIAL_CONDITIONS
    goal_conditions = goal_conditions if goal_conditions is not None else GOAL_CONDITIONS
    object_list = object_list if object_list is not None else OBJECT_LIST

    check_sampling_data = {
        "activityName": activity_name,
        "initialConditions": initial_conditions,
        "goalConditions": goal_conditions,
        "objectList": object_list,
        "scenes_ids": scenes_ids
    }
    res = requests.post(url + "/check_sampling", json.dumps(check_sampling_data))
    result = json.loads(res.text)
    success, feedback, scenes_ids = result["success"], result["feedback"], result["scenes_ids"]
    return success, feedback, scenes_ids

def run_teardown(scenes_ids):
    requests.post(url + "/teardown", json.dumps({"scenes_ids": scenes_ids}))


""" 
# Example usage:

>>> scenes_ids = run_setup()
>>> success, feedback, scenes_ids = run_check_sampling(scenes_ids)
>>> print(success)
True
>>> print(feedback)
{"init_success": "no", "goal_success": "yes", "init_feedback": "...", "goal_feedback": "..."}
>>> run_teardown(scenes_ids)
"""
    