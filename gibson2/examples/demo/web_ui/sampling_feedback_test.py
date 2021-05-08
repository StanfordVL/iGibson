import json
import requests
import sys

PRODUCTION_URL = "http://34.123.89.206:8000"
TEST_URL = "http://34.123.89.206:5000"
url = TEST_URL

SCENES = [
        "Wainscott_0_int",
        "Rs_int",
        "Benevolence_1_int"
    ]
ACTIVITY_NAME = "cleaning_microwave_oven"
OBJECT_LIST = """(:objects
 	microwave.n.02_1 - microwave.n.02
	rag.n.01_1 - rag.n.01
	cabinet.n.01_1 - cabinet.n.01
	sink.n.01_1 - sink.n.01
)
"""
INITIAL_CONDITIONS = "(:init (dusty microwave.n.02_1) (inside rag.n.01_1 cabinet.n.01_1) (inroom microwave.n.02_1 kitchen) (inroom sink.n.01_1 kitchen) (inroom cabinet.n.01_1 kitchen))"
GOAL_CONDITIONS = "(:goal (and (not (dusty ?microwave.n.02_1)) (inside ?rag.n.01_1 ?sink.n.01_1) (not (inside ?rag.n.01_1 ?cabinet.n.01_1))))"


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
    