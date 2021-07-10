"""
BEHAVIOR demo batch analysis script
"""

import argparse
import datetime
import os
import glob
import json
from pathlib import Path

import tasknet

from gibson2.examples.demo.vr_demos.atus.behavior_demo_segmentation import run_default_segmentation


def parse_args():
    parser = argparse.ArgumentParser(
        description='Save checkpoints for BEHAVIOR demos.')
    parser.add_argument('--vr_log_directory', type=str,
                        help='Path containing demos to replay')
    return parser.parse_args()


def main():
    args = parse_args()
    demo_names = ['sorting_mail_0_Rs_int_2021-05-30_22-56-33', 'sorting_mail_0_Rs_int_2021-06-04_15-06-41', 'storing_food_0_Rs_int_2021-05-31_11-45-10', 'storing_food_0_Rs_int_2021-05-31_11-49-30', 'storing_food_0_Rs_int_2021-06-05_19-14-59', 'storing_the_groceries_0_Wainscott_0_int_2021-05-23_22-13-12', 'storing_the_groceries_0_Wainscott_0_int_2021-06-04_15-16-12', 'storing_the_groceries_0_Wainscott_0_int_2021-06-04_17-13-16', 'thawing_frozen_food_0_Wainscott_0_int_2021-05-23_22-01-08', 'thawing_frozen_food_0_Wainscott_0_int_2021-06-01_20-55-26', 'thawing_frozen_food_0_Wainscott_0_int_2021-06-05_19-50-38', 'throwing_away_leftovers_0_Ihlen_1_int_2021-06-04_19-48-55', 'throwing_away_leftovers_0_Ihlen_1_int_2021-06-04_19-52-40', 'throwing_away_leftovers_0_Ihlen_1_int_2021-06-05_19-10-02', 'unpacking_suitcase_0_Ihlen_1_int_2021-05-24_18-12-48', 'unpacking_suitcase_0_Ihlen_1_int_2021-06-02_21-18-54', 'unpacking_suitcase_0_Ihlen_1_int_2021-06-04_15-28-22', 'vacuuming_floors_0_Benevolence_2_int_2021-05-23_18-27-15', 'vacuuming_floors_0_Benevolence_2_int_2021-06-02_19-31-10', 'vacuuming_floors_0_Benevolence_2_int_2021-06-04_15-35-58', 'washing_cars_or_other_vehicles_0_Ihlen_0_int_2021-05-28_14-03-36', 'washing_cars_or_other_vehicles_0_Ihlen_0_int_2021-06-02_21-25-15', 'washing_cars_or_other_vehicles_0_Ihlen_0_int_2021-06-04_15-43-02', 'washing_dishes_0_Wainscott_0_int_2021-05-27_16-50-19', 'washing_dishes_0_Wainscott_0_int_2021-05-31_18-28-33', 'washing_dishes_0_Wainscott_0_int_2021-06-04_15-49-38', 'washing_floor_0_Ihlen_1_int_2021-05-23_21-00-46', 'washing_floor_0_Ihlen_1_int_2021-06-02_21-32-45', 'washing_floor_0_Ihlen_1_int_2021-06-04_15-56-16', 'washing_pots_and_pans_0_Benevolence_1_int_2021-06-07_19-10-24', 'washing_pots_and_pans_0_Benevolence_1_int_2021-06-07_20-01-56', 'washing_pots_and_pans_0_Benevolence_1_int_2021-06-07_20-10-22', 'watering_houseplants_0_Beechwood_0_int_2021-05-23_21-51-52', 'watering_houseplants_0_Beechwood_0_int_2021-05-31_18-46-34', 'watering_houseplants_0_Beechwood_0_int_2021-06-04_16-03-26', 'waxing_cars_or_other_vehicles_0_Ihlen_0_int_2021-05-27_16-32-32', 'waxing_cars_or_other_vehicles_0_Ihlen_0_int_2021-06-02_21-49-34', 'waxing_cars_or_other_vehicles_0_Ihlen_0_int_2021-06-03_19-17-09']
    vr_logs = [os.path.join(args.vr_log_directory, "%s.hdf5" % demo_name) for demo_name in demo_names]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tasknet.set_backend("iGibson")
    result = {}
    for vr_log_path in vr_logs:
        if "replay" in vr_log_path:
            continue
        segmentation_save_directory = os.path.join(args.vr_log_directory, "segmentation")
        if not os.path.exists(segmentation_save_directory):
            os.mkdir(segmentation_save_directory)
        try:
            run_default_segmentation(vr_log_path, segmentation_save_directory)
            demo_statistics = {
                "demo_id": Path(vr_log_path).name,
                "failed": False,
            }
        except Exception as e:
            print("Demo failed withe error: ", e)
            demo_statistics = {
                "demo_id": Path(vr_log_path).name,
                "failed": True,
                "failure_reason": str(e)
            }

        result[Path(vr_log_path).name] = demo_statistics
        with open("./behavior_batch_segmentation_{}.json".format(timestamp), "w") as file:
            json.dump(result, file)


if __name__ == "__main__":
    main()
