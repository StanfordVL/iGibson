import os
import subprocess

dst_folder = "/cvgl2/u/chengshu/atus_dataset/processed"
src_folder = "/cvgl2/u/chengshu/gibsonv2/igibson/data/assets/models/processed"
obj_list_file = "/cvgl2/u/chengshu/gibsonv2/igibson/utils/data_utils/ext_object/scripts_wip/catalog_new.txt"


def main():
    obj_folder_list = []
    obj_category_list = []
    with open(obj_list_file) as f:
        for line in f.readlines():
            tokens = line.strip().split()
            obj_folder_list.append(tokens[0])
            obj_category_list.append(tokens[2])
    for obj_folder, obj_category in zip(obj_folder_list, obj_category_list):
        object_id = os.path.basename(obj_folder)
        obj_folder = os.path.join(src_folder, obj_folder, "blender_fixed")
        dst_obj_folder = os.path.join(dst_folder, obj_category, object_id)
        if os.path.isdir(dst_obj_folder):
            continue
        print(dst_obj_folder)
        subprocess.call(
            "./process_object_new_dir.sh {} {} {} {}".format(
                obj_folder,
                dst_folder,
                obj_category,
                object_id,
            ),
            shell=True,
        )


if __name__ == "__main__":
    main()
