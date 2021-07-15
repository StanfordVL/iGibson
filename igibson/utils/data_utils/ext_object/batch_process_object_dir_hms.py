import os
import subprocess

dst_folder = "/cvgl2/u/chengshu/atus_dataset/hms"
src_folder = "/cvgl2/u/chengshu/hms_dataset"


def main():
    for obj_category in os.listdir(src_folder):
        obj_category_folder = os.path.join(src_folder, obj_category)
        for object_id in os.listdir(obj_category_folder):
            obj_folder = os.path.join(obj_category_folder, object_id, "meshes")
            subprocess.call(
                './process_object_new_dir_hms.sh "{}" "{}" "{}" "{}"'.format(
                    obj_folder,
                    dst_folder,
                    obj_category,
                    object_id,
                ),
                shell=True,
            )


if __name__ == "__main__":
    main()
