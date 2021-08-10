import json
import os

import igibson

LINK_NAME = None  # TODO: Add the link name you want to add here.
OFFSETS = {
    # (category_name, object_name): [x, y, z]
    # TODO: Add the offsets you want to add here.
}


def get_category_directory(category):
    return os.path.join(igibson.ig_dataset_path, "objects", category)


def get_metadata_filename(objdir):
    return os.path.join(objdir, "misc", "metadata.json")


def main():
    for (cat, objdir), offset in OFFSETS.items():
        cd = get_category_directory(cat)
        objdirfull = os.path.join(cd, objdir)

        mfn = get_metadata_filename(objdirfull)
        print("Starting %s" % mfn)
        with open(mfn, "r") as mf:
            meta = json.load(mf)

        if "links" not in meta:
            meta["links"] = dict()

        meta["links"][LINK_NAME] = list(offset)

        with open(mfn, "w") as mf:
            json.dump(meta, mf)
            print("Updated %s" % mfn)


if __name__ == "__main__":
    main()
