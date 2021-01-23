import os
import argparse

DIRECT_QUERY_SUB = {
    'scale': 'mechanical_kitchen_scale',

}


def rename_mtl(args):
    obj_dir = args.obj_dir
    op_sys = args.op_sys
    img_exts = ['jpg', 'jpeg', 'png']
    for fname in os.listdir(obj_dir):
        if not fname.endswith('.mtl'):
            continue
        fname = os.path.join(obj_dir, fname)
        with open(fname, 'r') as f:
            lines = f.readlines()

        # with open(fname + '.new', 'w') as f:
        with open(fname, 'w') as f:
            for line in lines:
                has_img = False
                for img_ext in img_exts:
                    if '.' + img_ext in line:
                        has_img = True
                        break
                if has_img:
                    sep = '/' if op_sys == 'linux' else '\\'
                    original_path = line.strip().split()[-1]
                    print(original_path)
                    img_file = original_path.split(sep)[-1]
                    print(img_file)
                    img_full_path = os.path.join(obj_dir, img_file)
                    assert os.path.isfile(
                        img_full_path), (fname, img_full_path)
                    line = line.replace(original_path, img_file)
                f.write(line)
        f.close()


def main(args):
    rename_mtl(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_dir', required=True, type=str)
    parser.add_argument('--op_sys', required=True, type=str,
                        choices=['linux', 'windows'])
    args = parser.parse_args()
    main(args)
