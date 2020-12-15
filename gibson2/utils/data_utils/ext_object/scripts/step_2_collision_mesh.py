import os
import subprocess
import argparse
import threading

NUM_THREADS = 32
parser = argparse.ArgumentParser('gen all vhacd')
parser.add_argument('--object_name', dest='object_name')
parser.add_argument('--input_dir', dest='input_dir')
parser.add_argument('--output_dir', dest='output_dir')
parser.add_argument('--split_loose', dest='split_merge', 
                    action='store_true')

args = parser.parse_args()

if not os.path.isdir(args.input_dir):
    raise ValueError('Input directory not found: {}'.format(args.input_dir))
    quit()

os.makedirs(args.output_dir, exist_ok=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

if args.split_merge:
    tmp_dir = os.path.join(args.output_dir, 'tmp', 'split')
    os.makedirs(tmp_dir, exist_ok=True)
    ########################
    # split to loose parts #
    ########################
    cmd = 'cd {} && blender -b --python step_2_split.py -- {} {}'.format(
            script_dir, args.input_dir, tmp_dir)
    subprocess.call(cmd, shell=True,
        stdout=subprocess.DEVNULL)
    input_dir = tmp_dir
else:
    input_dir = args.input_dir

tmp_dir = os.path.join(args.output_dir, 'tmp', 'vhacd')
os.makedirs(tmp_dir, exist_ok=True)
objs = [o for o in os.listdir(input_dir) if os.path.splitext(o)[1] == '.obj']
print('Inititating V-HACD for {} meshes...'.format(len(objs)))
def vhacd(cmd):
    subprocess.call(cmd, shell=True,
        stdout=subprocess.DEVNULL)
threads = []
for o in objs:
    in_f = os.path.join(input_dir, o)
    out_f = os.path.join(tmp_dir, o)
    cmd = '../../blender_utils/vhacd --input {} --output {}'.format(in_f, out_f)
    thread = threading.Thread(target=vhacd, args=(cmd,))
    thread.start()
    threads.append(thread)

print('Waiting for finishing...')
for thread in threads:
    thread.join()        

print('Merging V-HACD...')
###########################
# Merge all V-HACD to one #
###########################
cmd = 'cd {} && blender -b --python step_2_merge.py -- {} {} {}'.format(
        script_dir, args.object_name, tmp_dir, args.output_dir)
subprocess.call(cmd, shell=True,
    stdout=subprocess.DEVNULL)

tmp_dir = os.path.join(args.output_dir, 'tmp')
cmd = 'rm -r {}'.format(tmp_dir)
subprocess.call(cmd, shell=True)
