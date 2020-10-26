import os
import subprocess
import glob
import sys

model_path = sys.argv[1]
save_dir = os.path.join(model_path, 'videos')
os.makedirs(save_dir, exist_ok=True)
pattern = '{}/0000/*_0??0.png'.format(model_path)
print(pattern)
images = [os.path.basename(f) for f in glob.glob(pattern)]

for i in images:
    print(i[:-4])
    cmd = 'ffmpeg -framerate 2 -i {root}/%4d/{imname} -y -r 16 -c:v libx264 -pix_fmt yuvj420p {save}/{vid_name}.mp4'.format(
            root=model_path, imname=i, save=save_dir,vid_name=i[:-4])
    subprocess.call(cmd, shell=True)
