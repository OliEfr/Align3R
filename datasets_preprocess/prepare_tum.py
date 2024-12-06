# %%
import glob
import os
import shutil
dirs = glob.glob("../data/tum/*/")
dirs = sorted(dirs)
# extract frames
for dir in dirs:
    frames = glob.glob(dir + 'rgb/*.png')
    frames = sorted(frames)
    # sample 110 frames at the stride of 2
    frames = frames[30:80]
    # cut frames after 110
    new_dir = dir + 'rgb_50/'

    for frame in frames:
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(frame, new_dir)
        # print(f'cp {frame} {new_dir}')

    depth_frames = glob.glob(dir + 'depth/*.png')
    depth_frames = sorted(depth_frames)
    # sample 110 frames at the stride of 2
    depth_frames = depth_frames[30:80]
    # cut frames after 110
    new_dir = dir + 'depth_50/'

    for frame in depth_frames:
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(frame, new_dir)
        # print(f'cp {frame} {new_dir}')
import numpy as np
for dir in dirs:
    gt_path = "groundtruth.txt"
    gt = np.loadtxt(dir + gt_path)
    gt_110 = gt[30:80]
    np.savetxt(dir + 'groundtruth_50.txt', gt_110)