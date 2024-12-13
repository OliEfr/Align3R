# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for Spring
# --------------------------------------------------------
import os.path as osp
from glob import glob
import itertools
import numpy as np
import re
import cv2
import os
import sys
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
TAG_FLOAT = 202021.25
def depth_read(filename):
    """Read depth data from file, return as numpy array."""
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert (
        width > 0 and height > 0 and size > 1 and size < 100000000
    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
        width, height
    )
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

class SintelDatasets(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT                        # ROOT = "/media/8TB/tyhuang/video_depth/vkitti_2.0.3_proc"
        super().__init__(*args, **kwargs)

        self.dataset_label = 'Sintel'
        test_scenes = []

        scene_list = []
        for scene in os.listdir(ROOT):
            scene_list.append(osp.join(ROOT, scene))

        self.pair_dict = {}
        pair_num = 0
        for scene in scene_list:
            imgs = sorted(glob(osp.join(scene, '*.png')))

            len_imgs = len(imgs)
            # combinations = [(i, j) for i, j in itertools.combinations(range(len_imgs), 2)
            #                 if abs(i - j) <= 15 or (abs(i - j) <= 30 and abs(i - j) % 5 == 0)]
            combinations = [(i, j) for i, j in itertools.combinations(range(len_imgs), 2) if abs(i - j) <= 3]

            for (i, j) in combinations:
                self.pair_dict[pair_num] = [imgs[i], imgs[j]]
                pair_num += 1

    def __len__(self):
        return len(self.pair_dict)


    def _get_views(self, idx, resolution, rng):

        views = []
        for img_path in self.pair_dict[idx]:
            rgb_image = imread_cv2(img_path)

            depthmap_path = img_path.replace('MPI-Sintel-training_images', 'MPI-Sintel-depth-training').replace('final/','depth/').replace('.png','.dpt')
            mask_path = img_path.replace('MPI-Sintel-training_images', 'MPI-Sintel-depth-training').replace('final/','dynamic_label_perfect/')
            metadata_path = img_path.replace('MPI-Sintel-training_images', 'MPI-Sintel-depth-training').replace('final/','camdata_left/').replace('.png','.cam')
            
            pred_depth = np.load(img_path.replace('final','depth_prediction_' + self.depth_prior_name).replace('.png', '.npz'))#['depth']
            focal_length_px = pred_depth['focallength_px']
            pred_depth = pred_depth['depth']
            pred_depth = self.pixel_to_pointcloud(pred_depth, focal_length_px)
            depthmap = depth_read(depthmap_path)
            if os.path.exists(mask_path):
              maskmap = imread_cv2(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
              maskmap = (maskmap / 255.0) > 0.1
              #print(maskmap.max())
              #maskmap = maskmap * (depthmap<100)
              depthmap *= maskmap
            intrinsics, extrinsics = cam_read(metadata_path)
            intrinsics, extrinsics = np.array(intrinsics, dtype=np.float32), np.array(extrinsics, dtype=np.float32)
            R = extrinsics[:3,:3]
            t = extrinsics[:3,3]
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R.T
            camera_pose[:3,3] = -R.T @ t
            #camera_pose = np.linalg.inv(camera_pose)
            # max_depth = np.float32(metadata['maximum_depth'])

            # depthmap = (depthmap.astype(np.float32) / 20.0)
            # camera_pose[:3, 3] /= 20.0
            # pred_depth = pred_depth/20.0
            rgb_image, depthmap, pred_depth, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, pred_depth, intrinsics, resolution, rng=rng, info=img_path)

            num_valid = (depthmap > 0.0).sum()
            # assert num_valid > 0
            # if num_valid==0:
            #   depthmap +=0.001
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=img_path,
                instance=img_path,
                pred_depth=pred_depth
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = SintelDatasets(split='train', ROOT="../../data/MPI-Sintel/MPI-Sintel-training_images/training/final", resolution=512, aug_crop=16)

    a = len(dataset)
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
        #     viz.add_pointcloud(pts3d, colors, valid_mask)
        #     viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
        #                    focal=views[view_idx]['camera_intrinsics'][0, 0],
        #                    color=(idx * 255, (1 - idx) * 255, 0),
        #                    image=colors,
        #                    cam_size=cam_size)
        # viz.show()