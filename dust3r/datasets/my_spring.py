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

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


class SpringDatasets(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT                        # ROOT = "/media/tyhuang/T9/videodepth_data/spring_proc/train"
        super().__init__(*args, **kwargs)

        self.dataset_label = 'Spring'
        test_scenes = []

        scene_list = []
        for scene in os.listdir(ROOT):
            if scene not in test_scenes and split == 'train':
                scene_list.append(osp.join(ROOT, scene))
            if scene in test_scenes and split == 'test':
                scene_list.append(osp.join(ROOT, scene))

        self.pair_dict = {}
        pair_num = 0
        for scene in scene_list:
            imgs = sorted(glob(osp.join(scene, '*_rgb.jpg')))

            len_imgs = len(imgs)
            # combinations = [(i, j) for i, j in itertools.combinations(range(len_imgs), 2)
            #                 if abs(i - j) <= 20 or (abs(i - j) <= 60 and abs(i - j) % 3 == 0)]
            combinations = [(i, j) for i, j in itertools.combinations(range(len_imgs), 2) if abs(i - j) <= 10 ]

            for (i, j) in combinations:
                self.pair_dict[pair_num] = [imgs[i], imgs[j]]
                pair_num += 1

    def __len__(self):
        return len(self.pair_dict)


    def _get_views(self, idx, resolution, rng):

        views = []
        for img_path in self.pair_dict[idx]:
            rgb_image = imread_cv2(img_path)

            depthmap_path = img_path.replace('_rgb.jpg', '_depth.pfm')
            mask_path = img_path.replace('_rgb.jpg', '_mask.png')
            metadata_path = img_path.replace('_rgb.jpg', '_metadata.npz')
            pred_depth = np.load(img_path.replace('.jpg', '_pred_depth.npz'))#['depth']
            focal_length_px = pred_depth['focallength_px']
            pred_depth = pred_depth['depth']
            pred_depth = self.pixel_to_pointcloud(pred_depth, focal_length_px)
            depthmap = readPFM(depthmap_path)
            #scale = depthmap.min()+depthmap.min()
            maskmap = imread_cv2(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            maskmap = (maskmap / 255.0) > 0.1
            #maskmap = maskmap * (depthmap<100)
            depthmap *= maskmap

            metadata = np.load(metadata_path)
            intrinsics = np.float32(metadata['camera_intrinsics'])
            camera_pose = np.float32(metadata['camera_pose'])
            # max_depth = np.float32(metadata['maximum_depth'])

            # depthmap = (depthmap.astype(np.float32) / 200.0)
            # pred_depth = pred_depth/200.0
            # camera_pose[:3, 3] /= 200.0

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

    dataset = SpringDatasets(split='train', ROOT="/media/8TB/tyhuang/video_depth/spring_proc/train", resolution=512, aug_crop=16)

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
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()