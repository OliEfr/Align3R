# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for SceneFlow
# --------------------------------------------------------
import os.path as osp
from glob import glob
import itertools
import numpy as np
import re
import cv2

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


#
#
# split = 'train'
#
# ROOT = "/media/8TB/tyhuang/video_depth/SceneFlow"
#
# outscene_list = ["Monkaa_proc"]
#
# # if split == 'train':
# #     outscene_list = ["FlyingThings3D_proc", "Driving_proc", "Monkaa_proc"]
# # elif split == 'test':
# #     outscene_list = ["FlyingThings3D_proc"]
#
# scene_list = []
# for outscene in outscene_list:
#     if outscene == "FlyingThings3D_proc":
#         split_folder = "TRAIN" if split == 'train' else "TEST"
#         scene_list.extend(sorted(glob(osp.join(ROOT, outscene, split_folder, '*/*/*'))))
#     if outscene == "Driving_proc":
#         scene_list.extend(sorted(glob(osp.join(ROOT, outscene, '*/*/*/*'))))
#     if outscene == "Monkaa_proc":
#         scene_list.extend(sorted(glob(osp.join(ROOT, outscene, '*/*'))))
#
#
# pair_dict = {}
# pair_num = 0
# for scene in scene_list:
#     depth_files = sorted(glob(osp.join(scene, '*_depth.pfm')))
#     mask_files = sorted(glob(osp.join(scene, '*_mask.png')))
#
#     max_depth = 0
#
#     for depth_file, mask_file in zip(depth_files, mask_files):
#
#         depth = readPFM(depth_file)
#
#         maskmap = imread_cv2(mask_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
#         maskmap = (maskmap / 255.0) > 0.1
#         # update the depthmap with mask
#
#         maskmap = (maskmap * (depth<400)).astype(np.float32)
#         cv2.imwrite(mask_file, (maskmap * 255).astype(np.uint8))
#
#         # depth *= maskmap
#         #
#         # maxdepth = np.max(depth) if np.max(depth) > max_depth else max_depth






class SceneFlowDatasets(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT                        # ROOT = "/media/tyhuang/T9/videodepth_data/SceneFlow"
        super().__init__(*args, **kwargs)

        self.dataset_label = 'SceneFlow'

        if split == 'train':
            self.outscene_list = ["Driving_proc", "Monkaa_proc","FlyingThings3D_proc"]
        elif split == 'test':
            self.outscene_list = ["FlyingThings3D_proc"]

        scene_list = []
        for outscene in self.outscene_list:
            if outscene == "FlyingThings3D_proc":
                split_folder = "TRAIN" if split == 'train' else "TEST"
                scene_list.extend(sorted(glob(osp.join(ROOT, outscene, split_folder, '*/*/*'))))
            if outscene == "Driving_proc":
                scene_list.extend(sorted(glob(osp.join(ROOT, outscene, '*/*/*/*'))))
            if outscene == "Monkaa_proc":
                scene_list.extend(sorted(glob(osp.join(ROOT, outscene, '*/*'))))

        self.pair_dict = {}
        pair_num = 0
        for scene in scene_list:
          
            imgs = sorted(glob(osp.join(scene, '*_rgb.jpg')))

            len_imgs = len(imgs)
            combinations = [(i, j) for i, j in itertools.combinations(range(len_imgs), 2) if abs(i - j) <= 10 ]
            # if "FlyingThings3D_proc" in scene:
            #     combinations = [(i, j) for i, j in itertools.combinations(range(len_imgs), 2)]
            # if "Driving_proc" in scene:
            #     if "fast" in scene:
            #         combinations = [(i, j) for i, j in itertools.combinations(range(len_imgs), 2)
            #                         if 0 < abs(i - j) <= 8 or (abs(i - j) <= 20 and abs(i - j) % 5 == 0)]
            #     elif "slow" in scene:
            #         combinations = [(i, j) for i, j in itertools.combinations(range(len_imgs), 2)
            #                         if abs(i - j) <= 12 or (abs(i - j) <= 25 and abs(i - j) % 5 == 0)]
            # if "Monkaa_proc" in scene:
            #     combinations = [(i, j) for i, j in itertools.combinations(range(len_imgs), 2)
            #                     if abs(i - j) <= 12 or (abs(i - j) <= 25 and abs(i - j) % 5 == 0)]

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
            depthmap = readPFM(depthmap_path)
            pred_depth = np.load(img_path.replace('.jpg', '_pred_depth_' + self.depth_prior_name + '.npz'))#['depth']
            focal_length_px = pred_depth['focallength_px']#[0][0]
            pred_depth = pred_depth['depth']
            pred_depth = self.pixel_to_pointcloud(pred_depth, focal_length_px)
            maskmap = imread_cv2(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            maskmap = (maskmap / 255.0) > 0.1
            #maskmap = maskmap * (depthmap<100)
            depthmap *= maskmap
            
            #pred_depth = pred_depth#/20.0
            metadata = np.load(metadata_path)
            intrinsics = np.float32(metadata['camera_intrinsics'])
            camera_pose = np.float32(metadata['camera_pose'])
            # max_depth = np.float32(metadata['maximum_depth'])
            #
            # depthmap = (depthmap.astype(np.float32) / 10.0)
            # camera_pose[:3, 3] /= 10.0

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

    dataset = SceneFlowDatasets(split='train', ROOT="/media/tyhuang/T9/videodepth_data/SceneFlow", resolution=512, aug_crop=16)

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