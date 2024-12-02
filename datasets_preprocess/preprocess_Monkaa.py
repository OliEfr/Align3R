import argparse
import random
import json
import os
import os.path as osp

import PIL.Image
import numpy as np
import cv2
from glob import glob

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import csv

from dust3r.utils.image import img_to_arr
from utils import *

import dust3r.datasets.utils.cropping as cropping  # noqa
from scipy.spatial.transform import Rotation



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="../data/SceneFlow/Monkaa_proc")
    parser.add_argument("--data_dir", type=str, default="../data/SceneFlow/Monkaa")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--img_size", type=int, default=512,
                        help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
    return parser


# 5352 video sequences

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    assert args.data_dir != args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)

    image_path = sorted(glob(osp.join(args.data_dir, 'frames_finalpass', '*/*')))
    diaparity_path = [im.replace('frames_finalpass', 'disparity') for im in image_path]
    cameradata_path = [im.replace('frames_finalpass', 'camera_data') for im in image_path]
    cameradata_path = [im[:-5] for im in cameradata_path]

    img_size = args.img_size

    focal_length = 35.0
    fx = 1050.0
    fy = 1050.0
    cx = 479.5
    cy = 269.5

    camera_intrinsics = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]]
    )


    for imgs, disp, camdata in zip(image_path, diaparity_path, cameradata_path):

        # each sequence
        img_files = sorted([os.path.join(imgs, file) for file in os.listdir(imgs)])
        disp_files = sorted([os.path.join(disp, file) for file in os.listdir(disp)])

        camdata = camdata + "/camera_data.txt"
        camposes = []
        if "left" in imgs:
            camposes, _ = read_camdata_sceneflow(camdata)
        else:
            _, camposes = read_camdata_sceneflow(camdata)

        # save path
        folder = get_all_folders(imgs)
        save_path = os.path.join(args.output_dir, folder[-2], folder[-1])
        os.makedirs(save_path, exist_ok=True)


        for img_file, disp_file, camepose in zip(img_files, disp_files, camposes):
            input_rgb_image = PIL.Image.open(img_file).convert('RGB')
            input_disp_image = readPFM(disp_file)

            input_disp_image[np.isnan(input_disp_image)] = 1e-3
            input_disp_image[input_disp_image <= 0] = 1e-3

            input_depth =  fx / input_disp_image

            # # for FlyingThings3D
            # assert not (True in (input_depth > 1e6))
            # assert not (True in (input_depth <= 1e6))
            mask = ((input_depth > 0) * (input_depth < 400)).astype(np.float32)


            # rescale the images
            depth_mask = np.stack((input_depth, mask), axis=-1)
            H, W = input_depth.shape

            min_margin_x = min(cx, W - cx)
            min_margin_y = min(cy, H - cy)

            # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
            l, t = int(cx - min_margin_x), int(cy - min_margin_y)
            r, b = int(cx + min_margin_x), int(cy + min_margin_y)
            crop_bbox = (l, t, r, b)
            input_rgb_image, depth_mask, input_camera_intrinsics = cropping.crop_image_depthmap(
                input_rgb_image, depth_mask, camera_intrinsics, crop_bbox)

            # try to set the lower dimension to img_size * 3/4 -> img_size=512 => 384
            scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
            output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
            if max(output_resolution) < img_size:
                # let's put the max dimension to img_size
                scale_final = (img_size / max(H, W)) + 1e-8
                output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)

            input_rgb_image, depth_mask, input_camera_intrinsics = cropping.rescale_image_depthmap(
                input_rgb_image, depth_mask, input_camera_intrinsics, output_resolution)

            input_depthmap = depth_mask[:, :, 0]
            input_mask = depth_mask[:, :, 1]

            # save crop images and depth, metadata
            frame_id, _ = os.path.splitext(os.path.split(img_file)[1])
            save_img_path = os.path.join(save_path, f'{frame_id}_rgb.jpg')
            save_depth_path = os.path.join(save_path, f'{frame_id}_depth.pfm')
            save_mask_path = os.path.join(save_path, f'{frame_id}_mask.png')

            input_rgb_image.save(save_img_path)
            writePFM(save_depth_path, input_depthmap)
            cv2.imwrite(save_mask_path, (input_mask * 255).astype(np.uint8))

            save_meta_path = os.path.join(save_path, f'{frame_id}_metadata.npz')
            np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,
                     camera_pose=camepose)



