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

import dust3r.datasets.utils.cropping as cropping  # noqa
from scipy.spatial.transform import Rotation

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="../data/vkitti_2.0.3_proc")
    parser.add_argument("--data_dir", type=str, default="../data/vkitti_2.0.3")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--img_size", type=int, default=512,
                        help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
    return parser


def read_extri_vkitti(cam_file):
    poses_left = []
    poses_right = []
    with open(cam_file) as f:
        for line in f:
            line_list = line.split(' ')
            if '0' == line_list[1]:
                pose = np.array(line_list[2:])
                pose = pose.reshape(4, 4).astype(np.float32)
                poses_left.append(pose)
            if '1' == line_list[1]:
                pose = np.array(line_list[2:])
                pose = pose.reshape(4, 4).astype(np.float32)
                poses_right.append(pose)

    return poses_left, poses_right


def read_intri_vkitti(cam_file):
    intri_left = []
    intri_right = []
    with open(cam_file) as f:
        for line in f:
            line_list = line.split(' ')
            if '0' == line_list[1]:
                intri = np.array(line_list[2:]).astype(np.float32)
                intri_left.append(intri)
            if '1' == line_list[1]:
                intri = np.array(line_list[2:]).astype(np.float32)
                intri_right.append(intri)

    return intri_left, intri_right


def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
           os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    assert args.data_dir != args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = sorted(glob(osp.join(args.data_dir, 'vkitti_2.0.3_rgb', '*/*/*/*')))
    depth_paths = [im.replace('rgb', 'depth') for im in image_paths]

    camera_paths = sorted(glob(osp.join(args.data_dir, 'vkitti_2.0.3_textgt', '*/*')))

    cameraintri_paths = [a + "/intrinsic.txt" for a in camera_paths]
    cameraextri_paths = [a + "/extrinsic.txt" for a in camera_paths]

    img_size = args.img_size

    scene_label = 0

    for image_path, depth_path, cameraintri_path, cameraextri_path in zip(image_paths, depth_paths, cameraintri_paths, cameraextri_paths):

        for view in ["Camera_0", "Camera_1"]:
            image_path_view = os.path.join(image_path, view)
            depth_path_view = os.path.join(depth_path, view)

            # each sequence
            img_files = sorted([os.path.join(image_path_view, file) for file in os.listdir(image_path_view)])
            depth_files = sorted([os.path.join(depth_path_view, file) for file in os.listdir(depth_path_view)])

            camposes = []
            camintris = []
            if "Camera_0" == view:
                camposes, _ = read_extri_vkitti(cameraextri_path)
                camintris, _ = read_intri_vkitti(cameraintri_path)
            else:
                _, camposes = read_extri_vkitti(cameraextri_path)
                _, camintris = read_intri_vkitti(cameraintri_path)

            # save path
            view_label = "left" if "Camera_0" == view else "right"
            save_path = os.path.join(args.output_dir, f'scene{scene_label:0>4d}_{view_label}')
            os.makedirs(save_path, exist_ok=True)

            for img_file, depth_file, camepose,  intrisics in zip(img_files, depth_files, camposes, camintris):
                input_rgb_image = PIL.Image.open(img_file).convert('RGB')

                input_depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 100

                mask = ((input_depth > 0) * (input_depth < 600))

                fx = intrisics[0]
                fy = intrisics[1]
                cx = intrisics[2]
                cy = intrisics[3]

                camera_intrinsics = np.array(
                    [[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]]
                )

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
                frame_id = frame_id[4:]
                save_img_path = os.path.join(save_path, f'{frame_id}_rgb.jpg')
                save_depth_path = os.path.join(save_path, f'{frame_id}_depth.pfm')
                save_mask_path = os.path.join(save_path, f'{frame_id}_mask.png')

                input_rgb_image.save(save_img_path)
                writePFM(save_depth_path, input_depthmap)
                cv2.imwrite(save_mask_path, (input_mask * 255).astype(np.uint8))

                save_meta_path = os.path.join(save_path, f'{frame_id}_metadata.npz')
                np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,
                         camera_pose=np.linalg.inv(camepose))

        scene_label += 1

