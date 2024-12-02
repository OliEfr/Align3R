import h5py
import numpy as np
import os
import os.path as osp
import cv2
from glob import glob
import PIL.Image
from tqdm import tqdm 
# from utils import *
import sys
import dust3r.datasets.utils.cropping as cropping  # noqa
from utils import *


blender2opencv = np.float32([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

img_size = 512

input_path = "../data/PointOdyssey"
set_list = ["train", "val"]
# set = "test"
for set in set_list:
    data_dir = os.path.join(input_path, set)
    out_dir = os.path.join(input_path + "_proc", set)
    os.makedirs(out_dir, exist_ok=True)

    for sequence in tqdm(sorted(os.listdir(data_dir))):
      if len(sequence.split('.'))==1:
        print(sequence)
        seq_savepath = osp.join(out_dir, sequence)
        os.makedirs(seq_savepath, exist_ok=True)

        imgs_path = osp.join(data_dir, sequence, "rgbs")
        depths_path = osp.join(data_dir, sequence, "depths")
        # masks_path = osp.join(data_dir, sequence, "maps/skymap_left")
        annotations = np.load(osp.join(data_dir, sequence, "anno.npz"))
        trajs_3d = annotations['trajs_3d'].astype(np.float32)
        intrinsics = annotations['intrinsics'].astype(np.float32)
        extrinsics = annotations['extrinsics'].astype(np.float32)


        for rgbfile, depthfile, i in zip(sorted(os.listdir(imgs_path)), sorted(os.listdir(depths_path)), range(len(extrinsics))):

            rgb_filepath = osp.join(imgs_path, rgbfile)
            input_rgb_image = PIL.Image.open(rgb_filepath).convert('RGB')

            depth_filepath = osp.join(depths_path, depthfile)
            depth_16bit = cv2.imread(depth_filepath, cv2.IMREAD_ANYDEPTH)
            input_depth = depth_16bit.astype(np.float32) / 65535.0 * 1000.0

            #assert int(depth_16bit.shape[1] / 2) == input_rgb_image.size[0]
            W = input_rgb_image.size[0]
            #assert int(depth_16bit.shape[0] / 2) == input_rgb_image.size[1]
            H = input_rgb_image.size[1]
            #input_depth_image = cv2.resize(depth_16bit, (W, H), interpolation=cv2.INTER_NEAREST)

            fx = intrinsics[i][0][0]
            fy = intrinsics[i][1][1]
            cx = intrinsics[i][0][2]
            cy = intrinsics[i][1][2]
            camera_intrinsics = np.array(
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]]
            )

            extrinsic_matrix = extrinsics[i].reshape(4, 4).astype(np.float32)@blender2opencv

            # input_disp_image[np.isnan(input_disp_image)] = 1e-3
            # input_disp_image[input_disp_image <= 0] = 1e-3
            # input_depth = fx / input_disp_image
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


            frame_id = rgbfile.split(".")[0][-4:]
            save_img_path = os.path.join(seq_savepath, f'{frame_id}_rgb.jpg')
            save_depth_path = os.path.join(seq_savepath, f'{frame_id}_depth.pfm')
            save_mask_path = os.path.join(seq_savepath, f'{frame_id}_mask.png')

            input_rgb_image.save(save_img_path)
            writePFM(save_depth_path, input_depthmap)
            cv2.imwrite(save_mask_path, (input_mask * 255).astype(np.uint8))

            save_meta_path = os.path.join(seq_savepath, f'{frame_id}_metadata.npz')
            np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,
                     camera_pose=extrinsic_matrix)

