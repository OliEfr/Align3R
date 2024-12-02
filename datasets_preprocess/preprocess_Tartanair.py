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
import transformation as tf

def ned2cam(traj):
    '''
    transfer a ned traj to camera frame traj
    '''
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = tf.pos_quats2SE_matrices(np.array(traj))

    for tt in traj_ses:
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(tf.SE2pos_quat(ttt))
        
    return np.array(new_traj)
def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    # 计算四元数对应的旋转矩阵
    q = np.array([qw, qx, qy, qz])
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(3)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]
    ])
    return rot_matrix

def pose_to_extrinsic_matrix(pose):
    # 将位姿信息转换为外参矩阵
    tx, ty, tz, qx, qy, qz, qw = pose
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = [tx, ty, tz]
    return extrinsic_matrix

blender2opencv = np.float32([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

img_size = 512

input_path = "../data/Tartanair"


data_dir = input_path
out_dir = input_path + "_proc"
os.makedirs(out_dir, exist_ok=True)
fx = 320.0  # focal length x
fy = 320.0  # focal length y
cx = 320.0  # optical center x
cy = 240.0  # optical center y

fov = 90 # field of view

width = 640
height = 480
for sequence in tqdm(sorted(os.listdir(data_dir))[2:]):
    
    if os.path.isdir(osp.join(data_dir, sequence)):
      for sequence1 in tqdm(sorted(os.listdir(osp.join(data_dir, sequence)))):
        if sequence1=='Easy':
          for sequence2 in tqdm(sorted(os.listdir(osp.join(data_dir, sequence, sequence1)))):
            for cam in ['left', 'right']:
              seq_savepath = osp.join(out_dir, sequence+'_'+sequence1+'_'+sequence2+'_'+cam)
              print(seq_savepath)
              os.makedirs(seq_savepath, exist_ok=True)

              imgs_path = osp.join(data_dir, sequence, sequence1, sequence2, "image_"+cam)
              depths_path = osp.join(data_dir, sequence, sequence1, sequence2, "depth_"+cam)
              # masks_path = osp.join(data_dir, sequence, "maps/skymap_left")

              #intrinsics = annotations['intrinsics'].astype(np.float32)
              extrinsics = np.loadtxt(osp.join(data_dir, sequence, sequence1, sequence2, "pose_"+cam+".txt"))
              extrinsics = ned2cam(extrinsics)


              for rgbfile, depthfile, i in zip(sorted(os.listdir(imgs_path)), sorted(os.listdir(depths_path)), range(len(extrinsics))):

                  rgb_filepath = osp.join(imgs_path, rgbfile)
                  input_rgb_image = PIL.Image.open(rgb_filepath).convert('RGB')

                  depth_filepath = osp.join(depths_path, depthfile)
                  input_depth = np.load(depth_filepath)

                  #assert int(depth_16bit.shape[1] / 2) == input_rgb_image.size[0]
                  W = input_rgb_image.size[0]
                  #assert int(depth_16bit.shape[0] / 2) == input_rgb_image.size[1]
                  H = input_rgb_image.size[1]
                  #input_depth_image = cv2.resize(depth_16bit, (W, H), interpolation=cv2.INTER_NEAREST)

                  # fx = intrinsics[i][0][0]
                  # fy = intrinsics[i][1][1]
                  # cx = intrinsics[i][0][2]
                  # cy = intrinsics[i][1][2]
                  camera_intrinsics = np.array(
                      [[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]]
                  )

                  extrinsic_matrix = pose_to_extrinsic_matrix(extrinsics[i])

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
                  input_rgb_image, depth_mask, depth_mask, input_camera_intrinsics = cropping.crop_image_depthmap(
                      input_rgb_image, depth_mask, depth_mask, camera_intrinsics, crop_bbox)

                  # try to set the lower dimension to img_size * 3/4 -> img_size=512 => 384
                  scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
                  output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
                  if max(output_resolution) < img_size:
                      # let's put the max dimension to img_size
                      scale_final = (img_size / max(H, W)) + 1e-8
                      output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)

                  input_rgb_image, depth_mask, depth_mask, input_camera_intrinsics = cropping.rescale_image_depthmap(
                      input_rgb_image, depth_mask, depth_mask, input_camera_intrinsics, output_resolution)

                  input_depthmap = depth_mask[:, :, 0]
                  input_mask = depth_mask[:, :, 1]


                  frame_id = rgbfile.split(".")[0][0:6]
                  save_img_path = os.path.join(seq_savepath, f'{frame_id}_rgb.jpg')
                  save_depth_path = os.path.join(seq_savepath, f'{frame_id}_depth.pfm')
                  save_mask_path = os.path.join(seq_savepath, f'{frame_id}_mask.png')

                  input_rgb_image.save(save_img_path)
                  writePFM(save_depth_path, input_depthmap)
                  cv2.imwrite(save_mask_path, (input_mask * 255).astype(np.uint8))

                  save_meta_path = os.path.join(seq_savepath, f'{frame_id}_metadata.npz')
                  np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,
                            camera_pose=extrinsic_matrix)

