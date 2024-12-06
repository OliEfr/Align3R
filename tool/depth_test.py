import os
import torch
import tempfile
import re
import argparse
import math
import builtins
import datetime
import gradio
import numpy as np
import functools
import trimesh
import copy
import cv2  # noqa
import torchvision.transforms as tvf
from cvxpy import length
import metric
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import PIL.Image
from PIL.ImageOps import exif_transpose
from pyglet.clock import schedule
from scipy.spatial.transform import Rotation
from PIL import Image
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from scipy.optimize import minimize
from dust3r.model import AsymmetricCroCo3DStereo
# from dust3r.demo import get_args_parser, main_demo, set_print_with_timestamp

import matplotlib.pyplot as pl
import matplotlib
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

TAG_FLOAT = 202021.25
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--dust3r_dynamic_model_path", type=str, help="path to the dust3r model weights", default=None)
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--align_with_lstsq", action='store_true', default=False,
                        help="align with lstsq")
    parser.add_argument("--align_with_lad", action='store_true', default=True,
                        help="align with lad")
    parser.add_argument("--align_with_lad2", action='store_true', default=False,
                        help="align with lad2")
    parser.add_argument("--align_with_scale", action='store_true', default=False,
                        help="align with scale")
    parser.add_argument("--eval", action='store_true', default=False,
                        help="eval or not")
    parser.add_argument("--depth_prior", action='store_true', default=False,
                        help="eval the monocular depth maps")
    parser.add_argument("--vis_img", action='store_true', default=False,
                        help="visualize the depth maps")
    parser.add_argument("--vis_img_align", action='store_true', default=False,
                        help="visualize the aligned depth maps")
    parser.add_argument("--depth_max", type=int, default=70,
                        help="depth_max")
    parser.add_argument("--output_postfix", type=str, default=None,
                        help="output archieve")
    parser.add_argument("--dataset_name", type=str, default=None, choices=['bonn', 'tum', 'davis', 'sintel', 'PointOdyssey', 'FlyingThings3D'], help="choose dataset for depth evaluation")
    parser.add_argument("--depth_prior_name", type=str, default='depthpro', choices=['depthpro', 'depthanything'], help="the name of monocular depth estimation model")
    parser.add_argument("--crop", action='store_true', default=False,
                        help="crop or not")
    return parser

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

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def resize_numpy_image(img, long_edge_size):
    """
    Resize the NumPy image to a specified long edge size using OpenCV.
    
    Args:
    img (numpy.ndarray): Input image with shape (H, W, C).
    long_edge_size (int): The size of the long edge after resizing.
    
    Returns:
    numpy.ndarray: The resized image.
    """
    # Get the original dimensions of the image
    h, w = img.shape[:2]
    S = max(h, w)

    # Choose interpolation method
    if S > long_edge_size:
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_CUBIC
    
    # Calculate the new size
    new_size = (int(round(w * long_edge_size / S)), int(round(h * long_edge_size / S)))
    
    # Resize the image
    resized_img = cv2.resize(img, new_size, interpolation=interp)
    
    return resized_img

def crop_center(img, crop_width, crop_height):
    """
    Crop the center of the image.
    
    Args:
    img (numpy.ndarray): Input image with shape (H, W) or (H, W, C).
    crop_width (int): The width of the cropped area.
    crop_height (int): The height of the cropped area.
    
    Returns:
    numpy.ndarray: The cropped image.
    """
    h, w = img.shape[:2]
    cx, cy = h // 2, w // 2
    x1 = max(cx - crop_height // 2, 0)
    x2 = min(cx + crop_height // 2, h)
    y1 = max(cy - crop_width // 2, 0)
    y2 = min(cy + crop_width // 2, w)
    
    cropped_img = img[x1:x2, y1:y2]
    
    return cropped_img

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def depth_read_bonn(filename):
    # loads depth map D from png file
    # and returns it as a numpy array
    depth_png = np.asarray(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth

def pixel_to_pointcloud(depth_map, focal_length_px):
    """
    Convert a depth map to a 3D point cloud.

    Args:
    depth_map (numpy.ndarray): The input depth map with shape (H, W), where each value represents the depth at that pixel.
    focal_length_px (float): The focal length of the camera in pixels.

    Returns:
    numpy.ndarray: The resulting point cloud with shape (H, W, 3), where each point is represented by (X, Y, Z).
    """
    height, width = depth_map.shape
    cx = width / 2
    cy = height / 2

    # Create meshgrid for pixel coordinates
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    # Convert pixel coordinates to camera coordinates
    Z = depth_map
    X = (u - cx) * Z / focal_length_px
    Y = (v - cy) * Z / focal_length_px
    
    # Stack the coordinates into a point cloud (H, W, 3)
    point_cloud = np.dstack((X, Y, Z)).astype(np.float32)
    point_cloud = normalize_pointcloud(point_cloud)
    # Optional: Filter out invalid depth values (if necessary)
    # point_cloud = point_cloud[depth_map > 0]
    #print(point_cloud)
    return point_cloud

def normalize_pointcloud(point_cloud):
    min_vals = np.min(point_cloud, axis=(0, 1))
    max_vals = np.max(point_cloud, axis=(0, 1))
    #print(min_vals, max_vals)
    normalized_point_cloud = (point_cloud - min_vals) / (max_vals - min_vals)
    return normalized_point_cloud

def load_images_my(fileforder, if_depth_prior, size, square_ok=False, verbose=True, crop=True, depth_prior_name='depthanything', dataset_name='sintel'):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
        """
    print("if_crop:", crop)
    root, folder_content = fileforder, sorted(os.listdir(fileforder))
    if dataset_name=='bonn':
      folder_depth = sorted(os.listdir(fileforder.replace('rgb_110','depth_110')))
    elif dataset_name=='tum':
      folder_depth = sorted(os.listdir(fileforder.replace('rgb_50','depth_50')))
    if dataset_name == 'PointOdyssey' or dataset_name == 'FlyingThings3D': 
      supported_images_extensions = ['rgb.jpg', '.jpeg', 'rgb.png']
    else:
      supported_images_extensions = ['.jpg', '.jpeg', '.png']

    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    rgb_imgs = []
    depth_list = []
    depth_prior = []
    i = 0

    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        if i<110:
          img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
          if depth_prior_name == 'depthanything':
            if dataset_name == 'sintel':
              pred_depth = np.load(os.path.join(root, path).replace('final','depth_prediction_depthanything').replace('.png', '.npz'))
              focal_length_px = 200
              pred_depth1 = pred_depth['depth']
            elif dataset_name == 'davis':
              pred_depth = np.load(os.path.join(root, path).replace('JPEGImages','depth_prediction_depthanything').replace('.jpg', '.npz').replace('480p', '1080p'))
              focal_length_px = 200
              pred_depth1 = pred_depth['depth']
            elif dataset_name == 'bonn':
              pred_depth = np.load(os.path.join(root, path).replace('rgb_110','rgb_110_depth_prediction_depthanything').replace('.png', '.npz'))
              focal_length_px = 200
              pred_depth1 = pred_depth['depth']
            elif dataset_name == 'PointOdyssey' or dataset_name == 'FlyingThings3D':
              pred_depth = np.load(os.path.join(root, path).replace('.jpg', '_pred_depth_depthanything.npz').replace('.png', '_pred_depth_depthanything.npz'))
              focal_length_px = 200
              pred_depth1 = pred_depth['depth']
            elif dataset_name == 'tum':
              pred_depth = np.load(os.path.join(root, path).replace('rgb_50','rgb_50_depth_prediction_depthanything').replace('.png', '.npz'))
              focal_length_px = 200
              pred_depth1 = pred_depth['depth']

          elif depth_prior_name == 'depthpro':
            if dataset_name == 'sintel':
              pred_depth = np.load(os.path.join(root, path).replace('final','depth_prediction_depthpro').replace('.png', '.npz'))
              focal_length_px = pred_depth['focallength_px']
              pred_depth1 = pred_depth['depth']
            elif dataset_name == 'davis':
              pred_depth = np.load(os.path.join(root, path).replace('JPEGImages','depth_prediction_depthpro').replace('.jpg', '.npz').replace('480p', '1080p'))
              focal_length_px = pred_depth['focallength_px']
              pred_depth1 = pred_depth['depth']
            elif dataset_name == 'bonn':
              pred_depth = np.load(os.path.join(root, path).replace('rgb_110','rgb_110_depth_prediction_depthpro').replace('.png', '.npz'))
              focal_length_px = pred_depth['focallength_px']
              pred_depth1 = pred_depth['depth']
            elif dataset_name == 'PointOdyssey' or dataset_name == 'FlyingThings3D':
              pred_depth = np.load(os.path.join(root, path).replace('.jpg', '_pred_depth_depthpro.npz').replace('.png', '_pred_depth_depthpro.npz'))
              focal_length_px = pred_depth['focallength_px']
              pred_depth1 = pred_depth['depth']
            elif dataset_name == 'tum':
              pred_depth = np.load(os.path.join(root, path).replace('rgb_50','rgb_50_depth_prediction_depthpro').replace('.png', '.npz'))
              focal_length_px = pred_depth['focallength_px']
              pred_depth1 = pred_depth['depth']

          if not if_depth_prior:
            pred_depth = pixel_to_pointcloud(pred_depth1, focal_length_px)
          else:
            pred_depth = pred_depth1

          if dataset_name == 'sintel':
            depth = depth_read(os.path.join(root, path).replace('MPI-Sintel-training_images', 'MPI-Sintel-depth-training').replace('final/','depth/').replace('.png','.dpt'))
          elif dataset_name == 'davis':
            depth = pred_depth1
          elif dataset_name == 'bonn':
            depth = depth_read_bonn(os.path.join(root.replace('rgb_110','depth_110'), folder_depth[i]))
          elif dataset_name == 'PointOdyssey' or dataset_name == 'FlyingThings3D':
            depth = readPFM(os.path.join(root, path).replace('_rgb.jpg', '_depth.pfm'))
          elif dataset_name == 'tum':
            depth = depth_read_bonn(os.path.join(root.replace('rgb_50','depth_50'), folder_depth[i]))

          W1, H1 = img.size
          if size == 224:
              # resize short side to 224 (then crop)
              img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
              pred_depth = resize_numpy_image(pred_depth, round(size * max(W1 / H1, H1 / W1)))
              pred_depth1 = resize_numpy_image(pred_depth1, round(size * max(W1 / H1, H1 / W1)))
          else:
              # resize long side to 512
              img = _resize_pil_image(img, size)
              pred_depth = resize_numpy_image(pred_depth, size)
              pred_depth1 = resize_numpy_image(pred_depth1, size)
          W, H = img.size
          cx, cy = W // 2, H // 2
          if size == 224:
              half = min(cx, cy)
              img = img.crop((cx - half, cy - half, cx + half, cy + half))
              pred_depth = crop_center(pred_depth, 2 * half, 2 * half)   
              pred_depth1 = crop_center(pred_depth1, 2 * half, 2 * half) 
          else:
              halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
              if not (square_ok) and W == H:
                  halfh = 3 * halfw / 4
              if crop:
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
                pred_depth = crop_center(pred_depth, 2 * halfw, 2 * halfh)
                pred_depth1 = crop_center(pred_depth1, 2 * halfw, 2 * halfh)
              else:
                img = img.resize((2*halfw, 2*halfh), PIL.Image.LANCZOS)
                pred_depth = cv2.resize(pred_depth, (2*halfw, 2*halfh), interpolation=cv2.INTER_CUBIC)
                pred_depth1 = cv2.resize(pred_depth1, (2*halfw, 2*halfh), interpolation=cv2.INTER_CUBIC)

          W2, H2 = img.size
          if verbose:
              print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')

          rgb_imgs.append(img)
          depth_list.append(depth)
          depth_prior.append(pred_depth1)
          imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
              [img.size[::-1]]), idx=len(imgs), pred_depth=pred_depth[None], instance=str(len(imgs))))
          i+=1

    assert imgs, 'no images foud at ' + root
    if verbose:
        print(f' (Found {len(imgs)} images)')

    return imgs, rgb_imgs, depth_list, depth_prior

def find_images(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def my_make_pairs(imgs, clip_size):

    keyframes_id = []
    coarse_init_clip = []
    clips = []

    for i in range(0, len(imgs), clip_size):
        coarse_init_clip.append(imgs[i].copy())
        keyframes_id.append(i)
        clips.append(imgs[i:i + clip_size])

    # coarse_init_clip
    coarse_init_pairs = []
    for index in range(0, len(coarse_init_clip)):
        coarse_init_clip[index]['idx'] = index
    for i in range(len(coarse_init_clip) - 1):
        for j in range(i+1, len(coarse_init_clip)):
            coarse_init_pairs.append((coarse_init_clip[i], coarse_init_clip[j]))
    # coarse_init_pairs += [(img2, img1) for img1, img2 in coarse_init_pairs]


    # clips
    all_clips_id = []
    for clip in clips:
        each_clip_id = []
        for index in range(0, len(clip)):
            each_clip_id.append(clip[index]['idx'])
            clip[index]['idx'] = index
        all_clips_id.append(each_clip_id)


    all_clips_pairs = []
    for clip in clips:
        each_clip_pairs = []
        for i in range(len(clip)-1):
            for j in range(i+1, len(clip)):
                each_clip_pairs.append((clip[i].copy(), clip[j].copy()))
        # each_clip_pairs += [(img2, img1) for img1, img2 in each_clip_pairs]
        all_clips_pairs.append(each_clip_pairs)

    return coarse_init_pairs, keyframes_id, all_clips_pairs, all_clips_id

def resize_depth(depth_list, size, square_ok=False):
    W1, H1 = depth_list[0].shape
    processed_depth_list = []
    max_depth = 0
    for depth in depth_list:
       max_depth = max(max_depth, depth.max())
    for depth in depth_list:
        depth = depth/max_depth
        if size == 224:
            depth = resize_numpy_image(depth, round(size * max(W1 / H1, H1 / W1)))
        else:
            depth = resize_numpy_image(depth, size)
        
        W, H = depth.shape[:2]
        cx, cy = W // 2, H // 2
        
        if size == 224:
            half = min(cx, cy)
            depth = crop_center(depth, 2 * half, 2 * half)
        else:
            halfw = ((2 * cx) // 16) * 8
            halfh = ((2 * cy) // 16) * 8
            if not square_ok and W == H:
                halfh = 3 * halfw // 4
            depth = torch.tensor(crop_center(depth, 2 * halfw, 2 * halfh)).cuda()
        
        processed_depth_list.append(depth)
    return processed_depth_list
    
def get_bottom_level_directories(root_dir):
    bottom_level_dirs = []
    
    # Walk through the directory
    for current_dir, sub_dirs, files in os.walk(root_dir):
        # Check if there are no subdirectories
        if not sub_dirs:
            bottom_level_dirs.append(current_dir)
    
    return bottom_level_dirs
def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params

    predicted_aligned = s * predicted_depth + t

    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)

def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)
    
    initial_params = [s, t]  # s = 1, t = 0
    
    result = minimize(absolute_error_loss, initial_params, args=(predicted_depth_np, ground_truth_depth_np))
    
    s, t = result.x  
    return s, t

def absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=1.0, t_init=0.0, lr=1e-4, max_iters=1000, tol=1e-6):
    # Initialize s and t as torch tensors with requires_grad=True
    s = torch.tensor([s_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    t = torch.tensor([t_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)

    optimizer = torch.optim.Adam([s, t], lr=lr)
    
    prev_loss = None

    for i in range(max_iters):
        optimizer.zero_grad()

        # Compute predicted aligned depth
        predicted_aligned = s * predicted_depth + t

        # Compute absolute error
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)

        # Compute loss
        loss = torch.sum(abs_error)

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Check convergence
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss.item()

    return s.detach().item(), t.detach().item()

if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    # setup
    schedule = "cosine"
    scenegraph_type = "my_swin"
    as_pointcloud = True
    mask_sky = True
    clean_depth = True

    # hyper parameter
    clip_size = 20                  # clip size
    complete_graph = False
    min_conf_thr = 3.0
    if_use_mono = False              # to adjust whether use the monodepth as initialization
    niter = 300  # iteration number for global alignment

    image_size = 512
    silent = args.silent
    dataset_name = args.dataset_name
    # load the image sequences or video
    if dataset_name == 'bonn':
      path = './data/bonn/rgbd_bonn_dataset/'
      seq_list = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
      directories = [path+'rgbd_bonn_'+i for i in seq_list]
    elif dataset_name == 'tum':
      path = './data/tum/'
      entries = os.listdir(path)
      directories = [os.path.join(path, entry) for entry in entries if os.path.isdir(os.path.join(path, entry))]
    elif dataset_name == 'davis':
      path = './data/davis/DAVIS/JPEGImages/480p/'
      entries = os.listdir(path)
      directories = [os.path.join(path, entry) for entry in entries if os.path.isdir(os.path.join(path, entry))]
      directories = sorted(get_bottom_level_directories(path))
    elif dataset_name == 'sintel':
      path = './data/MPI-Sintel/MPI-Sintel-training_images/training/final/'
      entries = os.listdir(path)
      directories = [os.path.join(path, entry) for entry in entries if os.path.isdir(os.path.join(path, entry))]
      directories = sorted(get_bottom_level_directories(path))
    elif dataset_name == 'PointOdyssey':
      path = './data/PointOdyssey_proc/val/'
      entries = os.listdir(path)
      directories = [os.path.join(path, entry) for entry in entries if os.path.isdir(os.path.join(path, entry))]
      directories = sorted(get_bottom_level_directories(path))
    elif dataset_name == 'FlyingThings3D':
      path = './data/SceneFlow/FlyingThings3D_proc/TEST/'
      entries = os.listdir(path)
      directories = [os.path.join(path, entry) for entry in entries if os.path.isdir(os.path.join(path, entry))]
      directories = sorted(get_bottom_level_directories(path))[::20]
    lr = 1e-4
    max_iters = 1000
    align_with_lstsq = args.align_with_lstsq
    align_with_lad = args.align_with_lad
    align_with_lad2 = args.align_with_lad2
    align_with_scale = args.align_with_scale
    gathered_depth_metrics = []
    for folder in tqdm(directories):
        seq = folder.split('/')[-1]
        print(seq)
        if dataset_name == 'bonn':
          folder = os.path.join(folder, 'rgb_110')
        elif dataset_name == 'tum':
          folder = os.path.join(folder, 'rgb_50')
        imgs, imgs_rgb, depth_list, depth_prior = load_images_my(folder, args.depth_prior, size=image_size, verbose=not silent, crop=args.crop, depth_prior_name=args.depth_prior_name, dataset_name=dataset_name)
        root, folder_content = folder, find_images(folder)
        
        while len(imgs) % clip_size == 1 or len(imgs) % clip_size == 0 or clip_size>len(imgs):
          clip_size -= 1
        if not args.depth_prior:
          coarse_init_pairs, keyframes_id, all_clips_pairs, all_clips_id = my_make_pairs(imgs, clip_size)
          mono_depths_torch = []
          # load the DUSt3R model
          dust3r_dynamic_weights_path = args.dust3r_dynamic_model_path
          dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_dynamic_weights_path).to(args.device)
          # infer key frames
          key_output = inference(coarse_init_pairs, dust3r_model, args.device, batch_size=1, verbose=not silent)
          key_output['pred1']['conf'][key_output['pred1']['conf']>1]=10
          key_output['pred2']['conf'][key_output['pred2']['conf']>1]=10
          # global optimization (key frames)
          mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
          scene = global_aligner(key_output, if_use_mono, mono_depths_torch, device=args.device, mode=mode, verbose=not silent,min_conf_thr=min_conf_thr)
          lr = 0.05

          if mode == GlobalAlignerMode.PointCloudOptimizer:
              loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

          init_keyposes = to_numpy(scene.get_im_poses()).tolist()
          init_keydepths = to_numpy(scene.get_depthmaps())
          init_keyfocals = to_numpy(scene.get_focals()).tolist()

          depth_imgs = []
          depth_gray = []
          # clip-wise inference and optimization
          for init_keypose, init_keydepth, init_keyfocal, clip_pair in zip(init_keyposes, init_keydepths, init_keyfocals, all_clips_pairs):
              clipwise_output = inference(clip_pair, dust3r_model, args.device, batch_size=1, verbose=not silent)
              if len(clipwise_output)>0:
                clipwise_output['pred1']['conf'][clipwise_output['pred1']['conf']>1]=10
                clipwise_output['pred2']['conf'][clipwise_output['pred2']['conf']>1]=10
                mode_clip = GlobalAlignerMode.PointCloudOptimizer
                scene_clip_i = global_aligner(clipwise_output, if_use_mono, mono_depths_torch, device=args.device, mode=mode_clip, verbose=not silent,min_conf_thr=min_conf_thr)

                lr = 0.05

                init_priors = [init_keypose, init_keydepth, init_keyfocal]

                loss = scene_clip_i.compute_global_alignment(init='mst', init_priors = init_priors, niter=niter, schedule=schedule, lr=lr)

                depths = to_numpy(scene_clip_i.get_depthmaps())
                
                for i in range(len(depths)):
                    depth_map_colored = cv2.applyColorMap((depths[i] * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    depth_imgs.append(depth_map_colored)
                    depth_gray.append(depths[i])

        if args.vis_img:
          if not os.path.exists(args.output_postfix+'/'+seq):
            os.makedirs(args.output_postfix+'/'+seq)
          images = []
          for i in range(len(depth_imgs)):
            cv2.imwrite(args.output_postfix+'/'+seq+f'/{(i):04d}.png', depth_imgs[i])
            images.append(Image.open(args.output_postfix+'/'+seq+f'/{(i):04d}.png'))
          images[0].save(args.output_postfix+'/'+seq+'_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
        
        if args.eval or args.vis_img_align:
          depth_gt = [i[None,...] for i in depth_list]
          depth_gt = np.concatenate(depth_gt)
          depth_pred = [cv2.resize(i, (depth_list[0].shape[1], depth_list[0].shape[0]))[None,...] for i in depth_gray]if not args.depth_prior else [cv2.resize(i, (depth_list[0].shape[1], depth_list[0].shape[0]))[None,...] for i in depth_prior]
          depth_pred = np.concatenate(depth_pred)

          # clip the depth
          valid_mask = np.logical_and(
            (depth_gt > 1e-3), 
            (depth_gt < args.depth_max)
              )
          num_valid_pixels = torch.sum(torch.tensor(valid_mask)).item()
          pred_depth_i = depth_pred[valid_mask].reshape((-1, 1))
          gt_depth = depth_gt[valid_mask].reshape((-1, 1))

          # calc scale and shift
          pred_depth_i = torch.tensor(pred_depth_i)
          gt_depth = torch.tensor(gt_depth)
          if align_with_lstsq:
              predicted_depth_np = pred_depth_i.cpu().numpy().reshape(-1, 1)
              ground_truth_depth_np = gt_depth.cpu().numpy().reshape(-1, 1)
              
              # Add a column of ones for the shift term
              A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
              
              # Solve for scale (s) and shift (t) using least squares
              result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
              s, t = result[0][0], result[0][1]

              # convert to torch tensor
              s = torch.tensor(s, device=pred_depth_i.device)
              t = torch.tensor(t, device=pred_depth_i.device)
              
              # Apply scale and shift
              aligned_pred = s * pred_depth_i + t

          elif align_with_lad:
              s, t = absolute_value_scaling(pred_depth_i, gt_depth, s=torch.median(gt_depth) / torch.median(pred_depth_i))
              aligned_pred = s * pred_depth_i + t
          elif align_with_lad2:
              s_init = (torch.median(gt_depth) / torch.median(pred_depth_i)).item()
              s, t = absolute_value_scaling2(pred_depth_i, gt_depth, s_init=s_init, lr=lr, max_iters=max_iters)
              aligned_pred = s * pred_depth_i + t
          elif align_with_scale:
              # Compute initial scale factor 's' using the closed-form solution (L2 norm)
              dot_pred_gt = torch.nanmean(gt_depth)
              dot_pred_pred = torch.nanmean(pred_depth_i)
              s = dot_pred_gt / dot_pred_pred

              # Iterative reweighted least squares using the Weiszfeld method
              for _ in range(10):
                  # Compute residuals between scaled predictions and ground truth
                  residuals = s * pred_depth_i - gt_depth
                  abs_residuals = residuals.abs() + 1e-8  # Add small constant to avoid division by zero
                  
                  # Compute weights inversely proportional to the residuals
                  weights = 1.0 / abs_residuals
                  
                  # Update 's' using weighted sums
                  weighted_dot_pred_gt = torch.sum(weights * pred_depth_i * gt_depth)
                  weighted_dot_pred_pred = torch.sum(weights * pred_depth_i ** 2)
                  s = weighted_dot_pred_gt / weighted_dot_pred_pred

              # Optionally clip 's' to prevent extreme scaling
              s = s.clamp(min=1e-3)
              
              # Detach 's' if you want to stop gradients from flowing through it
              s = s.detach()
              
              # Apply the scale factor to the predicted depth
              aligned_pred = s * pred_depth_i

          else:
              # Align the predicted depth with the ground truth using median scaling
              scale_factor = torch.median(gt_depth) / torch.median(pred_depth_i)
              aligned_pred *= scale_factor
          # clip the aligned depth
          aligned_pred = np.clip(aligned_pred, a_min=1e-5, a_max=args.depth_max) 
          
          if args.vis_img_align:
            norm_scale = 20
            mask = (depth_gt==0) | (depth_gt>100)
           
            depth_pred_align = depth_pred * s + t
            gt_depth_norm = depth_gt/norm_scale
            aligned_pred_norm = depth_pred_align/norm_scale
            if not os.path.exists(args.output_postfix+'/'+seq):
              os.makedirs(args.output_postfix+'/'+seq)
            images = []
            images_rgb = []
            images_gt = []
            for i in range(aligned_pred_norm.shape[0]):
              aligned_pred_norm_colored = cv2.applyColorMap((aligned_pred_norm[i] * 255).astype(np.uint8), cv2.COLORMAP_JET)
              aligned_pred_norm_colored[mask[i]]=np.array([255,255,255])

              gt_depth_norm_colored = cv2.applyColorMap((gt_depth_norm[i] * 255).astype(np.uint8), cv2.COLORMAP_JET)
              gt_depth_norm_colored[mask[i]]=np.array([255,255,255])
              cv2.imwrite(args.output_postfix+'/'+seq+f'/{(i):04d}.png', aligned_pred_norm_colored)
              cv2.imwrite(args.output_postfix+'/'+seq+f'/{(i):04d}_rgb.png', np.array(imgs_rgb[i])[...,::-1])
              cv2.imwrite(args.output_postfix+'/'+seq+f'/{(i):04d}_gt.png', gt_depth_norm_colored)
              #cv2.imwrite(args.output_postfix+'/'+seq+f'/{(i):04d}_depth_prior.png', depth_prior_gray[i])
              images.append(Image.open(args.output_postfix+'/'+seq+f'/{(i):04d}.png'))
              images_rgb.append(Image.open(args.output_postfix+'/'+seq+f'/{(i):04d}_rgb.png'))
              images_gt.append(Image.open(args.output_postfix+'/'+seq+f'/{(i):04d}_gt.png'))
            images[0].save(args.output_postfix+'/'+seq+'_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
            images_gt[0].save(args.output_postfix+'/'+seq+'_gt_depth_maps.gif', save_all=True, append_images=images_gt[1:], duration=100, loop=0)
            images_rgb[0].save(args.output_postfix+'/'+seq+'_rgb_depth_maps.gif', save_all=True, append_images=images_rgb[1:], duration=100, loop=0)

          # Calculate the metrics
          if args.eval:
            abs_rel = torch.mean(torch.abs(aligned_pred - gt_depth) / gt_depth).item()
            sq_rel = torch.mean(((aligned_pred - gt_depth) ** 2) / gt_depth).item()
            
            # Correct RMSE calculation
            rmse = torch.sqrt(torch.mean((aligned_pred - gt_depth) ** 2)).item()
            
            # Clip the depth values to avoid log(0)
            #aligned_pred = torch.clamp(aligned_pred, min=1e-5)
            log_rmse = torch.sqrt(torch.mean((torch.log(aligned_pred) - torch.log(gt_depth)) ** 2)).item()
            
            # Calculate the accuracy thresholds
            max_ratio = torch.maximum(aligned_pred / gt_depth, gt_depth / aligned_pred)
            threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
            threshold_2 = torch.mean((max_ratio < 1.25 ** 2).float()).item()
            threshold_3 = torch.mean((max_ratio < 1.25 ** 3).float()).item()

            results = {
            'Abs Rel': abs_rel,
            'Sq Rel': sq_rel,
            'RMSE': rmse,
            'Log RMSE': log_rmse,
            'δ < 1.25': threshold_1,
            'δ < 1.25^2': threshold_2,
            'δ < 1.25^3': threshold_3,
            'valid_pixels': num_valid_pixels
            }
            gathered_depth_metrics.append(results)
            print(results)

    if args.eval:
      average_metrics = {
          key: np.average(
              [metrics[key] for metrics in gathered_depth_metrics], 
              weights=[metrics['valid_pixels'] for metrics in gathered_depth_metrics]
          )
          for key in gathered_depth_metrics[0].keys() if key != 'valid_pixels'
      }
      print('Average depth evaluation metrics:', average_metrics)




