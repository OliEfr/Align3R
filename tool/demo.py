# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
import math
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import copy
from tqdm import tqdm
import cv2
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image_pose import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.utils.vo_eval import save_trajectory_tum_format
from dust3r.cloud_opt_flow import global_aligner, GlobalAlignerMode
import matplotlib.pyplot as pl
from transformers import pipeline
import depth_pro
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights", default='align3r_depthpro.pth')
    parser.add_argument("--model_name", type=str, default='cyun9286/Align3R_DepthPro_ViTLarge_BaseDecoder_512_dpt', help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--input_dir", type=str, default='', help="Path to input images directory")
    parser.add_argument("--seq_name", type=str, default='bear', help="Sequence name for evaluation")
    parser.add_argument("--depth_prior_name", type=str, default='depthpro', choices=['depthpro', 'depthanything'], help="the name of monocular depth estimation model")
    
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False, help='Use ground truth masks for DAVIS')
    parser.add_argument('--fps', type=int, default=0, help='FPS for video processing')
    parser.add_argument('--interval', type=int, default=30, help='Maximum number of frames for video processing')
    
    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False, help="Share the demo")
    parser.add_argument("--mode", type=str, default='eval_pose_h', choices=['eval_pose', 'eval_pose_h'], help="eval pose hierarchically or not")
    return parser

def video_to_images(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    frame_filename_list = []
    while True:
        # Read one frame from the video
        ret, frame = cap.read()
        
        # If the frame is successfully read
        if not ret:
            break
        
        # Save the current frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_filename_list.append(frame_filename)
        # Print progress
        print(f"Saving frame {frame_count}")
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames. Images saved to {output_folder}")
    return frame_filename_list

def generate_monocular_depth_maps(img_list, depth_prior_name):
    if depth_prior_name=='depthpro':
        model, transform = depth_pro.create_model_and_transforms(device='cuda')
        model.eval()
        for image_path in tqdm(img_list):
          path_depthpro = image_path.replace('.png','_pred_depth_depthpro.npz').replace('.jpg','_pred_depth_depthpro.npz')
          image, _, f_px = depth_pro.load_rgb(image_path)
          image = transform(image)
          # Run inference.
          prediction = model.infer(image, f_px=f_px)
          depth = prediction["depth"].cpu()  # Depth in [m].
          np.savez_compressed(path_depthpro, depth=depth, focallength_px=prediction["focallength_px"].cpu())  
    elif depth_prior_name=='depthanything':
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",device='cuda')
        for image_path in tqdm(img_list):
          path_depthanything = image_path.replace('.png','_pred_depth_depthanything.npz').replace('.jpg','_pred_depth_depthanything.npz')
          image = Image.open(image_path)
          depth = pipe(image)["predicted_depth"].numpy()
          np.savez_compressed(path_depthanything, depth=depth)  

def get_reconstructed_scene(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask, fps, interval, depth_prior_name):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()

    generate_monocular_depth_maps(filelist, depth_prior_name)

    imgs = load_images(filelist, size=image_size, verbose=not silent, fps=fps, start=0, interval=interval, traj_format='custom', depth_prior_name=depth_prior_name)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer  
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                               num_total_iter=niter, empty_cache= len(filelist) > 72)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)

    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder, 0)
    dynamic_masks = scene.save_dynamic_masks(save_folder, 0)
    conf = scene.save_conf_maps(save_folder, 0)
    init_conf = scene.save_init_conf_maps(save_folder, 0)
    rgbs = scene.save_rgb_imgs(save_folder, 0)
    enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 

    return scene

def get_reconstructed_scene_hierachical(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask, fps, interval, depth_prior_name):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()

    generate_monocular_depth_maps(filelist, depth_prior_name)

    imgs = load_images(filelist, size=image_size, verbose=not silent, fps=fps, start=0, interval=interval, traj_format='custom', depth_prior_name=depth_prior_name)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)
    clip_size = 20

    while len(imgs) % clip_size == 1 or len(imgs) % clip_size == 0 or clip_size>len(imgs):
        clip_size -= 1
    coarse_init_pairs, keyframes_id, all_clips_pairs, all_clips_id = my_make_pairs(imgs, clip_size)
    key_output = inference(coarse_init_pairs, model, device, batch_size=1, verbose=not silent)
  
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer  
        scene = global_aligner(key_output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                               num_total_iter=niter, empty_cache= len(filelist) > 72)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(key_output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    init_keyposes = to_numpy(scene.get_im_poses()).tolist()
    init_keydepths = to_numpy(scene.get_depthmaps())
    init_keyfocals = to_numpy(scene.get_focals()).tolist()
    offset = 0
    pred_traj_all = [np.zeros((0,7)), np.zeros((0,))]
    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)
    for init_keypose, init_keydepth, init_keyfocal, clip_pair in zip(init_keyposes, init_keydepths, init_keyfocals, all_clips_pairs):
        clipwise_output = inference(clip_pair, model, device, batch_size=1, verbose=not silent)
        mode_clip = GlobalAlignerMode.PointCloudOptimizer
        scene_clip = global_aligner(clipwise_output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                               num_total_iter=niter, empty_cache= len(filelist) > 72)
        lr = 0.01
        init_priors = [init_keypose, init_keydepth, init_keyfocal]
        # print('pose', init_keypose)
        loss = scene_clip.compute_global_alignment(
            init='mst', init_priors=init_priors, niter=niter, schedule=schedule, lr=lr,
        )
        pred_traj = scene_clip.get_tum_poses()
        pred_traj_all[0] = np.concatenate([pred_traj_all[0], pred_traj[0]], axis=0)
        pred_traj_all[1] = np.concatenate([pred_traj_all[1], pred_traj[1] + offset], axis=0)
        save_trajectory_tum_format(pred_traj_all, f'{save_folder}/pred_traj.txt')
        K = scene_clip.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
        depth_maps = scene_clip.save_depth_maps(save_folder, offset)
        dynamic_masks = scene_clip.save_dynamic_masks(save_folder, offset)
        conf = scene_clip.save_conf_maps(save_folder, offset)
        init_conf = scene_clip.save_init_conf_maps(save_folder, offset)
        rgbs = scene_clip.save_rgb_imgs(save_folder, offset)
        offset += pred_traj[0].shape[0]

    enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 
    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    return scene

def group_by_clip_size(keyframes_id, init_keyposes, init_keydepths, init_keyfocals, clip_size):
    grouped_poses = {}
    grouped_depths = {}
    grouped_focals = {}
    grouped_idx = {}
    for idx, kf_id in enumerate(keyframes_id):
        group_idx = kf_id // clip_size
        offset_idx = kf_id % clip_size  
        if group_idx not in grouped_poses:
            grouped_poses[group_idx] = []
            grouped_depths[group_idx] = []
            grouped_focals[group_idx] = []
            grouped_idx[group_idx] = []
        grouped_idx[group_idx].append(offset_idx)
        grouped_poses[group_idx].append(init_keyposes[idx])
        grouped_depths[group_idx].append(init_keydepths[idx])
        grouped_focals[group_idx].append(init_keyfocals[idx])
    
    return grouped_poses, grouped_depths, grouped_focals, grouped_idx

def get_reconstructed_scene_hierachical1(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask, fps, interval, depth_prior_name):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()

    generate_monocular_depth_maps(filelist, depth_prior_name)

    imgs = load_images(filelist, size=image_size, verbose=not silent, fps=fps, start=0, interval=interval, traj_format='custom', depth_prior_name=depth_prior_name)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)
    clip_size = 20

    while len(imgs) % clip_size == 1 or len(imgs) % clip_size == 0 or clip_size>len(imgs):
        clip_size -= 1
    coarse_init_pairs, keyframes_id, all_clips_pairs, all_clips_id = my_make_pairs2(imgs, clip_size)
    key_output = inference(coarse_init_pairs, model, device, batch_size=1, verbose=not silent)
  
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer  
        scene = global_aligner(key_output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                               num_total_iter=niter, empty_cache= len(filelist) > 72)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(key_output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        
    init_keyposes = to_numpy(scene.get_im_poses()).tolist()
    init_keypts3d = to_numpy(scene.get_pts3d())
    init_keyfocals = to_numpy(scene.get_focals()).tolist()

    grouped_poses, grouped_pts3d, grouped_focals, grouped_idx = group_by_clip_size(keyframes_id, init_keyposes, init_keypts3d, init_keyfocals, clip_size)
    offset = 0
    pred_traj_all = [np.zeros((0,7)), np.zeros((0,))]
    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)
    for i, clip_pair in enumerate(all_clips_pairs):
        init_keypose, init_pts3d, init_keyfocal, init_idx = grouped_poses[i], grouped_pts3d[i], grouped_focals[i], grouped_idx[i]

        clipwise_output = inference(clip_pair, model, device, batch_size=1, verbose=not silent)
        mode_clip = GlobalAlignerMode.PointCloudOptimizer
        scene_clip = global_aligner(clipwise_output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                               num_total_iter=niter, empty_cache= len(filelist) > 72)
        lr = 0.01
        init_priors = [init_keypose, init_idx, init_keyfocal, init_pts3d]
        # print('pose', init_keypose)
        loss = scene_clip.compute_global_alignment(
            init='mst', init_priors=init_priors, niter=niter, schedule=schedule, lr=lr,
        )
        pred_traj = scene_clip.get_tum_poses()
        pred_traj_all[0] = np.concatenate([pred_traj_all[0], pred_traj[0]], axis=0)
        pred_traj_all[1] = np.concatenate([pred_traj_all[1], pred_traj[1] + offset], axis=0)
        K = scene_clip.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
        depth_maps = scene_clip.save_depth_maps(save_folder, offset)
        dynamic_masks = scene_clip.save_dynamic_masks(save_folder, offset)
        conf = scene_clip.save_conf_maps(save_folder, offset)
        init_conf = scene_clip.save_init_conf_maps(save_folder, offset)
        rgbs = scene_clip.save_rgb_imgs(save_folder, offset)
        offset += pred_traj[0].shape[0]

    enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 
    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    return scene


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
    coarse_init_pairs += [(img2, img1) for img1, img2 in coarse_init_pairs]


    # clips
    all_clips_id = []
    for clip in clips:
        each_clip_id = []
        for index in range(0, len(clip)):
            each_clip_id.append(clip[index]['idx'])
            clip[index]['idx'] = index
        all_clips_id.append(each_clip_id)

    stride = 2
    all_clips_pairs = []
    for clip in clips:
        each_clip_pairs = []
        for i in range(len(clip)-1):
            for j in range(i+1, len(clip), stride):
                each_clip_pairs.append((clip[i].copy(), clip[j].copy()))
        each_clip_pairs += [(img2, img1) for img1, img2 in each_clip_pairs]
        all_clips_pairs.append(each_clip_pairs)

    return coarse_init_pairs, keyframes_id, all_clips_pairs, all_clips_id

def select_images_with_fixed_intervals(imgs, clip_size):

    if clip_size >= len(imgs):
        return imgs[:clip_size]

    interval = len(imgs) / clip_size

    selected_indices = [int(i * interval) for i in range(clip_size)]

    selected_images = [imgs[i] for i in selected_indices]

    return selected_images

def my_make_pairs2(imgs, clip_size):

    keyframes_id = []
    coarse_init_clip = []
    clips = []

    for i in range(0, len(imgs), clip_size):
        coarse_init_clip.append(imgs[i].copy())
        keyframes_id.append(i)
        clips.append(imgs[i:i + clip_size])

    # coarse_init_clip
    clip_size = 10
    interval = len(imgs) / clip_size

    selected_indices = [int(i * interval) for i in range(clip_size)]

    for i in selected_indices:
        if i not in keyframes_id:
            keyframes_id.append(i)
            coarse_init_clip.append(imgs[i].copy())

    sorted_indices = sorted(range(len(keyframes_id)), key=lambda k: keyframes_id[k])
    keyframes_id.sort()
    coarse_init_clip[:] = [coarse_init_clip[i] for i in sorted_indices]

    coarse_init_pairs = []
    for index in range(0, len(coarse_init_clip)):
        coarse_init_clip[index]['idx'] = index
    for i in range(len(coarse_init_clip) - 1):
        for j in range(i+1, len(coarse_init_clip)):
            coarse_init_pairs.append((coarse_init_clip[i], coarse_init_clip[j]))
    coarse_init_pairs += [(img2, img1) for img1, img2 in coarse_init_pairs]


    # clips
    all_clips_id = []
    for clip in clips:
        each_clip_id = []
        for index in range(0, len(clip)):
            each_clip_id.append(clip[index]['idx'])
            clip[index]['idx'] = index
        all_clips_id.append(each_clip_id)

    stride = 2
    all_clips_pairs = []
    for clip in clips:
        each_clip_pairs = []
        for i in range(len(clip)-1):
            for j in range(i+1, len(clip), stride):
                each_clip_pairs.append((clip[i].copy(), clip[j].copy()))
        each_clip_pairs += [(img2, img1) for img1, img2 in each_clip_pairs]
        all_clips_pairs.append(each_clip_pairs)

    return coarse_init_pairs, keyframes_id, all_clips_pairs, all_clips_id

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # Use the provided output_dir or create a temporary directory
    tmpdirname = args.output_dir if args.output_dir is not None else tempfile.mkdtemp(suffix='dust3r_gradio_demo')

    if not args.silent:
        print('Outputting stuff in', tmpdirname)

    # Process images in the input directory with default parameters
    if os.path.isdir(args.input_dir):    # input_dir is a directory of images
        input_files = []
        for fname in sorted(os.listdir(args.input_dir)):
           if fname.lower().endswith(('.jpg', '.png')):
            input_files.append(os.path.join(args.input_dir, fname))
    elif args.input_dir.endswith(('.mp4', '.avi', '.mov', '.mkv')):   # input_dir is a video
        input_files = video_to_images(args.input_dir, args.input_dir.replace('.mp4', '_img'))
    if args.mode == 'eval_pose':
      recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, args.device, args.silent, args.image_size)
    elif args.mode == 'eval_pose_h':
      recon_fun = functools.partial(get_reconstructed_scene_hierachical, args, tmpdirname, model, args.device, args.silent, args.image_size)
    # Call the function with default parameters
    scene = recon_fun(
        filelist=input_files,
        schedule='linear',
        niter=300,
        min_conf_thr=1.1,
        as_pointcloud=True,
        mask_sky=False,
        clean_depth=True,
        transparent_cams=False,
        cam_size=0.05,
        show_cam=True,
        scenegraph_type='swinstride',
        winsize=5,
        refid=0,
        seq_name=args.seq_name,
        new_model_weights=args.weights,
        temporal_smoothing_weight=0.01,
        translation_weight='1.0',
        shared_focal=True,
        flow_loss_weight=0.01,
        flow_loss_start_iter=0.1,
        flow_loss_threshold=25,
        use_gt_mask=args.use_gt_davis_masks,
        fps=args.fps,
        interval=args.interval,
        depth_prior_name=args.depth_prior_name
    )
    print(f"Processing completed. Output saved in {tmpdirname}/{args.seq_name}")
