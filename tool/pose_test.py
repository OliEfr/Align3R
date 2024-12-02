import os
import math
import cv2
import numpy as np
import torch
import argparse
from dust3r.utils.vo_eval import load_traj, eval_metrics, plot_trajectory, save_trajectory_tum_format, process_directory, calculate_averages
import croco.utils.misc as misc
import torch.distributed as dist
from tqdm import tqdm
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt_flow import global_aligner, GlobalAlignerMode
from dust3r.utils.image_pose  import load_images, rgb, enlarge_seg_masks
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.eval_metadata import dataset_metadata
import sys

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--dust3r_dynamic_model_path", type=str, help="path to the dust3r model weights", default=None)
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument('--pose_eval_freq', default=0, type=int, help='pose evaluation frequency')
    parser.add_argument('--pose_eval_stride', default=1, type=int, help='stride for pose evaluation')
    parser.add_argument('--scene_graph_type', default='swinstride-5-noncyclic', type=str, help='scene graph window size')
    parser.add_argument('--save_best_pose', action='store_true', default=False, help='save best pose')
    parser.add_argument('--n_iter', default=300, type=int, help='number of iterations for pose optimization')
    parser.add_argument('--save_pose_qualitative', action='store_true', default=False, help='save qualitative pose results')
    parser.add_argument('--temporal_smoothing_weight', default=0.01, type=float, help='temporal smoothing weight for pose optimization')
    parser.add_argument('--not_shared_focal', action='store_true', default=False, help='use shared focal length for pose optimization')
    parser.add_argument('--use_gt_focal', action='store_true', default=False, help='use ground truth focal length for pose optimization')
    parser.add_argument('--pose_schedule', default='linear', type=str, help='pose optimization schedule')
    
    parser.add_argument('--flow_loss_weight', default=0.01, type=float, help='flow loss weight for pose optimization')
    parser.add_argument('--flow_loss_fn', default='smooth_l1', type=str, help='flow loss type for pose optimization')
    parser.add_argument('--use_gt_mask', action='store_true', default=False, help='use gt mask for pose optimization, for sintel/davis')
    parser.add_argument('--motion_mask_thre', default=0.35, type=float, help='motion mask threshold for pose optimization')
    parser.add_argument('--sam2_mask_refine', action='store_true', default=False, help='use sam2 mask refine for the motion for pose optimization')
    parser.add_argument('--flow_loss_start_epoch', default=0.1, type=float, help='start epoch for flow loss')
    parser.add_argument('--flow_loss_thre', default=20, type=float, help='threshold for flow loss')
    parser.add_argument('--pxl_thresh', default=50.0, type=float, help='threshold for flow loss')
    parser.add_argument('--depth_regularize_weight', default=0.0, type=float, help='depth regularization weight for pose optimization')
    parser.add_argument('--start_frame', default=0, type=int, help='start frame')
    parser.add_argument('--interval_frame', default=30, type=int, help='start frame')
    parser.add_argument('--translation_weight', default=1, type=float, help='translation weight for pose optimization')
    parser.add_argument('--silent', action='store_true', default=False, help='silent mode for pose evaluation')
    parser.add_argument('--full_seq', action='store_true', default=False, help='use full sequence for pose evaluation')

    parser.add_argument('--seq_list', nargs='+', default=None, help='list of sequences for pose evaluation')

    parser.add_argument("--dataset_name", type=str, default=None, choices=['bonn', 'tum', 'sintel'], help="choose dataset for pose evaluation")

    # for monocular depth eval
    parser.add_argument('--no_crop', action='store_true', default=False, help='do not crop the image for monocular depth evaluation')

    # output dir
    parser.add_argument('--output_postfix', default='./results/tmp', type=str, help="path where to save the output")
    parser.add_argument("--depth_prior_name", type=str, default='depthpro', choices=['depthpro', 'depthanything'], help="the name of monocular depth estimation model")
    parser.add_argument("--mode", type=str, default='eval_pose', choices=['eval_pose', 'eval_pose_h'], help="eval pose hierarchically or not")
    return parser

def eval_pose_estimation(args, model, device, save_dir=None):
    metadata = dataset_metadata.get(args.dataset_name, dataset_metadata['sintel'])
    img_path = metadata['img_path']
    mask_path = metadata['mask_path']

    ate_mean, rpe_trans_mean, rpe_rot_mean, bug = eval_pose_estimation_dist(
        args, model, device, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean, bug

def eval_pose_estimation_dist(args, model, device, img_path, save_dir=None, mask_path=None):

    metadata = dataset_metadata.get(args.dataset_name, dataset_metadata['sintel'])
    anno_path = metadata.get('anno_path', None)
    #print(anno_path)
    silent = args.silent
    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get('full_seq', False):
            args.full_seq = True
        else:
            seq_list = metadata.get('seq_list', [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_postfix

    # Split seq_list across processes
    if misc.is_dist_avail_and_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    total_seqs = len(seq_list)
    seqs_per_proc = (total_seqs + world_size - 1) // world_size  # Ceiling division

    start_idx = rank * seqs_per_proc
    end_idx = min(start_idx + seqs_per_proc, total_seqs)

    seq_list = seq_list[start_idx:end_idx]
    start = args.start_frame
    interval = args.interval_frame
    ate_list = []
    rpe_trans_list = []
    rpe_rot_list = []
    valid_seq = []
    load_img_size = 512

    error_log_path = f'{save_dir}/_error_log_{rank}.txt'  # Unique log file per process
    bug = False
    for seq in tqdm(seq_list):
        try:
            dir_path = metadata['dir_path_func'](img_path, seq)
            print(dir_path)
            # Handle skip_condition
            skip_condition = metadata.get('skip_condition', None)
            if skip_condition is not None and skip_condition(save_dir, seq):
                continue

            mask_path_seq_func = metadata.get('mask_path_seq_func', lambda mask_path, seq: None)
            #print(mask_path)
            mask_path_seq = mask_path_seq_func(mask_path, seq)

            filelist = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            filelist.sort()
            filelist = filelist[::args.pose_eval_stride]
            max_winsize = max(1, math.ceil((len(filelist)-1)/2))
            scene_graph_type = args.scene_graph_type
            if int(scene_graph_type.split('-')[1]) > max_winsize:
                scene_graph_type = f'{args.scene_graph_type.split("-")[0]}-{max_winsize}'
                if len(scene_graph_type.split("-")) > 2:
                    scene_graph_type += f'-{args.scene_graph_type.split("-")[2]}'
            print("args.no_crop", args.no_crop)
            imgs = load_images(filelist, size=load_img_size, verbose=False, dynamic_mask_root=mask_path_seq, crop=not args.no_crop, traj_format=args.dataset_name, start=start, interval=interval, depth_prior_name=args.depth_prior_name)

            if args.dataset_name == 'davis' and len(imgs) > 95:
                # use swinstride-4
                scene_graph_type = scene_graph_type.replace('5', '4')
            #print(scene_graph_type)
            pairs = make_pairs(
                imgs, scene_graph=scene_graph_type, prefilter=None, symmetrize=True
            ) 


            output = inference(pairs, model, device, batch_size=1, verbose=not silent)

            if seq in ['temple_3']:
              args.flow_loss_thre=10
            else:
              args.flow_loss_thre=40
            
            with torch.enable_grad():
                if len(imgs) > 2:
                    mode = GlobalAlignerMode.PointCloudOptimizer
                    scene = global_aligner(
                        output, device=device, mode=mode, verbose=not silent,
                        shared_focal=not args.not_shared_focal and not args.use_gt_focal,
                        flow_loss_weight=args.flow_loss_weight, flow_loss_fn=args.flow_loss_fn,
                        depth_regularize_weight=args.depth_regularize_weight,
                        num_total_iter=args.n_iter, temporal_smoothing_weight=args.temporal_smoothing_weight, motion_mask_thre=args.motion_mask_thre,
                        flow_loss_start_epoch=args.flow_loss_start_epoch, flow_loss_thre=args.flow_loss_thre, translation_weight=args.translation_weight,
                        sintel_ckpt=args.dataset_name == 'sintel', use_self_mask=not args.use_gt_mask, sam2_mask_refine=args.sam2_mask_refine,
                        empty_cache=len(imgs) >= 80 and len(pairs) > 600, pxl_thre=args.pxl_thresh, # empty cache to make it run on 48GB GPU
                        #min_conf_thr=2,
                    )
                    print('min_conf_thr',scene.min_conf_thr)
                    if args.use_gt_focal:
                        focal_path = os.path.join(
                            img_path.replace('final', 'camdata_left'), seq, 'focal.txt'
                        )
                        focals = np.loadtxt(focal_path)
                        focals = focals[::args.pose_eval_stride]
                        original_img_size = cv2.imread(filelist[0]).shape[:2]
                        resized_img_size = tuple(imgs[0]['img'].shape[-2:])
                        focals = focals * max(
                            (resized_img_size[0] / original_img_size[0]),
                            (resized_img_size[1] / original_img_size[1])
                        )
                        scene.preset_focal(focals, requires_grad=False)  # TODO: requires_grad=False
                    lr = 0.01
                    loss = scene.compute_global_alignment(
                        init='mst', niter=args.n_iter, schedule=args.pose_schedule, lr=lr,
                    )
                else:
                    mode = GlobalAlignerMode.PairViewer
                    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)

            pred_traj = scene.get_tum_poses()

            os.makedirs(f'{save_dir}/{seq}', exist_ok=True)
            scene.clean_pointcloud()
            scene.save_tum_poses(f'{save_dir}/{seq}/pred_traj.txt')
            scene.save_focals(f'{save_dir}/{seq}/pred_focal.txt')
            scene.save_intrinsics(f'{save_dir}/{seq}/pred_intrinsics.txt')
            scene.save_depth_maps(f'{save_dir}/{seq}', start)
            scene.save_dynamic_masks(f'{save_dir}/{seq}',start)
            scene.save_conf_maps(f'{save_dir}/{seq}',start)
            scene.save_init_conf_maps(f'{save_dir}/{seq}',start)
            scene.save_rgb_imgs(f'{save_dir}/{seq}',start)
            enlarge_seg_masks(f'{save_dir}/{seq}', kernel_size=5 if args.use_gt_mask else 3)

            gt_traj_file = metadata['gt_traj_func'](img_path, anno_path, seq)
            traj_format = metadata.get('traj_format', None)

            if args.dataset_name == 'sintel':
                gt_traj = load_traj(gt_traj_file=gt_traj_file, stride=args.pose_eval_stride)
            elif traj_format is not None:
                gt_traj = load_traj(gt_traj_file=gt_traj_file, traj_format=traj_format)
            else:
                gt_traj = None
            #gt_traj =None
            if gt_traj is not None:
                #print(gt_traj[1],pred_traj[1])
                ate, rpe_trans, rpe_rot = eval_metrics(
                    pred_traj, (gt_traj[0][start:start+interval], pred_traj[1]), seq=seq, filename=f'{save_dir}/{seq}_eval_metric.txt'
                )
                plot_trajectory(
                    pred_traj, (gt_traj[0][start:start+interval], pred_traj[1]), title=seq, filename=f'{save_dir}/{seq}.png'
                )
                print(ate, rpe_trans, rpe_rot)
            else:
                ate, rpe_trans, rpe_rot = 0, 0, 0
                bug = True

            ate_list.append(ate)
            rpe_trans_list.append(rpe_trans)
            rpe_rot_list.append(rpe_rot)
            valid_seq.append(seq)
            # Write to error log after each sequence
            with open(error_log_path, 'a') as f:
                f.write(f'{args.dataset_name}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n')
                f.write(f'{ate:.5f}\n')
                f.write(f'{rpe_trans:.5f}\n')
                f.write(f'{rpe_rot:.5f}\n')

        except Exception as e:
            if 'out of memory' in str(e):
                # Handle OOM
                torch.cuda.empty_cache()  # Clear the CUDA memory
                with open(error_log_path, 'a') as f:
                    f.write(f'OOM error in sequence {seq}, skipping this sequence.\n')
                print(f'OOM error in sequence {seq}, skipping...')
            elif 'Degenerate covariance rank' in str(e) or 'Eigenvalues did not converge' in str(e):
                # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                with open(error_log_path, 'a') as f:
                    f.write(f'Exception in sequence {seq}: {str(e)}\n')
                print(f'Traj evaluation error in sequence {seq}, skipping.')
            else:
                raise e  # Rethrow if it's not an expected exception
            
    # Aggregate results across all processes
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    bug_tensor = torch.tensor(int(bug), device=device)

    bug = bool(bug_tensor.item())


    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if rank == 0:
        with open(f'{save_dir}/_error_log.txt', 'a') as f:
            # Copy the error log from each process to the main error log
            for i in range(world_size):
                with open(f'{save_dir}/_error_log_{i}.txt', 'r') as f_sub:
                    f.write(f_sub.read())
            f.write(f'Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n')
    print('valid_seq: ',valid_seq)
    return avg_ate, avg_rpe_trans, avg_rpe_rot, bug

def eval_pose_estimation_hierachical(args, model, device, save_dir=None):
    metadata = dataset_metadata.get(args.dataset_name, dataset_metadata['sintel'])
    img_path = metadata['img_path']
    mask_path = metadata['mask_path']

    ate_mean, rpe_trans_mean, rpe_rot_mean, bug = eval_pose_estimation_dist_h(
        args, model, device, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean, bug

def eval_pose_estimation_dist_h(args, model, device, img_path, save_dir=None, mask_path=None):

    metadata = dataset_metadata.get(args.dataset_name, dataset_metadata['sintel'])
    anno_path = metadata.get('anno_path', None)
    #print(anno_path)
    silent = args.silent
    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get('full_seq', False):
            args.full_seq = True
        else:
            seq_list = metadata.get('seq_list', [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    # Split seq_list across processes
    if misc.is_dist_avail_and_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    total_seqs = len(seq_list)
    seqs_per_proc = (total_seqs + world_size - 1) // world_size  # Ceiling division

    start_idx = rank * seqs_per_proc
    end_idx = min(start_idx + seqs_per_proc, total_seqs)

    seq_list = seq_list[start_idx:end_idx]
    start = args.start_frame
    interval = args.interval_frame
    ate_list = []
    rpe_trans_list = []
    rpe_rot_list = []
    valid_seq = []
    load_img_size = 512
    clip_size = 20
    args.flow_loss_thre=40
    error_log_path = f'{save_dir}/_error_log_{rank}.txt'  # Unique log file per process
    bug = False
    for seq in tqdm(seq_list):
        try:
            dir_path = metadata['dir_path_func'](img_path, seq)
            print(dir_path)
            # Handle skip_condition
            skip_condition = metadata.get('skip_condition', None)
            if skip_condition is not None and skip_condition(save_dir, seq):
                continue

            mask_path_seq_func = metadata.get('mask_path_seq_func', lambda mask_path, seq: None)
            #print(mask_path)
            mask_path_seq = mask_path_seq_func(mask_path, seq)

            filelist = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            filelist.sort()
            filelist = filelist[::args.pose_eval_stride]
            max_winsize = max(1, math.ceil((len(filelist)-1)/2))
            scene_graph_type = args.scene_graph_type
            if int(scene_graph_type.split('-')[1]) > max_winsize:
                scene_graph_type = f'{args.scene_graph_type.split("-")[0]}-{max_winsize}'
                if len(scene_graph_type.split("-")) > 2:
                    scene_graph_type += f'-{args.scene_graph_type.split("-")[2]}'
            print("args.no_crop", args.no_crop)
            imgs = load_images(filelist, size=load_img_size, verbose=False,dynamic_mask_root=mask_path_seq, crop=not args.no_crop, traj_format=args.dataset_name, start=start, interval=interval)

            if args.dataset_name == 'davis' and len(imgs) > 95:
                # use swinstride-4
                scene_graph_type = scene_graph_type.replace('5', '4')
            pairs = make_pairs(
                imgs, scene_graph=scene_graph_type, prefilter=None, symmetrize=True
            ) 
            while len(imgs) % clip_size == 1 or len(imgs) % clip_size == 0 or clip_size>len(imgs):
              clip_size -= 1
            coarse_init_pairs, keyframes_id, all_clips_pairs, all_clips_id = my_make_pairs(imgs, clip_size)


            if seq in ['temple_3']:
              args.flow_loss_thre=10
            else:
              args.flow_loss_thre=40
            key_output = inference(coarse_init_pairs, model, device, batch_size=1, verbose=not silent)
            with torch.enable_grad():
                if len(imgs) > 2:
                    mode = GlobalAlignerMode.PointCloudOptimizer
                    scene = global_aligner(
                        key_output, device=device, mode=mode, verbose=not silent,
                        shared_focal=not args.not_shared_focal and not args.use_gt_focal,
                        flow_loss_weight=args.flow_loss_weight, flow_loss_fn=args.flow_loss_fn,
                        depth_regularize_weight=args.depth_regularize_weight,
                        num_total_iter=args.n_iter, temporal_smoothing_weight=args.temporal_smoothing_weight, motion_mask_thre=args.motion_mask_thre,
                        flow_loss_start_epoch=args.flow_loss_start_epoch, flow_loss_thre=args.flow_loss_thre, translation_weight=args.translation_weight,
                        sintel_ckpt=args.dataset_name == 'sintel', use_self_mask=not args.use_gt_mask, sam2_mask_refine=args.sam2_mask_refine,
                        empty_cache=len(imgs) >= 80 and len(pairs) > 600, pxl_thre=args.pxl_thresh, # empty cache to make it run on 48GB GPU
                        #min_conf_thr=2,
                    )
                    print('min_conf_thr',scene.min_conf_thr)
                    if args.use_gt_focal:
                        focal_path = os.path.join(
                            img_path.replace('final', 'camdata_left'), seq, 'focal.txt'
                        )
                        focals = np.loadtxt(focal_path)
                        focals = focals[::args.pose_eval_stride]
                        original_img_size = cv2.imread(filelist[0]).shape[:2]
                        resized_img_size = tuple(imgs[0]['img'].shape[-2:])
                        focals = focals * max(
                            (resized_img_size[0] / original_img_size[0]),
                            (resized_img_size[1] / original_img_size[1])
                        )
                        scene.preset_focal(focals, requires_grad=False)  # TODO: requires_grad=False
                    lr = 0.01
                    loss = scene.compute_global_alignment(
                        init='mst', niter=args.n_iter, schedule=args.pose_schedule, lr=lr,
                    )
                else:
                    mode = GlobalAlignerMode.PairViewer
                    scene = global_aligner(key_output, device=device, mode=mode, verbose=not silent)
                init_keyposes = to_numpy(scene.get_im_poses()).tolist()
                init_keydepths = to_numpy(scene.get_depthmaps())
                init_keyfocals = to_numpy(scene.get_focals()).tolist()
                offset = 0
                pred_traj_all = [np.zeros((0,7)), np.zeros((0,))]
                os.makedirs(f'{save_dir}/{seq}', exist_ok=True)
                depth_imgs = []
                for init_keypose, init_keydepth, init_keyfocal, clip_pair in zip(init_keyposes, init_keydepths, init_keyfocals, all_clips_pairs):
                    clipwise_output = inference(clip_pair, model, device, batch_size=1, verbose=not silent)
                    mode_clip = GlobalAlignerMode.PointCloudOptimizer
                    scene_clip = global_aligner(
                        clipwise_output, device=device, mode=mode_clip, verbose=not silent,
                        shared_focal=not args.not_shared_focal and not args.use_gt_focal,
                        flow_loss_weight=args.flow_loss_weight, flow_loss_fn=args.flow_loss_fn,
                        depth_regularize_weight=args.depth_regularize_weight,
                        num_total_iter=args.n_iter, temporal_smoothing_weight=args.temporal_smoothing_weight, motion_mask_thre=args.motion_mask_thre,
                        flow_loss_start_epoch=args.flow_loss_start_epoch, flow_loss_thre=args.flow_loss_thre, translation_weight=args.translation_weight,
                        sintel_ckpt=args.dataset_name == 'sintel', use_self_mask=not args.use_gt_mask, sam2_mask_refine=args.sam2_mask_refine,
                        empty_cache=len(imgs) >= 80 and len(pairs) > 600, pxl_thre=args.pxl_thresh, # empty cache to make it run on 48GB GPU
                        #min_conf_thr=2,
                    )
                    if args.use_gt_focal:
                        focal_path = os.path.join(
                            img_path.replace('final', 'camdata_left'), seq, 'focal.txt'
                        )
                        focals = np.loadtxt(focal_path)
                        focals = focals[::args.pose_eval_stride]
                        original_img_size = cv2.imread(filelist[0]).shape[:2]
                        resized_img_size = tuple(imgs[0]['img'].shape[-2:])
                        focals = focals * max(
                            (resized_img_size[0] / original_img_size[0]),
                            (resized_img_size[1] / original_img_size[1])
                        )
                        scene.preset_focal(focals, requires_grad=False)  # TODO: requires_grad=False
                    lr = 0.01
                    init_priors = [init_keypose, init_keydepth, init_keyfocal]
                    loss = scene_clip.compute_global_alignment(
                        init='mst', init_priors=init_priors, niter=args.n_iter, schedule=args.pose_schedule, lr=lr,
                    )
                    pred_traj = scene_clip.get_tum_poses()
                    pred_traj_all[0] = np.concatenate([pred_traj_all[0], pred_traj[0]],axis=0)
                    pred_traj_all[1] = np.concatenate([pred_traj_all[1], pred_traj[1] + offset],axis=0)
                    scene_clip.save_depth_maps(f'{save_dir}/{seq}', offset)
                    scene_clip.save_dynamic_masks(f'{save_dir}/{seq}',offset)
                    scene_clip.save_conf_maps(f'{save_dir}/{seq}',offset)
                    scene_clip.save_init_conf_maps(f'{save_dir}/{seq}',offset)
                    scene_clip.save_rgb_imgs(f'{save_dir}/{seq}',offset)
                    offset += pred_traj[0].shape[0]
                    scene_clip.clean_pointcloud()
                    depths = to_numpy(scene_clip.get_depthmaps())
                    

            enlarge_seg_masks(f'{save_dir}/{seq}', kernel_size=5 if args.use_gt_mask else 3)

            gt_traj_file = metadata['gt_traj_func'](img_path, anno_path, seq)
            traj_format = metadata.get('traj_format', None)

            if args.dataset_name == 'sintel':
                gt_traj = load_traj(gt_traj_file=gt_traj_file, stride=args.pose_eval_stride)
            elif traj_format is not None:
                gt_traj = load_traj(gt_traj_file=gt_traj_file, traj_format=traj_format)
            else:
                gt_traj = None
            #gt_traj =None
            if gt_traj is not None:
                #print(gt_traj[0])
                ate, rpe_trans, rpe_rot = eval_metrics(
                    pred_traj_all, gt_traj, seq=seq, filename=f'{save_dir}/{seq}_eval_metric.txt'
                )
                plot_trajectory(
                    pred_traj_all, gt_traj, title=seq, filename=f'{save_dir}/{seq}.png'
                )
            else:
                ate, rpe_trans, rpe_rot = 0, 0, 0
                bug = True

            ate_list.append(ate)
            rpe_trans_list.append(rpe_trans)
            rpe_rot_list.append(rpe_rot)
            valid_seq.append(seq)
            # Write to error log after each sequence
            with open(error_log_path, 'a') as f:
                f.write(f'{args.dataset_name}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n')
                f.write(f'{ate:.5f}\n')
                f.write(f'{rpe_trans:.5f}\n')
                f.write(f'{rpe_rot:.5f}\n')

        except Exception as e:
            if 'out of memory' in str(e):
                # Handle OOM
                torch.cuda.empty_cache()  # Clear the CUDA memory
                with open(error_log_path, 'a') as f:
                    f.write(f'OOM error in sequence {seq}, skipping this sequence.\n')
                print(f'OOM error in sequence {seq}, skipping...')
            elif 'Degenerate covariance rank' in str(e) or 'Eigenvalues did not converge' in str(e):
                # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                with open(error_log_path, 'a') as f:
                    f.write(f'Exception in sequence {seq}: {str(e)}\n')
                print(f'Traj evaluation error in sequence {seq}, skipping.')
            else:
                raise e  # Rethrow if it's not an expected exception
            
    # Aggregate results across all processes
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    bug_tensor = torch.tensor(int(bug), device=device)

    bug = bool(bug_tensor.item())

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if rank == 0:
        with open(f'{save_dir}/_error_log.txt', 'a') as f:
            # Copy the error log from each process to the main error log
            for i in range(world_size):
                with open(f'{save_dir}/_error_log_{i}.txt', 'r') as f_sub:
                    f.write(f_sub.read())
            f.write(f'Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n')
    print('valid_seq: ',valid_seq)
    return avg_ate, avg_rpe_trans, avg_rpe_rot, bug

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

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    dust3r_dynamic_weights_path = args.dust3r_dynamic_model_path
    dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_dynamic_weights_path).to(args.device)
    os.makedirs(args.output_postfix, exist_ok=True)
    if args.mode == 'eval_pose_h':
      ate_mean, rpe_trans_mean, rpe_rot_mean, bug = eval_pose_estimation_hierachical(args, dust3r_model, args.device, save_dir=args.output_postfix)
    elif args.mode == 'eval_pose':
      ate_mean, rpe_trans_mean, rpe_rot_mean, bug = eval_pose_estimation(args, dust3r_model, args.device, save_dir=args.output_postfix)
    print(f'ATE mean: {ate_mean}, RPE trans mean: {rpe_trans_mean}, RPE rot mean: {rpe_rot_mean}')