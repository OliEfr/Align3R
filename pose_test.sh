#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R

CUDA_VISIBLE_DEVICES='0' python tool/pose_test.py --dust3r_dynamic_model_path="align3r_depthpro.pth" --output_postfix="results/sintel_pose_ours_depthpro" --dataset_name=sintel --depth_prior_name=depthpro --start_frame=0 --interval_frame=3000 --mode=eval_pose --scene_graph_type=swinstride-5-noncyclic