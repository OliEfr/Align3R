#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R
CUDA_VISIBLE_DEVICES='7' python tool/depth_test.py --dust3r_dynamic_model_path="/home/lipeng/ljh_code/Video_Depth_CVPR2025-main/dust3r_train/checkpoints/dust3r_512dpt_finedynamic_depthanything_1/checkpoint-best.pth" --align_with_lad  --depth_max=70 --depth_prior_name=depthanything --dataset_name=PointOdyssey --eval