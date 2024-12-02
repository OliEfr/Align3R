#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R
CUDA_VISIBLE_DEVICES='7' python tool/depth_test.py --dust3r_dynamic_model_path="align3r_depthpro.pth" --align_with_lad  --depth_max=70 --depth_prior_name=depthpro --dataset_name=sintel --eval