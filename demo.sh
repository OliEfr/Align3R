#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R
CUDA_VISIBLE_DEVICES=0 python tool/demo.py --input input_dir/scene_name --output_dir demo_dir --seq_name scene_name --interval=50