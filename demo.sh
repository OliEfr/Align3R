#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R
CUDA_VISIBLE_DEVICES=0 python tool/demo.py --input assets/maila --output_dir output --seq_name maila --interval=100