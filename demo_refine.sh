#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R
CUDA_VISIBLE_DEVICES=0 python tool/demo_refine.py --input assets/maila --output_dir output/maila_refine --seq_name maila --interval=100