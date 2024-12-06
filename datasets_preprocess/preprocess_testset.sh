#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R
python datasets_preprocess/preprocess_bonn.py
python datasets_preprocess/prepare_tum.py