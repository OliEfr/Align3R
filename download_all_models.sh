#!/bin/bash

# if models folder exists, we most likely downloaded the models already
if [ ! -d "third_party/RAFT/models" ]; then
    # Depthpro
    cd third_party/ml-depth-pro
    mkdir -p checkpoints && \
    wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints && \
    cd ../.. &&


    wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
    gdown --fuzzy https://drive.google.com/file/d/1-qhRtgH7rcJMYZ5sWRdkrc2_9wsR1BBG/view?usp=sharing
    gdown --fuzzy https://drive.google.com/file/d/1PPmpbASVbFdjXnD3iea-MRIHGmKsS8Vh/view?usp=sharing

    cd third_party/RAFT
    gdown --fuzzy https://drive.google.com/file/d/1KJxQ7KPuGHlSftsBCV1h2aYpeqQv3OI-/view?usp=drive_link -O models/
fi


