#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/../..; pwd)"
cd "$PROJ_DIR/pretrained_models"

wget -O FlowNet2_checkpoint.pth.tar https://web.eecs.umich.edu/~szetor/media/video-inpainting-evaluation/FlowNet2_checkpoint.pth.tar &
wget -O rgb_imagenet.pt https://github.com/piergiaj/pytorch-i3d/raw/eb3580bc5a9f3f7dd07d3162ed1d9674581ed3a5/models/rgb_imagenet.pt &

wait
