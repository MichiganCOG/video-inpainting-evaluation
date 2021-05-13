#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/../..; pwd)"
cd "$PROJ_DIR/pretrained_models"

wget -O FlowNet2_checkpoint.pth.tar https://umich.box.com/shared/static/py7acekxq36v3lzwnxatuzl7z1b1reqy.tar &
wget -O alex.pth https://github.com/richzhang/PerceptualSimilarity/raw/c33f89e9f46522a584cf41d8880eb0afa982708b/lpips/weights/v0.1/alex.pth &
wget -O rgb_imagenet.pt https://github.com/piergiaj/pytorch-i3d/raw/eb3580bc5a9f3f7dd07d3162ed1d9674581ed3a5/models/rgb_imagenet.pt &

wait
