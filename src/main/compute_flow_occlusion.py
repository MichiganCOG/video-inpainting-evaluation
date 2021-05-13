#!/usr/bin/python
# Parts adapted from https://github.com/phoenix104104/fast_blind_video_consistency

from __future__ import print_function

### python lib
import os, argparse, glob, math, cv2

### torch lib
import torch

### custom lib
from ..models.flownet2.models import FlowNet2
from src.common_util import flow as cuf, image as cui


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='optical flow estimation')

    ### testing options
    parser.add_argument('-gpu',             type=int,     default=0,            help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                help='use cpu?')
    parser.add_argument('--gt_root', type=str, required=True, help='Path where GT frames are stored')
    parser.add_argument('--output_root', type=str, required=True, help='Where the flow occlusion data will be stored')
    parser.add_argument('--save_visual_flow', action='store_true', help='Flag to save visualizations of flow')
    parser.add_argument('--file_name_pattern', type=str, default='*_gt.png',
                        help='Glob pattern for selecting GT video frames')

    opts = parser.parse_args()

    ### update options
    opts.cuda = (opts.cpu != True)
    opts.grads = {} # dict to collect activation gradients (for training debug purpose)

    ### FlowNet options
    opts.rgb_max = 1.0
    opts.fp16 = False

    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")
    
    ### initialize FlowNet
    print('===> Initializing model from FlowNet2...')
    model = FlowNet2(opts)

    ### load pre-trained FlowNet
    model_filename = os.path.join(PROJ_DIR, 'pretrained_models', 'FlowNet2_checkpoint.pth.tar')
    print("===> Load %s" %model_filename)
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda" if opts.cuda else "cpu")
    model = model.to(device)
    model.eval()

    video_names = sorted(os.listdir(opts.gt_root))

    for video_name in video_names:

        frame_dir = os.path.join(opts.gt_root, video_name)
        fw_flow_dir = os.path.join(opts.output_root, 'fw_flow', video_name)
        if not os.path.isdir(fw_flow_dir):
            os.makedirs(fw_flow_dir)

        fw_occ_dir = os.path.join(opts.output_root, 'fw_occlusion', video_name)
        if not os.path.isdir(fw_occ_dir):
            os.makedirs(fw_occ_dir)
        if opts.save_visual_flow:
            fw_rgb_dir = os.path.join(opts.output_root, 'fw_flow_rgb', video_name)
            if not os.path.isdir(fw_rgb_dir):
                os.makedirs(fw_rgb_dir)

        frame_list = sorted(glob.glob(os.path.join(frame_dir, opts.file_name_pattern)))

        for t in range(len(frame_list) - 1):

            ### load input images
            img1 = cui.read_img(frame_list[t])
            img2 = cui.read_img(frame_list[t + 1])

            ### resize image
            size_multiplier = 64
            H_orig = img1.shape[0]
            W_orig = img1.shape[1]

            H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)

            img1 = cv2.resize(img1, (W_sc, H_sc))
            img2 = cv2.resize(img2, (W_sc, H_sc))

            with torch.no_grad():

                ### convert to tensor
                img1 = cui.numpy_image_to_tensor(img1).to(device)
                img2 = cui.numpy_image_to_tensor(img2).to(device)

                ### compute fw flow
                fw_flow = model(img1, img2)
                fw_flow = cui.tensor2img(fw_flow)

                ### compute bw flow
                bw_flow = model(img2, img1)
                bw_flow = cui.tensor2img(bw_flow)

            ### resize flow
            fw_flow = cuf.resize_flow(fw_flow, W_out = W_orig, H_out = H_orig)
            bw_flow = cuf.resize_flow(bw_flow, W_out = W_orig, H_out = H_orig)

            ### compute occlusion
            fw_occ = cuf.detect_occlusion(bw_flow, fw_flow)

            ### save flow
            output_flow_filename = os.path.join(fw_flow_dir, "%05d.flo" %t)
            cuf.save_flo(fw_flow, output_flow_filename)

            ### save occlusion map
            output_occ_filename = os.path.join(fw_occ_dir, "%05d.png" %t)
            cui.save_img(fw_occ, output_occ_filename, binary=True)

            ### save rgb flow
            if opts.save_visual_flow:
                output_filename = os.path.join(fw_rgb_dir, "%05d.png" %t)
                flow_rgb = cuf.flow_to_rgb(fw_flow)
                cui.save_img(flow_rgb, output_filename)
