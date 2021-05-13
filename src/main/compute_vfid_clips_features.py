import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from ..common_util.image import get_gt_frame
from ..common_util.misc import get_video_names_and_frame_counts, makedirs
from ..fid.util import extract_video_clip_features
from ..models.i3d.pytorch_i3d import InceptionI3d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))


def main(gt_root, output_path, max_num_videos):
    video_names, video_frame_counts = get_video_names_and_frame_counts(gt_root, max_num_videos)

    model = InceptionI3d(400, in_channels=3)
    weights = torch.load(os.path.join(PROJ_DIR, 'pretrained_models', 'rgb_imagenet.pt'))
    model.load_state_dict(weights)
    model.cuda()
    model.eval()
    torch.set_grad_enabled(False)

    prog_bar = tqdm(total=len(video_names))

    def get_gt_frame_wrapper(video_name, t):
        return get_gt_frame(gt_root, video_name, t)

    def update_progress_cb():
        prog_bar.update()

    gt_frame_features = extract_video_clip_features(model, get_gt_frame_wrapper, video_names, video_frame_counts,
                                                    update_progress_cb)
    makedirs(os.path.dirname(output_path))
    np.save(output_path, gt_frame_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_root', type=str, required=True, help='Path where all GT and mask frames are stored')
    parser.add_argument('--output_path', type=str, required=True, help='Path where features will be stored')
    parser.add_argument('--max_num_videos', type=int, default=None, help='How many videos to extract features from')
    args = parser.parse_args()
    main(**vars(args))
