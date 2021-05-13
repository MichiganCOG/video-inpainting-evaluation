import argparse

import numpy as np
import os
import torch
from tqdm import tqdm

from ..common_util.image import get_gt_frame
from ..common_util.misc import get_video_names_and_frame_counts, makedirs
from ..fid import InceptionV3
from ..fid.util import extract_video_frame_features


def main(gt_root, output_path, max_num_videos):
    video_names, video_frame_counts = get_video_names_and_frame_counts(gt_root, max_num_videos)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model.cuda()
    model.eval()
    torch.set_grad_enabled(False)

    prog_bar = tqdm(total=len(video_names))

    def get_gt_frame_wrapper(video_name, t):
        return get_gt_frame(gt_root, video_name, t)

    def update_progress_cb():
        prog_bar.update()

    gt_frame_features = extract_video_frame_features(model, get_gt_frame_wrapper, video_names, video_frame_counts,
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
