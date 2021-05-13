import os

import numpy as np
import torch

from .metric_computer import MetricComputer
from ..common_util.image import get_gt_frame, get_comp_frame
from ..fid import calculate_frechet_distance
from ..fid.util import extract_video_clip_features
from ..models.i3d.pytorch_i3d import InceptionI3d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))


class VFIDClipsComputer(MetricComputer):
    
    def compute_metric(self):
        num_work_items = 2 * len(self.opts.video_names) if self.opts.gt_vfid_feats_path is None \
            else len(self.opts.video_names)
        self.send_work_count_msg(num_work_items)
    
        model = InceptionI3d(400, in_channels=3)
        weights = torch.load(os.path.join(PROJ_DIR, 'pretrained_models', 'rgb_imagenet.pt'))
        model.load_state_dict(weights)
        model.cuda()
        model.eval()
        torch.set_grad_enabled(False)
    
        def get_gt_frame_wrapper(video_name, t):
            return get_gt_frame(self.opts.gt_root, video_name, t)
    
        def get_comp_frame_wrapper(video_name, t):
            return get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, t)
    
        def update_progress_cb():
            self.send_update_msg(1)
    
        if self.opts.gt_vfid_feats_path is None:
            gt_clip_features = extract_video_clip_features(model, get_gt_frame_wrapper, self.opts.video_names,
                                                           self.opts.video_frame_counts, update_progress_cb)
        else:
            gt_clip_features = np.load(self.opts.gt_vfid_clips_feats_path)
        gt_mu = np.mean(gt_clip_features, axis=0)
        gt_sigma = np.cov(gt_clip_features, rowvar=False)
    
        comp_clip_feats = extract_video_clip_features(model, get_comp_frame_wrapper, self.opts.video_names,
                                                      self.opts.video_frame_counts, update_progress_cb)
        comp_mu = np.mean(comp_clip_feats, axis=0)
        comp_sigma = np.cov(comp_clip_feats, rowvar=False)
    
        vfid_clips_all = calculate_frechet_distance(gt_mu, gt_sigma, comp_mu, comp_sigma)
    
        self.send_result_msg(vfid_clips_all)
