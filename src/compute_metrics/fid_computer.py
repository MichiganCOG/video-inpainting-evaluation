import numpy as np
import torch

from .metric_computer import MetricComputer
from ..common_util.image import get_gt_frame, get_comp_frame
from ..fid import calculate_frechet_distance, InceptionV3
from ..fid.util import extract_video_frame_features


class FIDComputer(MetricComputer):

    def compute_metric(self):
        num_work_items = 2 * len(self.opts.video_names) if self.opts.gt_fid_feats_path is None \
            else len(self.opts.video_names)
        self.send_work_count_msg(num_work_items)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx])
        model.cuda()
        model.eval()
        torch.set_grad_enabled(False)

        def get_gt_frame_wrapper(video_name, t):
            return get_gt_frame(self.opts.gt_root, video_name, t)

        def get_comp_frame_wrapper(video_name, t):
            return get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, t)

        def update_progress_cb():
            self.send_update_msg(1)

        if self.opts.gt_fid_feats_path is None:
            gt_frame_features = extract_video_frame_features(model, get_gt_frame_wrapper, self.opts.video_names,
                                                             self.opts.video_frame_counts, update_progress_cb)
        else:
            gt_frame_features = np.load(self.opts.gt_fid_feats_path)
        gt_mu = np.mean(gt_frame_features, axis=0)
        gt_sigma = np.cov(gt_frame_features, rowvar=False)

        comp_frame_features = extract_video_frame_features(model, get_comp_frame_wrapper, self.opts.video_names,
                                                           self.opts.video_frame_counts, update_progress_cb)
        comp_mu = np.mean(comp_frame_features, axis=0)
        comp_sigma = np.cov(comp_frame_features, rowvar=False)

        fid_all = calculate_frechet_distance(gt_mu, gt_sigma, comp_mu, comp_sigma)

        self.send_result_msg(fid_all)
