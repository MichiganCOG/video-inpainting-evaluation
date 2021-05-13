# Parts adapted from https://github.com/phoenix104104/fast_blind_video_consistency

import os

import torch
import numpy as np

from .metric_computer import MetricComputer
from ..common_util.flow import read_flo
from ..common_util.image import get_comp_frame, pil_rgb_to_numpy, numpy_image_to_tensor, read_img, tensor2img, \
    numpy_3d_array_to_tensor
from ..models.flownet2.networks.resample2d_package.resample2d import Resample2d


class WarpErrorComputer(MetricComputer):

    def compute_metric(self):
        self.send_work_count_msg(len(self.opts.video_names))
    
        torch.set_grad_enabled(False)
    
        # Flow warping layer
        device = torch.device('cuda' if self.opts.cuda else 'cpu')
        flow_warping = Resample2d().to(device)
        
        warp_error_all = np.zeros(len(self.opts.video_names))
        for v in range(len(self.opts.video_names)):
    
            video_name = self.opts.video_names[v]
            num_frames = self.opts.video_frame_counts[v]
    
            occ_dir = os.path.join(self.opts.eval_feats_root, 'fw_occlusion', video_name)
            flow_dir = os.path.join(self.opts.eval_feats_root, 'fw_flow', video_name)
    
            comp_frame_a = get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, 0)
            comp_frame_a = pil_rgb_to_numpy(comp_frame_a)

            cur_warp_error_values = []
    
            for t in range(1, num_frames):

                # Load input images for warping consistency
                comp_frame_b = get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, t)
                comp_frame_b = pil_rgb_to_numpy(comp_frame_b)
                comp_frame_b_t = numpy_image_to_tensor(comp_frame_b).contiguous().to(device)
    
                # Load flow
                flow = read_flo(os.path.join(flow_dir, '%05d.flo' % (t - 1)))
                flow_t = numpy_3d_array_to_tensor(flow).contiguous().to(device)
    
                # Warp frame b
                warp_comp_frame_b_t = flow_warping(comp_frame_b_t, flow_t)
                warp_comp_frame_b = tensor2img(warp_comp_frame_b_t)
    
                # Compute warping error
                occ_mask = read_img(os.path.join(occ_dir, '%05d.png' % (t - 1)))
                no_occ_mask = 1 - occ_mask
                no_occ_mask = np.where(no_occ_mask, 1., np.nan)

                # Compute warping error
                diff = np.multiply(warp_comp_frame_b - comp_frame_a, no_occ_mask)
                cur_warp_error_values.append(np.nanmean(np.square(diff)))
    
                comp_frame_a = comp_frame_b

            warp_error_all[v] = np.nanmean(cur_warp_error_values)
            self.send_update_msg(1)
    
        self.send_result_msg(warp_error_all)
