import os

import numpy as np
import torch

from .metric_computer import MetricComputer
from ..common_util.image import get_gt_frame, get_comp_frame, pil_rgb_to_numpy, numpy_image_to_tensor
from ..lpips.networks_basic import PNetLinI3D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

WINDOW_SIZE = 10


class PVCSComputer(MetricComputer):

    def compute_metric(self):
        self.send_work_count_msg(len(self.opts.video_names))

        torch.set_grad_enabled(False)
        device = torch.device('cuda:0')

        # Initialize PVCS model
        model = PNetLinI3D()
        weights = torch.load(os.path.join(PROJ_DIR, 'pretrained_models', 'rgb_imagenet.pt'))
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

        pvcs_all = np.zeros(len(self.opts.video_names))
        for v in range(len(self.opts.video_names)):
            video_name = self.opts.video_names[v]
            num_frames = self.opts.video_frame_counts[v]
            num_windows = max(1, num_frames - WINDOW_SIZE + 1)

            # Generate frames for first window
            cur_gt_frames = []
            cur_comp_frames = []
            for t in range(WINDOW_SIZE):
                if t < num_frames:
                    cur_gt_frame = get_gt_frame(self.opts.gt_root, video_name, t)
                    cur_comp_frame = get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, t)
                    cur_gt_frames.append(_convert_frame(cur_gt_frame))
                    cur_comp_frames.append(_convert_frame(cur_comp_frame))
                else:
                    # Pad end of video with copies of last frame if video is too short
                    cur_gt_frames.append(cur_gt_frames[-1])
                    cur_comp_frames.append(cur_gt_frames[-1])

            for t in range(num_windows):
                # Pass current window through model
                cur_Gs_t = torch.stack(cur_gt_frames, dim=2)
                cur_Os_t = torch.stack(cur_comp_frames, dim=2)
                dist_t = model.forward(cur_Gs_t.to(device), cur_Os_t.to(device))
                pvcs_all[v] += dist_t / num_windows

                # Prepare next window
                if t + WINDOW_SIZE < num_frames:
                    # Remove first frame from array
                    del cur_gt_frames[0]
                    del cur_comp_frames[0]
                    # Append new frame
                    cur_gt_frame = get_gt_frame(self.opts.gt_root, video_name, t + WINDOW_SIZE)
                    cur_comp_frame = get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, t + WINDOW_SIZE)
                    cur_gt_frames.append(_convert_frame(cur_gt_frame))
                    cur_comp_frames.append(_convert_frame(cur_comp_frame))

            self.send_update_msg(1)

        self.send_result_msg(pvcs_all)


def _convert_frame(image):
    ret = pil_rgb_to_numpy(image)
    ret = numpy_image_to_tensor(ret)

    return ret
