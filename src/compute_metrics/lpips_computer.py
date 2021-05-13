import os

import numpy as np
import torch

from .metric_computer import MetricComputer
from ..common_util.image import get_gt_frame, get_comp_frame, pil_rgb_to_numpy, numpy_image_to_tensor
from ..lpips.networks_basic import PNetLinAlex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))


class LPIPSComputer(MetricComputer):

    def compute_metric(self):
        self.send_work_count_msg(len(self.opts.video_names))

        torch.set_grad_enabled(False)
        device = torch.device('cuda:0')

        ## Initializing LPIPS model
        network = PNetLinAlex()
        snapshot_path = os.path.join(PROJ_DIR, 'pretrained_models', 'alex.pth')
        print('Loading model from: %s' % snapshot_path)

        network.load_state_dict(torch.load(snapshot_path, map_location=device), strict=False)
        network.eval()
        network.to(device)

        lpips_all = np.zeros(len(self.opts.video_names))
        for v in range(len(self.opts.video_names)):
            video_name = self.opts.video_names[v]
            num_frames = self.opts.video_frame_counts[v]

            for t in range(num_frames):
                cur_gt_frame = _convert_frame(get_gt_frame(self.opts.gt_root, video_name, t))
                cur_comp_frame = _convert_frame(get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, t))

                dist_t = network.forward(cur_gt_frame.to(device), cur_comp_frame.to(device))
                lpips_all[v] += dist_t.item() / num_frames

            self.send_update_msg(1)

        self.send_result_msg(lpips_all)


def _convert_frame(image):
    ret = pil_rgb_to_numpy(image)
    ret = numpy_image_to_tensor(ret)
    ret = ret * 2.0 - 1  # Map from [0, 1] to [-1, 1]

    return ret
