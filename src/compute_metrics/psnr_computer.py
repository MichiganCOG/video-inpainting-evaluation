import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from .metric_computer import MetricComputer
from ..common_util.image import get_gt_frame, get_comp_frame, pil_rgb_to_numpy


class PSNRComputer(MetricComputer):

    def compute_metric(self):
        self.send_work_count_msg(len(self.opts.video_names))

        psnr_all = np.zeros(len(self.opts.video_names))
        for v in range(len(self.opts.video_names)):
            video_name = self.opts.video_names[v]
            num_frames = self.opts.video_frame_counts[v]

            for t in range(num_frames):
                cur_gt_frame = get_gt_frame(self.opts.gt_root, video_name, t)
                cur_gt_frame = pil_rgb_to_numpy(cur_gt_frame)
                cur_comp_frame = get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, t)
                cur_comp_frame = pil_rgb_to_numpy(cur_comp_frame)
                psnr_all[v] += psnr(cur_gt_frame, cur_comp_frame, data_range=1) / num_frames

            self.send_update_msg(1)

        self.send_result_msg(psnr_all)
