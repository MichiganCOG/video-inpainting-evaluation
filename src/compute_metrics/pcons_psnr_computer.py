import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from .metric_computer import MetricComputer
from ..common_util.image import get_comp_frame, pil_rgb_to_numpy


class PConsPSNRComputer(MetricComputer):

    def compute_metric(self):
        self.send_work_count_msg(len(self.opts.video_names))
    
        pcons_psnr_all = np.zeros(len(self.opts.video_names))
        ### initialize RNG for patch-based consistency measure
        rng_psnr = np.random.default_rng(1234)
        for v in range(len(self.opts.video_names)):
    
            video_name = self.opts.video_names[v]
            num_frames = self.opts.video_frame_counts[v]
    
            comp_frame_a = get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, 0)
            comp_frame_a = pil_rgb_to_numpy(comp_frame_a)
    
            for t in range(1, num_frames):
    
                comp_frame_b = get_comp_frame(self.opts.gt_root, self.opts.pred_root, video_name, t)
                comp_frame_b = pil_rgb_to_numpy(comp_frame_b)
    
                ### measure video consistency by finding random patch in next frame
                a_sy = rng_psnr.integers(comp_frame_a.shape[0] - self.opts.sim_cons_ps)
                a_sx = rng_psnr.integers(comp_frame_a.shape[1] - self.opts.sim_cons_ps)
                comp_frame_a_patch = comp_frame_a[a_sy:a_sy + self.opts.sim_cons_ps, a_sx:a_sx + self.opts.sim_cons_ps]
                best_patch_psnr = 0.0
                best_b_sy = None
                best_b_sx = None
                for b_sy in range(a_sy - self.opts.sim_cons_sw, a_sy + self.opts.sim_cons_sw):
                    for b_sx in range(a_sx - self.opts.sim_cons_sw, a_sx + self.opts.sim_cons_sw):
                        comp_frame_b_patch = comp_frame_b[b_sy:b_sy + self.opts.sim_cons_ps,
                                                          b_sx:b_sx + self.opts.sim_cons_ps]
                        if comp_frame_a_patch.shape != comp_frame_b_patch.shape:
                            # Invalid patch at given location in comp_frame_b, so skip
                            continue
                        patch_psnr = psnr(comp_frame_a_patch, comp_frame_b_patch)
                        if patch_psnr > best_patch_psnr:
                            best_patch_psnr = patch_psnr
                            best_b_sy = b_sy
                            best_b_sx = b_sx
    
                best_comp_frame_b_patch = comp_frame_b[best_b_sy:best_b_sy + self.opts.sim_cons_ps,
                                                       best_b_sx:best_b_sx + self.opts.sim_cons_ps]
                pcons_psnr_all[v] += psnr(comp_frame_a_patch, best_comp_frame_b_patch) / (num_frames - 1)
    
                comp_frame_a = comp_frame_b
    
            self.send_update_msg(1)
    
        self.send_result_msg(pcons_psnr_all)
