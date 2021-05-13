import argparse
import os
import sys
from datetime import datetime
from multiprocessing import Queue
from time import time

import numpy as np
from tqdm import tqdm

from ..compute_metrics import *
from ..common_util.misc import get_all_frame_paths, get_video_names_and_frame_counts, LockedFile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

METRIC_NAME_TO_COMPUTER_MAP = {
    'fid': FIDComputer,
    'lpips': LPIPSComputer,
    'pcons_psnr': PConsPSNRComputer,
    'pcons_psnr_mask': PConsPSNRMaskComputer,
    'pcons_ssim': PConsSSIMComputer,
    'psnr': PSNRComputer,
    'pvcs': PVCSComputer,
    'ssim': SSIMComputer,
    'vfid': VFIDComputer,
    'vfid_clips': VFIDClipsComputer,
    'warp_error': WarpErrorComputer,
    'warp_error_mask': WarpErrorMaskComputer,
}


def main():
    print(str(datetime.now()))
    start_time = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True, help='Where to store the performance values')
    parser.add_argument('--gt_root', type=str, required=True, help='Path where all GT and mask frames are stored')
    parser.add_argument('--pred_root', type=str, required=True, help='Path where all prediction frames are stored')
    parser.add_argument('--eval_feats_root', type=str, required=True,
                        help='Path containing pre-processed data required for evaluation')

    parser.add_argument('--sim_cons_search_width', type=int, default=20, dest='sim_cons_sw',
                        help='The maximum distance from the source patch for consistency similarity')
    parser.add_argument('--sim_cons_patch_size', type=int, default=50, dest='sim_cons_ps',
                        help='The size of the patches used for consistency similarity')
    parser.add_argument('--max_num_videos', type=int, default=None, help='How many videos to evaluate on')
    parser.add_argument('--update', action='store_true',
                        help='Flag to enable updating individual values of an existing npz file')
    parser.add_argument('--log_path', type=str, default=os.devnull,
                        help='Path for storing output and error messages from worker processes')

    include_exclude_args = parser.add_mutually_exclusive_group()
    include_exclude_args.add_argument('--include', type=str, nargs='+', choices=METRIC_NAME_TO_COMPUTER_MAP.keys(),
                                      help='Metrics to compute')
    include_exclude_args.add_argument('--exclude', type=str, nargs='+', choices=METRIC_NAME_TO_COMPUTER_MAP.keys(),
                                      help='Metrics to not compute')

    opts = parser.parse_args()
    print(opts)

    # Create worker output log file
    worker_log_file = LockedFile(opts.log_path, 'w')

    # Determine metrics to compute
    if opts.include:
        for x in opts.include:
            assert x in METRIC_NAME_TO_COMPUTER_MAP, 'Metric {} is not supported'.format(x)
        include_list = opts.include
    else:
        # Compute all metrics by default
        include_list = [k for k in METRIC_NAME_TO_COMPUTER_MAP]
        # Remove metrics if exclude is specified
        if opts.exclude:
            include_list = [x for x in filter(lambda x: x not in opts.exclude, include_list)]

    # Get video info and ensure that frame counts match between gt, mask, and pred frames
    opts.video_names, opts.video_frame_counts = get_video_names_and_frame_counts(opts.gt_root, opts.max_num_videos)
    for v, video_name in enumerate(opts.video_names):
        mask_frame_list = get_all_frame_paths(opts.gt_root, video_name, 'mask')
        pred_frame_list = get_all_frame_paths(opts.pred_root, video_name, 'pred')
        assert opts.video_frame_counts[v] == len(mask_frame_list)
        assert opts.video_frame_counts[v] == len(pred_frame_list)

    # Set paths to pre-processed data
    opts.gt_fid_feats_path = os.path.join(opts.eval_feats_root, 'fid.npy')
    opts.gt_vfid_feats_path = os.path.join(opts.eval_feats_root, 'vfid.npy')
    opts.gt_vfid_clips_feats_path = os.path.join(opts.eval_feats_root, 'vfid_clips.npy')
    # Use the GPU
    opts.cuda = True

    # Run separate processes to compute each metric
    final_values = {}
    metric_proc_info = {}
    queue = Queue()
    # Keep track of running and failed processes
    remaining_metrics_count = len(include_list)
    failed_metric_names = []

    for i, metric_name in enumerate(include_list):
        # Run the child process
        process_class = METRIC_NAME_TO_COMPUTER_MAP[metric_name]
        process = process_class(opts, queue, worker_log_file)
        metric_prog_bar = tqdm(desc=metric_name, position=i, file=sys.stdout)
        process.start()
        # Store info needed to recover results
        metric_proc_info[process.name] = (process, metric_prog_bar, metric_name)

    while remaining_metrics_count > 0:
        item_type, process_name, data = queue.get()  # Format: (type, metric, data)
        process, metric_prog_bar, metric_name = metric_proc_info[process_name]
        if item_type == 'count':
            work_count = data
            metric_prog_bar.reset(total=work_count)
        elif item_type == 'update':
            update_count = data
            metric_prog_bar.update(update_count)
        elif item_type == 'result':
            result = data
            final_values[metric_name] = result
            process.join()
            remaining_metrics_count -= 1
        elif item_type == 'exception':
            failed_metric_names.append(metric_name)
            process.join()
            remaining_metrics_count -= 1

    worker_log_file.close()
    for _, metric_prog_bar, _ in metric_proc_info.values():
        metric_prog_bar.close()

    for metric_name in include_list:
        if metric_name in final_values:
            value = final_values[metric_name]
            print('Mean {} = {}'.format(metric_name, value.mean() if hasattr(value, 'mean') else value))

    output_path_dirname = os.path.dirname(opts.output_path)
    if output_path_dirname and not os.path.isdir(output_path_dirname):
        os.makedirs(output_path_dirname)

    if opts.update:
        if not os.path.isfile(opts.output_path):
            raise ValueError('Tried to update nonexistent file at {}'.format(opts.output_path))
        updated_values = dict(np.load(opts.output_path))
        if not np.array_equal(updated_values['clip_labels'], np.array(opts.video_names)):
            raise RuntimeError('Old and new video names do not match.\nExpected: {}\nActual: {}'
                               .format(list(updated_values['clip_labels']), opts.video_names))
        for k, v in final_values.items():
            updated_values[k] = v
        np.savez(opts.output_path, **updated_values)
    else:
        # Newly save or completely overwrite output path
        np.savez(opts.output_path, clip_labels=opts.video_names, **final_values)

    print(str(datetime.now()))
    end_time = time()
    print('Run time was {:.2f} minutes'.format((end_time - start_time) / 60))

    if len(failed_metric_names) > 0:
        print('The processes for some metrics failed: {}'.format(', '.join(failed_metric_names)))
        sys.exit(len(failed_metric_names))


if __name__ == "__main__":
    main()
