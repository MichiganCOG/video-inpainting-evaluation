import numpy as np
import torch
from torch.nn import functional as F

from ..common_util.image import pil_rgb_to_numpy, numpy_image_to_tensor


def extract_video_frame_features(model, get_frame_fn, video_names, video_frame_counts, update_progress_cb=None):
    """

    :param model:
    :param get_frame_fn:
    :param video_names:
    :param video_frame_counts:
    :param update_progress_cb:
    :return: VxF np.float32 array
    """
    # Extract descriptors of all frames
    frame_features = []

    for video_name, num_frames in zip(video_names, video_frame_counts):
        # Go through each frame
        for t in range(num_frames):
            cur_frame = get_frame_fn(video_name, t)
            cur_frame_np = pil_rgb_to_numpy(cur_frame)
            cur_frame_t = numpy_image_to_tensor(cur_frame_np).cuda()
            pred = model(cur_frame_t)[0]
            # Obtain global spatially-pooled features as single vector
            features = F.adaptive_avg_pool2d(pred, output_size=(1, 1)).squeeze()
            features_cpu_np = features.cpu().numpy()
            frame_features.append(features_cpu_np)
        # Update work count
        if update_progress_cb is not None:
            update_progress_cb()

    frame_features = np.stack(frame_features)
    return frame_features


def extract_video_clip_features(model, get_frame_fn, video_names, video_frame_counts, update_progress_cb=None,
                                window_size=10):
    """

    :param model:
    :param get_frame_fn:
    :param video_names:
    :param video_frame_counts:
    :param update_progress_cb:
    :param window_size:
    :return: NxF np.float32 array, where N = number of 10-frame clips in the dataset
    """
    clip_features = []

    for video_name, num_frames in zip(video_names, video_frame_counts):
        num_windows = max(1, num_frames - window_size + 1)

        # Generate frames for first window
        cur_video_frames = []
        for t in range(window_size):
            if t < num_frames:
                cur_video_frame = get_frame_fn(video_name, t)
                cur_video_frames.append(_convert_frame_vfid(cur_video_frame))
            else:
                cur_video_frames.append(cur_video_frames[-1])

        for t in range(num_windows):
            # Pass current window through model
            model_input = torch.stack(cur_video_frames, dim=2).cuda()
            pred = model.extract_features(model_input, target_endpoints='Logits')
            # Obtain global spatially-pooled features as single vector
            features = pred[0].squeeze()  # 1024
            clip_features.append(features.cpu().numpy())

            # Prepare next window
            if t + window_size < num_frames:
                # Remove first frame from array
                del cur_video_frames[0]
                # Append new frame
                cur_video_frame = get_frame_fn(video_name, t + window_size)
                cur_video_frames.append(_convert_frame_vfid(cur_video_frame))

        # Update work count
        if update_progress_cb is not None:
            update_progress_cb()

    clip_features = np.stack(clip_features)
    return clip_features


def extract_video_features(model, get_frame_fn, video_names, video_frame_counts, update_progress_cb=None):
    """

    :param model:
    :param get_frame_fn:
    :param video_names:
    :param video_frame_counts:
    :param update_progress_cb:
    :return: VxF np.float32 array
    """
    clip_features = []

    for video_name, num_frames in zip(video_names, video_frame_counts):
        cur_video_frames = []
        for t in range(num_frames):
            cur_video_frame = get_frame_fn(video_name, t)
            cur_video_frames.append(_convert_frame_vfid(cur_video_frame))

        # Pass current window through model
        model_input = torch.stack(cur_video_frames, dim=2).cuda()
        pred = model.extract_features(model_input, target_endpoints='Logits')
        # Obtain global spatially-pooled features as single vector
        features = pred[0].squeeze()  # 1024
        clip_features.append(features.cpu().numpy())

        # Update work count
        if update_progress_cb is not None:
            update_progress_cb()

    clip_features = np.stack(clip_features)
    return clip_features


def _convert_frame_vfid(image):
    ret = pil_rgb_to_numpy(image)
    ret = numpy_image_to_tensor(ret)

    return ret
