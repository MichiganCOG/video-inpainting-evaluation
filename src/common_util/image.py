# Parts adapted from https://github.com/phoenix104104/fast_blind_video_consistency

import cv2
import numpy as np
import os
import torch
from glob import glob

from PIL import Image


def read_img(filename, grayscale=False, mean=0.0, std=1.0):
    """

    :param filename:
    :param grayscale:
    :param mean:
    :param std:
    :return: HxWxC np.float32 array with mode RGB and range [0, 1] subject to mean and std
    """

    ## read image and convert to RGB in [0, 1]

    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename)

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        img = img[:, :, ::-1] ## BGR to RGB

    img = np.float32(img) / 255.0
    img = (img - mean) / std

    return img


def numpy_image_to_tensor(img):
    """Converts an RGB OpenCV-like image to a PyTorch-like image.

    :param img: HxWxC np.float32 array with range [0, 1]
    :return: 1xCxHxW FloatTensor with range [0, 1]
    """
    assert img.min() >= 0
    assert img.max() <= 1

    return numpy_3d_array_to_tensor(img)


def numpy_3d_array_to_tensor(arr):
    """Converts a multichannel np.float32 array to a single-batch PyTorch FloatTensor with prioritized channel dim.

    :param arr: HxWxC np.float32 array
    :return: 1xCxHxW FloatTensor
    """
    _check_numpy_3d_float_array(arr)

    arr_reshaped = np.expand_dims(arr.transpose(2, 0, 1), axis=0)
    tensor = torch.from_numpy(arr_reshaped)

    return tensor


def _check_numpy_3d_float_array(image_np):
    """Asserts that the given argument is a well-formed 3D np.float32 array.

    :param image_np: HxWxC np.float32 array with range [0, 1]
    """
    assert isinstance(image_np, np.ndarray)
    assert image_np.ndim == 3
    assert image_np.dtype == np.float32


def tensor2img(img_t):

    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


def rotate_image(img, degree, interp=cv2.INTER_LINEAR):

    height, width = img.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, degree, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    img_out = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), flags=interp+cv2.WARP_FILL_OUTLIERS)

    return img_out


def save_img(img, filename, binary=False):

    print("Save %s" %filename)

    if img.ndim == 3:
        img = img[:, :, ::-1] ### RGB to BGR

    ## clip to [0, 1]
    img = np.clip(img, 0, 1)

    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)

    if binary:
        cv2.imwrite(filename, img, (cv2.IMWRITE_PNG_BILEVEL, 1))
    else:
        cv2.imwrite(filename, img)


def get_gt_frame(videos_root, video_name, frame_index):
    """Returns the ground-truth frame corresponding to the given video and frame index as an RGB PIL Image.

    :param videos_root: Root directory containing input video frame directories
    :param video_name: The name of a video
    :param frame_index: The frame index
    :return: PIL Image
    """
    file_path_pattern = os.path.join(videos_root, video_name, f'frame_{frame_index:04d}_gt.*')
    file_path = _get_single_path_from_pattern(file_path_pattern)
    image = Image.open(file_path)
    _check_pil_image_mode(image, 'RGB')

    return image


def _check_pil_image_mode(pil_image, format):
    """Checks that the given image matches the expected format.

    :param pil_image: PIL Image
    :param format: PIL mode string (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)
    """

    assert isinstance(pil_image, Image.Image)
    assert pil_image.mode == format


def get_mask_frame(videos_root, video_name, frame_index):
    """Returns the mask frame corresponding to the given video and frame index as a binary PIL Image.

    :param videos_root: Root directory containing input video frame directories
    :param video_name: The name of a video
    :param frame_index: The frame index
    :return: PIL Image
    """
    file_path_pattern = os.path.join(videos_root, video_name, f'frame_{frame_index:04d}_mask.*')
    file_path = _get_single_path_from_pattern(file_path_pattern)
    image = Image.open(file_path)
    _check_pil_image_mode(image, '1')

    return image


def get_pred_frame(videos_root, video_name, frame_index):
    """Returns the full-frame prediction frame corresponding to the given video and frame index as an RGB PIL Image.

    :param videos_root: Root directory containing output video frame directories
    :param video_name: The name of a video
    :param frame_index: The frame index
    :return: PIL Image
    """
    file_path_pattern = os.path.join(videos_root, video_name, f'frame_{frame_index:04d}_pred.*')
    file_path = _get_single_path_from_pattern(file_path_pattern)
    image = Image.open(file_path)
    _check_pil_image_mode(image, 'RGB')

    return image


def _get_single_path_from_pattern(file_path_pattern):
    """Returns exactly one path that matches the given glob pattern. Fails if more or fewer than one path is found.

    :param file_path_pattern: A glob pattern that matches the path of the desired file
    :return: The path matching the given pattern
    """

    matched_files = glob(file_path_pattern)
    if len(matched_files) == 0:
        raise ValueError(f'Failed to find any files matching pattern {file_path_pattern}')
    elif len(matched_files) > 1:
        raise ValueError(f'Found too many files matching pattern {file_path_pattern}')
    file_path = matched_files[0]
    return file_path


def get_comp_frame(videos_input_root, videos_output_root, video_name, frame_index):
    """Returns the composited prediction frame corresponding to the given video and frame index as an RGB PIL Image.

    A composited prediction is a prediction whose known values have been replaced with the ground truth. Formally, if
    G, M, P, and C correspond to the ground-truth image, mask, full-frame prediction, and composited prediction
    respectively, then

    C = M .* G + (1-M) .* P

    (M is 0 for unknown values; .* is broadcasted element-wise product).

    :param videos_input_root: Root directory containing input video frame directories
    :param videos_output_root: Root directory containing output video frame directories
    :param video_name: The name of a video
    :param frame_index: The frame index
    :return: PIL Image
    """
    gt_frame = get_gt_frame(videos_input_root, video_name, frame_index)
    mask_frame = get_mask_frame(videos_input_root, video_name, frame_index)
    pred_frame = get_pred_frame(videos_output_root, video_name, frame_index)
    comp_frame = Image.composite(gt_frame, pred_frame, mask_frame)

    return comp_frame


def pil_rgb_to_numpy(pil_image):
    """Converts an RGB PIL Image to an RGB OpenCV-like np.float32 array with range [0, 1].

    :param pil_image: PIL Image with mode RGB
    :return: HxWxC np.float32 array with mode RGB and range [0, 1]
    """
    _check_pil_image_mode(pil_image, 'RGB')
    ret = np.array(pil_image, dtype=np.float32) / 255

    return ret


def pil_binary_to_numpy(pil_image):
    """Converts a binary PIL Image to a grayscale OpenCV-like np.float32 array with range [0, 1].

    :param pil_image: PIL Image with mode 1
    :return: HxW np.float32 array with grayscale mode and range [0, 1]
    """
    _check_pil_image_mode(pil_image, '1')
    ret = np.array(pil_image, dtype=np.float32)

    return ret