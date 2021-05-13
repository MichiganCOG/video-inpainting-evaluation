import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from glob import glob
from multiprocessing import RLock


def makedirs(path):
    """Constructs a folder at the given path if it does not exist; otherwise, does nothing. NOT THREAD-SAFE."""
    if not os.path.isdir(path):
        os.makedirs(path)


def get_all_frame_paths(videos_root, video_name, suffix):
    """Gives a list of all frame paths for the given video and frame type.

    :param videos_root: Root directory containing all video frame directories
    :param video_name: The name of the video, or "*" to select all videos under video_root
    :param suffix: The suffix for the frame type. Can be "gt", "mask", or "pred"
    :return: list
    """
    if suffix not in ['gt', 'mask', 'pred']:
        raise ValueError(f'Unsupported suffix {suffix}')

    file_path_pattern = os.path.join(videos_root, video_name, f'frame_*_{suffix}.*')
    matched_files = sorted(glob(file_path_pattern))

    return matched_files


def get_video_names_and_frame_counts(video_frame_root_path, max_num_videos):
    # Determine the videos to evaluate
    video_names = sorted(os.listdir(video_frame_root_path))
    if max_num_videos is not None:
        video_names = video_names[:max_num_videos]
    # Count number of frames per video
    video_frame_counts = [None for _ in video_names]
    for v, video_name in enumerate(video_names):
        gt_frame_list = get_all_frame_paths(video_frame_root_path, video_name, 'gt')
        video_frame_counts[v] = len(gt_frame_list)
    return video_names, video_frame_counts


@contextmanager
def no_stdout_stderr(f, redirect=True):
    """Provides a context manager in which stderr and stdout are redirected to the given file.

    :param f: The file handle used for redirecting stderr and stdout
    :param redirect: Whether to perform redirection
    """
    if redirect:
        with redirect_stderr(f), redirect_stdout(f):
            yield
    else:
        yield


class LockedFile(object):

    def __init__(self, *args, **kwargs):
        self.file = open(*args, **kwargs)
        self.lock = RLock()


    def __getattr__(self, item):
        return getattr(self.file, item)


    def write(self, *args, **kwargs):
        with self.lock:
            self.file.write(*args, **kwargs)
            self.file.flush()
