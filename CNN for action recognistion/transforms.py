import torch
from torchvision.transforms import Compose, RandomCrop
import numpy as np

class MyPermute:
    """
    Custom transformation to permute dimensions of a tensor.
    """
    def __init__(self, change_order):
        self.change_order = change_order

    def __call__(self, video):
        return video.permute(self.change_order)

class RandomTemporalCrop:
    """
    Randomly crops a video tensor along the temporal dimension.
    """
    def __init__(self, num_frames_crop=25):
        self.num_frames_crop = num_frames_crop

    def __call__(self, frames):
        return frames[np.random.choice(frames.size(0), self.num_frames_crop, replace=False), ...]

class TemporalViewCrop:
    """
    Crops video into multiple temporal views.
    """
    def __init__(self, num_views=4, num_frames_crop=32):
        self.num_views = num_views
        self.num_frames_crop = num_frames_crop

    def __call__(self, video):
        step_d = video.size(0) // self.num_views
        view_frames = self.num_frames_crop // self.num_views
        transformed_video = torch.zeros(self.num_views, view_frames, *video.shape[1:])
        for i in range(self.num_views):
            indices = np.arange(i * step_d, i * step_d + view_frames)
            transformed_video[i] = video[indices, ...]
        return transformed_video

class MyRandomCrop_4dim:
    """
    Applies a frame-wise random crop to a video tensor.
    """
    def __init__(self, output):
        self.output = output

    def __call__(self, video):
        transformed_video = torch.zeros(video.shape[0], *self.output, video.shape[3])
        for i in range(video.shape[0]):
            transformed_video[i, ...] = RandomCrop(self.output)(video[i, ...].permute(2, 0, 1)).permute(1, 2, 0)
        return transformed_video

class MyToTensor:
    """
    Converts a numpy ndarray to a PyTorch tensor.
    """
    def __call__(self, video):
        return torch.from_numpy(video)
