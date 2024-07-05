import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class MyPermute:
    def __init__(self, change_order):
        self.change_order = change_order
    
    def __call__(self, video):
        return video.permute(self.change_order) 

class RandomTemporalCrop:
    '''
        Random temporal crop video (num_frames x height x width x channels) to (num_frames_crop x height x width x channels)
    '''
    def __init__(self, num_frames_crop=25):
        self.num_frames_crop = num_frames_crop
    def __call__(self, frames):
        return frames[np.random.choice(frames.size(0), self.num_frames_crop, replace=False), ...]

class TemporalViewCrop:
    '''
        Multi view crop video (num_frames x height x width x channels) to (num_views, num_frames x height x width x channels)
    '''
    def __init__(self, num_views=4, num_frames_crop=32):
        self.num_views = num_views
        self.num_frames_crop = num_frames_crop
    def __call__(self, video):
        step_d = video.size(0) // self.num_views
        view_frames = self.num_frames_crop // self.num_views
        transformed_video = torch.zeros(self.num_views, view_frames, video.shape[1], video.shape[2], video.shape[3])
        for i in range(self.num_views):
            indices = np.arange(i*step_d, i*step_d + view_frames)
            transformed_video[i] = video[indices, ...]
        
        return transformed_video
    
class MyRandomCrop_4dim:
    '''
        Crop frame-wise of video (num_frames x height x width x channels) to (num_frames x new_height x new_width x channels)
    '''
    def __init__(self, output):
        self.output = output
    def __call__(self, video):
        transformed_video = torch.zeros(video.shape[0], video.shape[3], self.output[0], self.output[1])
        
        for i in range(video.shape[0]):
            transformed_video[i, ...] = RandomCrop(self.output)(video.permute(0, 3, 1, 2)[i, ...])
        
        transformed_video = transformed_video.permute(0, 2, 3, 1)
        return transformed_video
    
class MyToTensor:
    '''
        Tensor ndarray (num_frames x height x width x channels) with torch.tensor with same shape
    '''
    def __call__(self, video):
        new_tensor = torch.zeros(video.shape)

        for i in range(video.shape[0]):
            new_tensor[i, ...] = torch.from_numpy(video[i, ...])

        return new_tensor
