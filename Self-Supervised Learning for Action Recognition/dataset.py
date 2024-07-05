"""
"""

import dataclasses as dt
import logging
import os
import random
import typing as t

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io.video import read_video, read_video_timestamps
import torchvision.transforms as tvt
from tqdm import tqdm
import glob
import pathlib as pl
# import cv2


class ConvertBHWCtoBCHW(nn.Module):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


training_transforms = tvt.Compose(
    [
        ConvertBHWCtoBCHW(),
        tvt.ConvertImageDtype(torch.float32),
        tvt.RandomCrop(224),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ConvertBCHWtoCBHW(),
    ]
)

training_transforms_with_flip = tvt.Compose(
    [
        ConvertBHWCtoBCHW(),
        tvt.ConvertImageDtype(torch.float32),
        tvt.RandomHorizontalFlip(),
        tvt.RandomCrop(224),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ConvertBCHWtoCBHW(),
    ]
)

testing_transforms = tvt.Compose(
    [
        ConvertBHWCtoBCHW(),
        tvt.ConvertImageDtype(torch.float32),
        tvt.CenterCrop(224),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ConvertBCHWtoCBHW(),
    ]
)


class ResNet3D(Dataset):
    """ """

    def __init__(
        self,
        root_path: str = "data/",
        video_path: str = "data/mini_UCF/",
        classes_path: str = "data/classes.txt",
        video_list_name: str = "train.txt",
        clip_length: int = 64,
        mode: str = "train",  # "train" and "eval"
        
        num_clips_per_video: int = 1,
        transforms: t.Optional[tvt.Compose] = training_transforms,
        origin_transforms: t.Optional[tvt.Compose] = testing_transforms
    ) -> None:
        """ """
        super().__init__()
        # validate the inputs

        assert video_list_name[video_list_name.rfind('/')+1:] in ["train.txt", "validation.txt"]
        assert mode in ["train", "eval"]
        assert num_clips_per_video >= 1
        if mode == "train":  # it doesn't make sense to use more than 1 clip in train
            assert num_clips_per_video == 1

        # expanduser is used to change "~/" into the absolute path.
        self.root_path = os.path.expanduser(root_path)
        self.video_path = os.path.expanduser(video_path)
        self.classes_path = os.path.expanduser(classes_path)
        self.video_list_path = os.path.join(self.root_path, video_list_name)
        self.clip_length = clip_length
        self.mode = mode
        self.num_clips_per_video = num_clips_per_video
        self.transforms = transforms
        self.original_transforms = origin_transforms

        # 1. gather list of avi files
        ## read class label list
        self.class_name_to_ind = {}
        self.class_ind_to_name = {}
        with open(self.classes_path) as f:
            lines = [x.strip() for x in f.read().split("\n") if x.strip()]
            for ci, cn in [l.split(" ") for l in lines]:
                self.class_name_to_ind[cn] = int(ci)
                self.class_ind_to_name[int(ci)] = cn
        ## read video list
        with open(self.video_list_path) as f:
            video_list_lines = [x.strip() for x in f.read().split("\n") if x.strip()]
        ## open each video and find out its frame numbers
        self.video_ids = []
        self.absolute_video_paths = []
        self.video_frame_numbers = []
        print("Loading frame numbers in videos ...")
        for current_vid in tqdm(video_list_lines):
            current_video_path = os.path.join(self.video_path, f"{current_vid}.avi")

            if os.path.exists(current_video_path):
                current_frame_numbers, current_video_fps = read_video_timestamps(
                    current_video_path
                )
                self.video_frame_numbers.append(current_frame_numbers)
                self.video_ids.append(current_vid)
                self.absolute_video_paths.append(current_video_path)
            else:
                logging.warn(f"Video not found: {current_video_path}")

    def _get_class_ind_from_video_id(self, video_id: str) -> int:
        video_class_name, video_name = video_id.split("/")
        return self.class_name_to_ind[video_class_name]

    def num_classes(self) -> int:
        return len(self.class_name_to_ind)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.video_ids)
        else:
            return self.num_clips_per_video * len(self.video_ids)

    def __getitem__(self, index: int) -> t.Dict[str, Tensor]:
        # if self.mode == "train":
        video_index = index

        video_frame_numbers = self.video_frame_numbers[video_index]

        possible_clip_start_frame_min = 1
        possible_clip_start_frame_max = (
            video_frame_numbers[-1] - self.clip_length + 1
        )
        if possible_clip_start_frame_max > possible_clip_start_frame_min:
            clip_start_frame_number = random.choice(
                list(
                    range(
                        possible_clip_start_frame_min,
                        possible_clip_start_frame_max + 1,
                    )
                )
            )
        else:
            clip_start_frame_number = 1

        # -1 because it is inclusive.
        clip_end_frame_number = clip_start_frame_number + self.clip_length - 1

        # elif self.mode == "eval":
        #     # if `num_clips_per_video` is greater than 1, then index can be larger than
        #     # the number of video_ids we have.
        #     video_index = int(index / self.num_clips_per_video)
        #     # what is the clip index for this video_index
        #     clip_index = index % self.num_clips_per_video

        #     video_frame_numbers = self.video_frame_numbers[video_index]

        #     # additional number of frames and step size
        #     additional_number_of_frames = len(video_frame_numbers) - self.clip_length
        #     step_size = int(additional_number_of_frames / self.num_clips_per_video)

        #     clip_start_frame_number = clip_index * step_size
        #     clip_end_frame_number = clip_start_frame_number + self.clip_length
        # else:
        #     raise Exception(f"Invalid mode ({self.mode})")

        # load the video frames for the clip
        clip_video_frames, clip_audio_frames, clip_info = read_video(
            self.absolute_video_paths[video_index],
            start_pts=clip_start_frame_number,
            end_pts=clip_end_frame_number,
            pts_unit="sec"
        )  # clip_video_frames: [T x H x W x 3]

        snippet_video_frames = clip_video_frames[::2, :, :, :]  # [T/2 x H x W x C]

        if snippet_video_frames.shape[0] != int(self.clip_length / 2):
            padded_data = torch.zeros(
                (
                    int(self.clip_length / 2),
                    snippet_video_frames.shape[1],
                    snippet_video_frames.shape[2],
                    snippet_video_frames.shape[3],
                ),
                dtype=snippet_video_frames.dtype,
                device=snippet_video_frames.device,
            )
            temporal_size = min(
                int(self.clip_length / 2), snippet_video_frames.shape[0]
            )

            padded_data[:temporal_size, :, :, :] = snippet_video_frames[
                :temporal_size, :, :, :
            ]
            snippet_video_frames = padded_data
        
        
        if self.mode == "train":

            return {
                "id": self.video_ids[video_index],
                "original": snippet_video_frames,
                "input_1": self.transforms[0](snippet_video_frames),  # [C, T, H, W]
                "input_2": self.transforms[1](snippet_video_frames),  # [C, T, H, W]
                "label": torch.tensor(
                    self._get_class_ind_from_video_id(self.video_ids[video_index])
                ),
            }
        else:
            return {
                "id": self.video_ids[video_index],
                "original": self.original_transforms(snippet_video_frames),
                "label": torch.tensor(
                    self._get_class_ind_from_video_id(self.video_ids[video_index])
                ),
            }

