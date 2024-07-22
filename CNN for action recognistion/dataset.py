import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2  # OpenCV for video and image processing

class MiniUCFRGBDataset(Dataset):
    """
    Custom dataset for handling RGB video data for action recognition tasks.
    """
    def __init__(self, data_root, list_file, classes_root, transform):
        """
        Initializes the dataset with file paths and a transform.
        :param data_root: Directory containing the video files.
        :param list_file: File containing the list of video file names.
        :param classes_root: File containing class labels and corresponding indices.
        :param transform: Transformations to be applied on each frame of the video.
        """
        self.data_root = data_root
        self.transform = transform
        with open(list_file, 'r') as f:
            self.video_list = [x.strip() for x in f.readlines()]
        with open(classes_root, 'r') as f:
            classes = [x.strip() for x in f.readlines()]
        self.class_dict = {c.split(' ')[1]: int(c.split(' ')[0]) for c in classes}

    def __len__(self):
        """
        Returns the total number of videos in the dataset.
        """
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        Retrieves and transforms the video and its label by index.
        :param idx: Index of the video in the list.
        """
        video_path = os.path.join(self.data_root, self.video_list[idx] + '.avi')
        frames = self.load_video_frames(video_path)
        frames = self.transform(frames)
        label = self.get_label_from_path(self.video_list[idx])
        return frames, label

    def load_video_frames(self, video_path):
        """
        Loads video frames using OpenCV.
        :param video_path: Path to the video file.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def get_label_from_path(self, path):
        """
        Extracts the class label from the video file path.
        :param path: Path of the video file.
        """
        class_name = path.split('/')[0]
        return self.class_dict[class_name]
