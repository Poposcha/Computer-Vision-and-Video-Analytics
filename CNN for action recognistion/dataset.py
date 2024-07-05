import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import cv2

class MiniUCFRGBDataset(Dataset):
    def __init__(self, data_root, list_file, classes_root, transform):
        self.data_root = data_root
        self.transform = transform
        
        with open(list_file, 'r') as f:
            self.video_list = f.readlines()
        
        self.video_list = [x.strip() for x in self.video_list]

        with open(classes_root, 'r') as f:
            classes = f.readlines()
        classes = [x.strip() for x in classes]
        self.class_dict = {c.split(' ')[1]: int(c.split(' ')[0]) for c in classes}
               
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.data_root, self.video_list[idx] + '.avi')
        frames = self.load_video_frames(video_path)
        
        frames = self.transform(frames)

        label = self.get_label_from_path(self.video_list[idx])
        
        return frames, label
    
    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        frames = np.array(frames)
        return frames
    
    def get_label_from_path(self, path):
        class_name = path.split('/')[0]
        return self.class_dict[class_name]

