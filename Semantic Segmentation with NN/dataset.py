import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, num_classes=20):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.num_classes = num_classes
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        label = np.array(label)
        label = torch.from_numpy(label).long()

        # Ensure label is 2D before applying one_hot
        label = torch.nn.functional.one_hot(label, num_classes=self.num_classes).permute(2, 0, 1).float()
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
