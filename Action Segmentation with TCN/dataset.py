import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class VideoDataset(Dataset):
    def __init__(self, features_dir, groundtruth_dir, mapping_file, bundle_file):
        self.features_dir = features_dir
        self.groundtruth_dir = groundtruth_dir

        # Load the mapping file
        self.action_to_id = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                action_id, action = line.strip().split()
                self.action_to_id[action] = int(action_id)

        # Load the bundle file
        with open(bundle_file, 'r') as f:
            self.video_files = [line.strip()[:-4] for line in f]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        import os

        # Load features
        feature_path = os.path.join(self.features_dir, video_file + '.npy')
        features = np.load(feature_path)
        
        # Load ground truth
        groundtruth_path = os.path.join(self.groundtruth_dir, video_file + '.txt')
        with open(groundtruth_path, 'r') as f:
            groundtruth = [self.action_to_id[line.strip()] for line in f]
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(groundtruth, dtype=torch.long)
    

def collate_fn(batch):
    features, labels = zip(*batch)

    # Find the max length in this batch
    max_length = max(f.shape[1] for f in features)
    
    # Pad features and labels to the max length
    padded_features = [torch.nn.functional.pad(f, (0, max_length - f.shape[1])) for f in features]
    padded_labels = [torch.nn.functional.pad(l, (0, max_length - l.shape[0]), value=0) for l in labels]
    
    return torch.stack(padded_features), torch.stack(padded_labels)