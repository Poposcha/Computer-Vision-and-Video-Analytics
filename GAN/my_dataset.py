from torch.utils.data import Dataset
import torch

class GeneratedDataset(Dataset):
    def __init__(self, generator, length, image_size):
        self.generator = generator
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        z = torch.randn(1, 3, self.image_size, self.image_size).to(device)
        gen_img = self.generator(z).squeeze(0)
        return (gen_img, 0.0)
    
class OneClassDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        label = 1  # Change the label to 1
        return image, label
