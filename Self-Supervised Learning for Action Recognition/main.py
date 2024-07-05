import torch
import torch.nn as nn

from dataset import ResNet3D, training_transforms_with_flip, testing_transforms
from model import ResNet
import pytorchvideo.transforms as pv_transforms
from torchvision.transforms import Compose
# from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet18
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from train import train_model
from losses import NTXentLoss, InfoNCELoss

import warnings
warnings.filterwarnings("ignore")

# Visualize augmentations
def visualize_augmentations(dataset, num_examples=5):
        for i in range(0, num_examples*20, 20):
            print(dataset[i]["input_1"].shape)
            # print(dataset[i]['original'].shape)
            sample = dataset[i]
            augmented_sample_1 = sample['input_1']
            augmented_sample_2 = sample['input_2']
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(augmented_sample_1[0][0])  
            axs[0].set_title("Augmented 1")
            axs[1].imshow(augmented_sample_2[0][0])  
            axs[1].set_title("Augmented_2")

            plt.show()

def main():
    dataset_path = "data/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batch_size = 32
    validation_batch_size = 32
    clip_length = 64
    num_workers = 8
    num_epochs = 20
    lr = 1e-4
    wd = 1e-5

    # Augmentation with PyTorchVideo
    # Define the augmentations
    augmentations = [
        Compose([
            training_transforms_with_flip,
            # pv_transforms.RandomShortSideScale(min_size=64, max_size=256),
            pv_transforms.RandomResizedCrop(224, 224, (0.5, 0.6), (0.8, 0.4)),
            # pv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # pv_transforms.MixVideo(),
            # pv_transforms.RandAugment()
    ]),
        Compose([
            training_transforms_with_flip,
            pv_transforms.RandomShortSideScale(min_size=128, max_size=128),
            # pv_transforms.RandomResizedCrop(224, 224, (0.5, 0.6), (0.8, 0.4)),
            # pv_transforms.Normalize((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)),
            # pv_transforms.MixVideo(cutmix_prob=0.5, mixup_alpha=0.7, cutmix_alpha=0.8, label_smoothing=0.5, num_classes=25),
            # pv_transforms.functional.horizontal_flip_with_boxes(prob=0.5, boxes=15)
    ])]
        
    train_dataset = ResNet3D(
        mode="train",
        clip_length=clip_length,
        video_list_name="train.txt",
        transforms=augmentations,
    )

    validation_dataset = ResNet3D(
        mode='eval',
        clip_length=clip_length,
        video_list_name='validation.txt',
        transforms=testing_transforms,

    )

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


    model = resnet18(num_classes=train_dataset.num_classes()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion1  = NTXentLoss()
    criterion2 = InfoNCELoss()
    criterions = [criterion1, criterion2]

    pre_result = train_model(
        model, 
        train_dataloader, 
        validation_dataloader, 
        train_dataset, 
        validation_dataset, 
        criterions, 
        optimizer, 
        10, 
        device,
    )


    # Finetuning

    # Load pretrained model
    model = resnet18(num_classes=train_dataset.num_classes())
    model.load_state_dict(torch.load('3dresnet43.model'))
    model.fc = nn.Linear(model.fc.in_features, 25)  # Replace final layer for 25 classes
    model = model.to(device)

    tuned_result = train_model(
        model, 
        train_dataloader, 
        validation_dataloader, 
        train_dataset, 
        validation_dataset, 
        criterions, 
        optimizer, 
        num_epochs=20, 
    )

if __name__ == '__main__':
    main()