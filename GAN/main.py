import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset

from models import Generator, Discriminator
from my_dataset import GeneratedDataset, OneClassDataset
from train import train_generator, train_discriminator
from fid import calculate_fid

def main():
    # Task 1. Unconditional image generation with GAN
    # Hyperparameters
    batch_size = 128
    image_size = 64
    num_epochs = 500
    lr = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data loading
    dataset = datasets.ImageFolder(root='data/data_gan',
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                               ]))
    
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Generator training
    gen_model = Generator().to(device)
    # gen_criterion = nn.MSELoss()
    # gen_optimizer = optim.Adam(gen_model.parameters(), lr=lr, betas=(0.5, 0.999))
    # gen_losses = train_generator(gen_model, train_loader, test_loader, num_epochs, gen_criterion, gen_optimizer, batch_size, device)

    # torch.save({
    #     'model_state_dict': gen_model.state_dict(),
    #     'optimizer_state_dict': gen_optimizer.state_dict(),
    #     'loss': gen_losses,
    # }, 'gen_model.pth')

    # if model is trained
    gen_w = torch.load("gen_model.pth")['model_state_dict']
    gen_model.load_state_dict(gen_w)

    # Discriminator data
    one_class_dataset = OneClassDataset(dataset)

    train_dataset = Subset(one_class_dataset, train_indices)
    test_dataset = Subset(one_class_dataset, test_indices)

    generated_train_dataset = GeneratedDataset(gen_model, len(train_loader), image_size)
    generated_test_dataset = GeneratedDataset(gen_model, len(test_loader), image_size)

    # Combine real and generated datasets for training and test sets
    combined_train_dataset = ConcatDataset([train_dataset, generated_train_dataset])
    combined_test_dataset = ConcatDataset([test_dataset, generated_test_dataset])

    # Create DataLoaders for the combined datasets
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)

    # Discriminator training
    disc_model = Discriminator().to(device)
    disc_criterion = nn.BCELoss()
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_acc_losses= train_discriminator(disc_model, combined_train_loader, combined_test_loader, num_epochs, disc_criterion, disc_optimizer, batch_size, device)
    
    # Fid score calculation
    image_size = 299 # InceptionV3 299x299 input size
    generated_train_dataset = GeneratedDataset(gen_model, len(train_loader), image_size)
    generated_train_loader = DataLoader(generated_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    fid_score = calculate_fid(train_loader, generated_train_loader, device)
    print(f"Fid score = {fid_score}")

if __name__ == '__main__':
    main()