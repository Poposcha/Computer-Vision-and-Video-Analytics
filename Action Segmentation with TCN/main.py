import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SingleStageTCN, MultiStageTCN
from train import train
from dataset import VideoDataset, collate_fn
from evaluate import evaluate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Parameters
root_dataset = 'data/'
features_dir = root_dataset + 'features'
groundtruth_dir = root_dataset + 'groundTruth'
mapping_file = root_dataset + 'mapping.txt'
train_bundle_file = root_dataset + 'train.bundle'
test_bundle_file = root_dataset + 'test.bundle'

batch_size = 4
# Create dataset and dataloader for training
train_dataset = VideoDataset(features_dir, groundtruth_dir, mapping_file, train_bundle_file)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# Create dataset and dataloader for testing
test_dataset = VideoDataset(features_dir, groundtruth_dir, mapping_file, test_bundle_file)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model parameters
num_layers = 10
num_filters = 64
kernel_size = 3
learning_rate = 0.001
num_classes = max(int(line.split()[0]) for line in open('data/mapping.txt')) + 1 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_stages = 4

# model = SingleStageTCN(num_layers, num_filters, kernel_size, num_classes)
model = MultiStageTCN(num_stages=num_stages, num_layers=num_layers, num_filters=num_filters, kernel_size=kernel_size, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 50
result = train(model=model, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs, train_loader=train_loader, test_loader=test_loader, device=device, batch_size=batch_size, num_classes=num_classes)
# result = train(model=model, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs, train_loader=train_loader, test_loader=test_loader, device=device, batch_size=batch_size, num_classes=num_classes)
print(result)
