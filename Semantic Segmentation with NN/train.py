import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SegmentationDataset, transform
from model import UNet

# Hyperparameters
num_classes = 20
num_epochs = 50
learning_rate = 1e-3
batch_size = 16

# Directories
train_image_dir = 'data/train_images'
train_label_dir = 'data/train_labels'
val_image_dir = 'data/val_images'
val_label_dir = 'data/val_labels'

# Datasets and Dataloaders
train_dataset = SegmentationDataset(train_image_dir, train_label_dir, transform, num_classes)
val_dataset = SegmentationDataset(val_image_dir, val_label_dir, transform, num_classes)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = UNet(num_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f'Validation Loss: {val_loss/len(val_loader)}')

# Save the model
torch.save(model.state_dict(), 'model.pth')