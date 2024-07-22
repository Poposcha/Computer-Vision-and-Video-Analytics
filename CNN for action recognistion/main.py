import torch
from torchvision.transforms import Compose
import os.path as osp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ResNet3D
from transforms import MyPermute, RandomTemporalCrop, TemporalViewCrop, MyRandomCrop_4dim, MyToTensor
from dataset import MiniUCFRGBDataset

def train_one_epoch(dataloader, model_info):
    """
    Trains the model for one epoch using the provided dataloader.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    epoch_loss = 0.0
    epoch_accuracy = 0.0

    for frames, labels in tqdm(dataloader):
        # Prepare data and perform forward pass
        inputs = frames.to(DEVICE)
        labels = labels.to(DEVICE)
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Aggregate statistics
        epoch_loss += loss.item()
        correct_predictions = (predictions.argmax(1) == labels).type(torch.float).sum().item()
        epoch_accuracy += correct_predictions

    # Calculate average loss and accuracy
    epoch_loss /= num_batches
    epoch_accuracy /= size
    return epoch_loss, epoch_accuracy

def evaluate(dataloader, model_info):
    """
    Evaluates the model on the provided dataloader.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0
    test_accuracy = 0.0

    with torch.no_grad():
        for frames, labels in tqdm(dataloader):
            # Prepare data
            inputs = frames.to(DEVICE)
            labels = labels.to(DEVICE)
            predictions = model(inputs)

            # Compute loss
            loss = loss_fn(predictions, labels)
            test_loss += loss.item()

            # Calculate accuracy
            correct_predictions = (predictions.argmax(1) == labels).type(torch.float).sum().item()
            test_accuracy += correct_predictions

    # Calculate average loss and accuracy
    test_loss /= num_batches
    test_accuracy /= size
    return test_loss, test_accuracy

def main():
    """
    Main function to execute training and evaluation.
    """
    # Data loading and transformations
    train_transform = Compose([MyToTensor(), MyRandomCrop_4dim((112, 112)), RandomTemporalCrop(), MyPermute((3, 0, 1, 2))])
    val_transform = Compose([MyToTensor(), MyRandomCrop_4dim((112, 112)), TemporalViewCrop(), MyPermute((0, 4, 1, 2, 3))])

    # Initialize datasets
    train_dataset = MiniUCFRGBDataset(video_path, osp.join(data_path, 'train.txt'), osp.join(data_path, 'classes.txt'), train_transform)
    val_dataset = MiniUCFRGBDataset(video_path, osp.join(data_path, 'validation.txt'), osp.join(data_path, 'classes.txt'), val_transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Model initialization
    model = ResNet3D().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(train_loader, {'model': model, 'optimizer': optimizer, 'loss_fn': loss_fn})
        val_loss, val_accuracy = evaluate(val_loader, {'model': model, 'loss_fn': loss_fn})

        # Display training progress
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

    # Save trained model
    torch.save(model.state_dict(), 'resnet3d.pth')

if __name__ == "__main__":
    main()
