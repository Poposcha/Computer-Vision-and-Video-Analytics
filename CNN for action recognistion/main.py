import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ResNet3D
from transforms import MyPermute, RandomTemporalCrop, TemporalViewCrop, MyRandomCrop_4dim, MyToTensor
from dataset import MiniUCFRGBDataset


def train_one_epoch(dataloader, models):
    """
    Train for one epoch.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    epoch_losses = np.zeros(len(models), dtype=float)
    epoch_acc = np.zeros(len(models), dtype=float)

    for batch, (frames, labels) in enumerate(tqdm(dataloader)):
        labels = labels.to(DEVICE)
        inputs = {'frames': frames.to(DEVICE)}

        for i, model_info in enumerate(models):
            model, optimizer, loss_fn, input_type = model_info['model'], model_info['optimizer'], model_info['loss_fn'], model_info['input_type']
            predictions = model(inputs[input_type])
            loss = loss_fn(predictions, labels)

            epoch_losses[i] += loss.item()
            epoch_acc[i] += (predictions.argmax(1) == labels).type(torch.float).sum().item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    epoch_losses /= num_batches
    epoch_acc /= size

    return epoch_losses, epoch_acc

def evaluate(dataloader, models):
    """
    Evaluate the models on the test set.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_losses = np.zeros(len(models), dtype=float)
    test_acc = np.zeros(len(models), dtype=float)

    loss_result = 0.
    acc_res = 0.
    with torch.no_grad():
        for frames, labels in tqdm(dataloader):
            labels = labels.to(DEVICE)
            inputs = {'frames': frames.to(DEVICE)}

            in_shape = inputs['frames'].shape
            inputs['frames'] = inputs['frames'].view(in_shape[0]*in_shape[1], in_shape[2], in_shape[3], in_shape[4], in_shape[5])

            for ind, model_info in enumerate(models):
                model, loss_fn, input_type = model_info['model'], model_info['loss_fn'], model_info['input_type']
                predictions = model(inputs[input_type]).reshape(4, -1, 25)

                now_loss = 0.
                for i in range(predictions.shape[0]):
                    loss = loss_fn(predictions[i], labels)
                
                    now_loss += loss.item()
                loss_result += now_loss / predictions.shape[0]

                now_acc = 0.
                for i in range(predictions.shape[0]):
                    _, preds = torch.max(predictions[i], 1)
                    now_acc = max(now_acc, torch.sum(preds == labels.data))
                acc_res += now_acc
                
                test_losses[ind] += loss_result / len(dataloader)
                test_acc[ind] += acc_res / len(dataloader)

    test_losses /= num_batches
    test_acc /= size

    return test_losses, test_acc

def save_model_weights(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Saved {filepath}")


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Define data augmentation
    train_transform = Compose([
        MyToTensor(),
        MyRandomCrop_4dim((112, 112)),
        RandomTemporalCrop(),
        MyPermute((3, 0, 1, 2)),
    ])

    val_transform = Compose([
        MyToTensor(),
        MyRandomCrop_4dim((112, 112)),
        TemporalViewCrop(),
        MyPermute((0, 4, 1, 2, 3))
    ])

    print("Loading datasets...")
    data_path = "data"
    video_path = osp.join(data_path, 'mini_UCF')

    train_dataset = MiniUCFRGBDataset(
        video_path, 
        osp.join(data_path, 'train.txt'),
        osp.join(data_path, 'classes.txt'),
        train_transform,
    )

    val_dataset = MiniUCFRGBDataset(
        video_path, 
        osp.join(data_path, 'validation.txt'),
        osp.join(data_path, 'classes.txt'),
        val_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


    # Optimizer, loss function, and learning rate scheduler
    print("Initializing models...")
    model_ResNet3D = ResNet3D().to(DEVICE)
    optimizer_ResNet3D = torch.optim.SGD(model_ResNet3D.parameters(), lr=0.01, momentum=0.9)
    loss_fn_ResNet3D = nn.CrossEntropyLoss()
    ResNet3D_model_info = {
        'model': model_ResNet3D,
        'optimizer': optimizer_ResNet3D,
        'loss_fn': loss_fn_ResNet3D,
        'input_type': 'frames',
        'model_name': 'ResNet3D'
    }

    # Training loop
    epochs = 10
    models = [ResNet3D_model_info]

    # Track losses and accuracies
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}\n-------------------------------")

            train_epoch_losses, train_epoch_acc = train_one_epoch(train_loader, models)
            train_losses.append(train_epoch_losses)
            train_acc.append(train_epoch_acc)

            val_epoch_losses, val_epoch_acc = evaluate(val_loader, models)
            val_losses.append(val_epoch_losses)
            val_acc.append(val_epoch_acc)

            print(f"EPOCH: {epoch+1}\ntrain accuracy: {train_epoch_acc}\nval_acc: {val_epoch_acc}\n\ntrain losses: {train_epoch_losses}\nval losses: {val_epoch_losses}")

            
            if epoch == 0:
                save_model_weights(models[0]["model"], f'{models[0]["model_name"]}.pth')

                prev_acc_train = train_epoch_acc
                prev_acc_val = val_epoch_acc
            else:
                if prev_acc_val[0] < val_epoch_acc[0]:
                    save_model_weights(models[0]["model"], f'{models[0]["model_name"]}.pth')

                prev_acc_train = train_epoch_acc
                prev_acc_val = val_epoch_acc
    
    # Convert lists to arrays for plotting
    train_losses = np.array(train_losses).T
    val_losses = np.array(val_losses).T
    train_acc = np.array(train_acc).T
    val_acc = np.array(val_acc).T

    # Plot training losses
    plt.title("Training losses")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for i, model_info in enumerate(models):
        plt.plot(train_losses[i], label=model_info['model_name'])
    plt.legend()
    plt.savefig("losses_train.jpg")
    plt.clf()

    # Plot validation losses
    plt.title("Validation losses")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for i, model_info in enumerate(models):
        plt.plot(val_losses[i], label=model_info['model_name'])
    plt.legend()
    plt.savefig("losses_val.jpg")
    plt.clf()

    # Plot training accuracies
    plt.title("Training accuracies")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for i, model_info in enumerate(models):
        plt.plot(train_acc[i], label=model_info['model_name'])
    plt.legend()
    plt.savefig("acc_train.jpg")
    plt.clf()

    # Plot validation accuracies
    plt.title("Validation accuracies")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for i, model_info in enumerate(models):
        plt.plot(val_acc[i], label=model_info['model_name'])
    plt.legend()
    plt.savefig("acc_val.jpg")
    plt.clf()

    # Print final results
    print("Final training losses:", train_losses[:, -1])
    print("Final training accuracies:", train_acc[:, -1])
    print("Final validation losses:", val_losses[:, -1])
    print("Final validation accuracies:", val_acc[:, -1])
    print('Training complete!')

if __name__ == "__main__":
    main()