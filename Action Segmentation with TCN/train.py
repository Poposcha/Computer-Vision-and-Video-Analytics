import torch
import torch.nn as nn
from tqdm import tqdm
from evaluate import evaluate
from metrics import *

def check_class_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print("Class distribution:", distribution)

def train(model, criterion, optimizer, num_epochs, train_loader, test_loader, device, batch_size, num_classes, video_loss_required=False,):
    val_meter = ValMeter()
    model.train()

    train_stats_dict = {
        "MoF": [], 
        "Edit": [], 
        "F1@10": [], 
        "F1@25": [], 
        "F1@50": [], 
    }
    
    val_stats_dict = {
        "MoF": [], 
        "Edit": [], 
        "F1@10": [], 
        "F1@25": [], 
        "F1@50": [], 
    }
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for features, labels in train_loader:
                # print(check_class_distribution(labels))
                outputs = model(features)
                sftmax = nn.Softmax(dim=1)
                probs = sftmax(outputs)
                predictions = torch.argmax(probs, dim=1)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if video_loss_required:
                    _, max_indices = torch.max(probs, dim=1)

                    one_hot_output = torch.zeros_like(probs)
                    one_hot_output.scatter_(1, max_indices.unsqueeze(1), 1)  # one-hot encoded output
                    
                    one_hot_labels = torch.zeros(labels.shape[0], num_classes, labels.shape[1])
                    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

                    bce_loss = nn.BCEWithLogitsLoss()
                    loss += bce_loss(one_hot_output, one_hot_labels)

                epoch_loss += loss.item()
                num_batches += 1
                avg_loss = epoch_loss / batch_size / num_batches

                val_meter.update_stats(target=labels, prediction=predictions, num_videos=batch_size)    

                pbar.set_postfix(val_meter.log_stats(), loss=loss.item() / batch_size, avg_loss=avg_loss)
                pbar.update(1)
            
            train_stats = val_meter.log_stats()
            for key in train_stats:
                train_stats_dict[key].append(train_stats[key])

            val_stats = evaluate(model=model, test_loader=test_loader, device=device, batch_size=batch_size)
            for key in val_stats:
                val_stats_dict[key].append(val_stats[key])

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)/batch_size:.4f}')

    print("Training is finished.")

    return train_stats_dict, val_stats_dict
