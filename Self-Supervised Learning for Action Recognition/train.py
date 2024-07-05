from ResNet_sheet4.model import resnet18
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import torch


def train_model(model, train_dataloader, validation_dataloader, train_dataset, validation_dataset, criterions, optimizer, num_epochs):
    train_loss = {"NTX_loss": [], "Info_loss" : [], "f": [], "all": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batch_size = validation_batch_size= 32
    acc = [0]
    for epoch_ind in range(num_epochs):
        # Training
        print(f"Training epoch {epoch_ind + 1} ...")
        model.train()
        avg_loss = {"NTX_loss": 0, "Info_loss" : 0, "f": 0, "all": 0}
        for batch in tqdm(train_dataloader, maxinterval=int(len(train_dataset) / train_batch_size) + 1):
            input_reshaped_1 = batch["input_1"].to(device)
            input_reshaped_2 = batch["input_2"].to(device)
            target = batch["label"].to(device)

            logits_1 = model(input_reshaped_1)
            logits_2 = model(input_reshaped_2)

            # predictions = torch.argmax(logits_1, dim=1)
            loss3 = F.cross_entropy(logits_1, target)

            loss1 = criterions[0](logits_1, logits_2)
            loss2 = criterions[1](logits_1, logits_2)
            # print(f"NTX_loss: {loss1.item()}, Info_loss: {loss2.item()}")
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss["NTX_loss"] += loss1.item()
            avg_loss["Info_loss"] += loss2.item()
            avg_loss["f"] += loss3.item()
            avg_loss["all"] += loss.item()

        avg_loss["NTX_loss"] /= train_batch_size / len(train_dataloader)
        avg_loss["Info_loss"] /= train_batch_size / len(train_dataloader)
        avg_loss["f"] /= train_batch_size / len(train_dataloader)
        avg_loss["all"] /= train_batch_size / len(train_dataloader)

        train_loss["NTX_loss"].append(avg_loss["NTX_loss"])
        train_loss["Info_loss"].append(avg_loss["Info_loss"])
        train_loss['f'].append(avg_loss["f"])
        train_loss["all"].append(avg_loss["all"])

        #Validation

        print("Validating ...")
        model.eval()
        accuracy = 0.
        total_samples = 0
        for batch in tqdm(validation_dataloader, maxinterval=int(len(validation_dataset) / validation_batch_size) + 1):
            with torch.no_grad():
                input_orig = batch["original"].to(device)
                validation_target = batch["label"].to(device)
                logits = model(input_orig)
                validation_predictions = torch.argmax(logits, dim=1)

            # print(f"validation_predictions.shape: {validation_predictions.shape}, validation_target.shape: {validation_target.shape}, len(validation_target): {len(validation_target)}")
            correct_predictions = (validation_predictions == validation_target).sum().item()
            total_samples += validation_target.size(0)
            accuracy += correct_predictions

        accuracy /= total_samples
        print(f"Validation accuracy: {accuracy}")
        acc.append(accuracy)
        print('Saving model')
        torch.save(model.state_dict(), f"tuned_{accuracy}.pth")

    return (train_loss, acc)
