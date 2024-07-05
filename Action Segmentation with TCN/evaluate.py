import numpy as np
import torch
import torch.nn as nn
from metrics import ValMeter
from tqdm import tqdm


def evaluate(model, test_loader, device, batch_size):
    model.eval()
    val_meter = ValMeter()

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f'Evaluation', unit='batch') as pvalbar:
            for features, labels in test_loader:

                outputs = model(features)
                sftmax = nn.Softmax(dim=1)
                probs = sftmax(outputs)
                predictions = torch.argmax(outputs, dim=1)
                val_meter.update_stats(target=labels, prediction=predictions, num_videos=batch_size)

                
                pvalbar.set_postfix(val_meter.log_stats())
                pvalbar.update(1)
    return val_meter.log_stats()
