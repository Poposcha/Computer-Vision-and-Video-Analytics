import torch
from dataset import SegmentationDataset, transform
from model import UNet
from torch.utils.data import DataLoader

# Hyperparameters and setup
num_classes = 20
batch_size = 16

# Load trained model for evaluation
model = UNet(num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Setup dataset and dataloader for validation
val_dataset = SegmentationDataset('data/val_images', 'data/val_labels', transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define evaluation metrics
def pixel_accuracy(outputs, labels):
    """
    Calculates pixel-level accuracy from the model predictions.
    :param outputs: Model output logits.
    :param labels: Ground truth labels.
    """
    _, preds = torch.max(outputs, 1)
    correct = torch.eq(preds, labels).sum().item()
    total = labels.numel()
    return correct / total

def mean_iou(outputs, labels, num_classes):
    """
    Calculates mean Intersection over Union (IoU) for segmentation.
    :param outputs: Model output logits.
    :param labels: Ground truth labels.
    :param num_classes: Number of classes in segmentation.
    """
    _, preds = torch.max(outputs, 1)
    iou = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds[target_inds]).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            iou.append(float('nan'))  # Avoid division by zero
        else:
            iou.append(intersection / union)
    return torch.tensor(iou).mean().item()

# Evaluation loop for accuracy and IoU
val_acc = 0.0
val_iou = 0.0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        val_acc += pixel_accuracy(outputs, labels)
        val_iou += mean_iou(outputs, labels, num_classes)
    
    val_acc /= len(val_loader)
    val_iou /= len(val_loader)

print(f'Validation Accuracy: {val_acc}')
print(f'Mean IoU: {val_iou}')
