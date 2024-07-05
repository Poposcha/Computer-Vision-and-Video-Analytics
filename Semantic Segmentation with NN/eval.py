import torch
from dataset import SegmentationDataset, transform
from model import UNet
from torch.utils.data import DataLoader

# Hyperparameters
num_classes = 20
batch_size = 16

# Directories
val_image_dir = 'data/val_images'
val_label_dir = 'data/val_labels'

# Load the model
model = UNet(num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Dataset and Dataloader
val_dataset = SegmentationDataset(val_image_dir, val_label_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Evaluation metrics
def pixel_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = torch.eq(preds, labels).sum().item()
    total = labels.numel()
    return correct / total

def mean_iou(outputs, labels, num_classes):
    _, preds = torch.max(outputs, 1)
    iou = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds[target_inds]).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            iou.append(float('nan'))
        else:
            iou.append(intersection / union)
    return torch.tensor(iou).mean().item()

# Evaluation loop
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
