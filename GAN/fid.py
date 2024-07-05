import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from torchvision.models import inception_v3

def prepare_inception_model(device):
    # imput images 299x299
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # Remove the classification head
    model.to(device)
    return model

def get_activations(data_loader, model, device):
    model.eval()
    activations = []
    with torch.no_grad():
        for batch, _ in data_loader:
            batch = batch.to(device)
            features = model(batch)
            activations.append(features.cpu().numpy())
    return np.concatenate(activations, axis=0)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])).dot(sigma2 + eps * np.eye(sigma2.shape[0])))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(real_loader, generated_loader, device):
    model = prepare_inception_model(device)

    # Extract features from real images
    real_activations = get_activations(real_loader, model, device)
    fake_activations = get_activations(generated_loader, model, device)

    # Calculate FID
    mu1, sigma1 = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid