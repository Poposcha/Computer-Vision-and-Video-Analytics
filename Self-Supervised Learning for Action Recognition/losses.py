# Contrastive Learning Setup

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        
        z = torch.cat([z_i, z_j], dim=0)
        
        sim_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
        sim_matrix_exp = torch.exp(sim_matrix)

        sim_sum = sim_matrix_exp.masked_fill(mask, 0).sum(dim=1)
        
        pos_sim = torch.exp(self.cosine_similarity(z_i, z_j) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        
        loss = -torch.log(pos_sim / sim_sum)
        
        return loss.mean()

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        
        # Normalize the feature vectors
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Compute similarity between positive pairs
        positive_sim = torch.exp(torch.sum(z_i * z_j, dim=1) / self.temperature)
        
        # Compute similarity between all pairs
        similarity_matrix = torch.exp(torch.mm(z_i, z_j.T) / self.temperature)
        
        # Create labels for the positive pairs
        labels = torch.arange(batch_size)
        
        # Compute the denominator (sum of all similarities)
        denominator = similarity_matrix.sum(dim=1)
        
        # Compute the InfoNCE loss
        loss = -torch.log(positive_sim / denominator).mean()
        
        return loss