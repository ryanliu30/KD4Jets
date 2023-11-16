# system imports
import sys

# 3rd party imports
from pytorch_lightning import LightningModule
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from ..knowledge_distillation_base import KnowledgeDistillationBase
from .lorentz_net import LorentzNetWrapper

class DeepSetKD(KnowledgeDistillationBase):
    def __init__(self, hparams):
        super().__init__(hparams)        
    
    def get_student(self, hparams):
        """
        return a torch.nn.Module model that is the student model. The forward() function should return logits and a list of extra guidance vectors.
        """        
        return DeepSetTagger(**hparams)
        
    def get_teacher(self, hparams):
        """
        return a torch.nn.Module model that is the teacher model. The forward() function should return logits and a list of extra guidance vectors.
        """
        model = LorentzNetWrapper()
        for param in model.parameters():
            param.requires_grad = False
        return model

class DeepSetTagger(nn.Module):
    def __init__(self, d_input = 5, d_ff = 72, d_latent = 72, d_output = 2, dropout = 0., depth = 2, **kwargs):
        super().__init__()
        
        phi_config = []
        d = d_input
        for _ in range(depth - 1):
            phi_config.extend([
                nn.Linear(d, d_ff, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(d_ff),
            ])
            d = d_ff
        phi_config.append(nn.Linear(d_ff, d_latent, bias=True))
        
        self.phi = nn.Sequential(*phi_config)
        
        rho_config = [nn.BatchNorm1d(d_latent),]
        d = d_latent
        for _ in range(depth - 1):
            rho_config.extend([
                nn.Linear(d, d_ff, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(d_ff),
            ])
            d = d_ff
        self.rho = nn.Sequential(*rho_config)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d, d_output, bias=True)
        )
        
        self.d_latent = d_latent
        
    def preprocess(self, batch):
        mask = batch["label"].float()
        Pjet = (batch["Pmu"][:, 2:] * mask[:, :, None]).sum(1)
        rel_pT = norm(batch["Pmu"][:, 2:, 1:3])/norm(Pjet[:, None, 1:3])
        deta = torch.atanh(batch["Pmu"][:, 2:, [3]]/norm(batch["Pmu"][:, 2:, 1:4])) - torch.atanh(Pjet[:, [3]]/norm(Pjet[:, 1:4])).view(-1, 1, 1)
        dphi = torch.atan2(batch["Pmu"][:, 2:, [2]], batch["Pmu"][:, 2:, [1]]) - torch.atan2(Pjet[:, [2]], Pjet[:, [1]]).view(-1, 1, 1)
        dphi = torch.remainder(dphi + torch.pi, 2 * torch.pi) - torch.pi
        features = torch.cat([rel_pT, deta, dphi, batch["nodes"][:, 2:, :1]], dim = -1).float()
        features.masked_fill_((~batch["label"][:, :, None]) | (features != features), 0)
        rel_pT.masked_fill_((~batch["label"][:, :, None]) | (rel_pT != rel_pT), 0)
        return features, mask, rel_pT.float()
        
    def forward(self, batch):
        features, mask, weights = self.preprocess(batch)
        x = torch.zeros((*features.shape[:2], self.d_latent), dtype = features.dtype, device = features.device)
        x[mask.squeeze() == 1] = self.phi(features[mask.squeeze() == 1])
        
        z = (x * weights).sum(1)
        z = self.rho(z)
        
        output = self.output_layer(z)
        
        return output, {"rep": z}
    
def norm(x):
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)