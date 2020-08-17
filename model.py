import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

import torchvision
import torchvision.transforms.functional as FT
import numpy as np

class OriginalModel(nn.Module):
    def __init__(self, feature_dim=128):
        super(OriginalModel, self).__init__()

        self.f = []
        for name, module in resnet34().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential( nn.Linear(2048, 512, bias=False), 
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True), 
                                nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

# differentiable image augmentation module
torch_pi = torch.acos(torch.zeros(1)).item() * 2

class AugmentationModule(nn.Module):

    def __init__(self, batch_size=512):
        super().__init__()
        # theta = 2 * torch_pi * torch.rand(1) - torch_pi
        # zero = torch.zeros(1)

        # rot_mat = torch.stack([
        #         torch.stack([torch.cos(theta), torch.sin(theta), zero]), 
        #         torch.stack([-torch.sin(theta), torch.cos(theta), zero])
        #     ])

        # self.rot_mat = nn.Parameter(rot_mat.repeat(batch_size, 1, 1))

    # theta should be in degrees between -180 to 180
    def forward(self, x, rot_mat):
        grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

class ProposedModel(nn.Module):
    def __init__(self, feature_dim=128, norm_type='layer', output_norm='layer'):
        super(ProposedModel, self).__init__()

        self.output_norm = output_norm
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)

        proj_layers = [nn.Linear(2048, 512, bias=False)]
        if norm_type is not None:
            if norm_type == 'batch':
                proj_layers.append(nn.BatchNorm1d(512))
            elif norm_type == 'layer':
                proj_layers.append(nn.LayerNorm(512))
            else:
                raise ValueError(f"Unknown norm type : {norm_type}")
        proj_layers.append(nn.ReLU(inplace=True))
        proj_layers.append(nn.Linear(512, feature_dim, bias=True))

        if output_norm is not None:
            if output_norm == 'layer':
                proj_layers.append(nn.LayerNorm(feature_dim))
        # projection head
        self.g = nn.Sequential(*proj_layers)

        self.augment = AugmentationModule()

    def forward(self, x, rot_mat):
        x = self.augment(x, rot_mat)
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        if self.output_norm is None:
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
            
        # else we are getting normalized output from projection head
        return F.normalize(feature, dim=-1), out


