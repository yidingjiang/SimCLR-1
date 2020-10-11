import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet34, resnet18

import torchvision
import torchvision.transforms.functional as FT
import numpy as np
import kornia.augmentation as K
from kornia.constants import Resample
from data_aug.augmentation_utils import RandomResizedCrop

# Placeholder class
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class OriginalModel(nn.Module):
    def __init__(self, feature_dim=256, model='resnet18', dataset='imagenet'):
        super(OriginalModel, self).__init__()
        
        resnet = None 
        if model == 'resnet50': 
            resnet = resnet50
        elif model == 'resnet34':
            resnet = resnet34
        elif model == 'resnet18':
            resnet = resnet18
        else:
            raise ValueError(f"Specified resnet model {model} not supported.")

        self.f = []
        if dataset == 'cifar10':
            for name, module in resnet().named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            # encoder
            self.f = nn.Sequential(*self.f)
        else:
            self.f = resnet(pretrained=False)
            self.f.fc = Identity()

        if model == 'resnet18' or model == 'resnet34':
            # projection head
            self.g = nn.Sequential( nn.Linear(512, 512, bias=False), 
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(512, feature_dim, bias=False))

        else:
            # projection head
            self.g = nn.Sequential( nn.Linear(2048, 2048, bias=False), 
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(2048, feature_dim, bias=False))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

# differentiable image augmentation module - affine + jitter
torch_pi = torch.acos(torch.zeros(1)).item() * 2
class AugmentationModule(nn.Module):
    def __init__(self, batch_size=512, dataset='imagenet'):
        super().__init__()

    
        if dataset == 'cifar10':
            # These are standard values for CIFAR10.
            self.mu = torch.Tensor([0.4914, 0.4822, 0.4465])
            self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010])
        elif dataset == 'imagenet':
            # These are standard values for ImageNet 
            # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
            self.mu = torch.Tensor([0.485, 0.456, 0.406])
            self.sigma = torch.Tensor([0.229, 0.224, 0.225])
        else:
            raise ValueError("Unknown dataset {}".format(dataset))
        
    # Note that I should only normalize in test mode; no other type of augmentation should be performed
    def forward(self, x, rot_mat, brightness, mode='train', visualize=False):
        B = x.shape[0]

        ###### Uncomment and use following code to visualize images
        if visualize:
            pil_img = FT.to_pil_image(x[0])
            pil_img.show()

        if mode == 'train':
            # Rotation and translation
            grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False)
            if visualize:
                pil_img_rotated = FT.to_pil_image(x[0])
                pil_img_rotated.show()

            # Color jitter
            x = x + brightness
            x = torch.clamp(x, 0.0, 1.0)

            if visualize:
                pil_img_bright = FT.to_pil_image(x[0])
                pil_img_bright.show()

        #  Normalize - implementing this because inbuilt normalize doesn't seem to support batch normalization
        mean = self.mu.repeat(B, 1, 1, 1).view(B, 3, 1, 1)
        std = self.sigma.repeat(B, 1, 1 ,1).view(B, 3, 1, 1)
        if torch.cuda.is_available():
            mean = mean.cuda()
            std = std.cuda()
        x = (x - mean)/std
        
        # Used to check if normalization above gives same value as inbuilt normalization
        # x_test = torch.zeros_like(x)
        # for b in range(B):
        #     x_test[b] = FT.normalize(x[b], self.mu, self.sigma)

        if visualize:
            pil_img_normalized = FT.to_pil_image(x[0])
            pil_img_normalized.show()
            # pil_img_normalized_test = FT.to_pil_image(x_test[0])
            # pil_img_normalized_test.show()

        return x

class KorniaAugmentationModule(nn.Module):
    def __init__(self, batch_size=512, hor_flip_prob=0.5, jit_prob=0.8, gs_prob=0.2, strength=1, dataset='imagenet'):
        super().__init__()

        if dataset == 'cifar10':
            # These are standard values for CIFAR10.
            self.mu = torch.Tensor([0.4914, 0.4822, 0.4465])
            self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010])
        elif dataset == 'imagenet':
            # These are standard values for ImageNet 
            # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
            self.mu = torch.Tensor([0.485, 0.456, 0.406])
            self.sigma = torch.Tensor([0.229, 0.224, 0.225])
        else:
            raise ValueError("Unknown dataset {}".format(dataset))

        # self.augment = nn.Sequential(
        #     K.RandomResizedCrop(size=(32, 32)),
        #     K.RandomHorizontalFlip(p=0.5),
        #     K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        #     K.RandomGrayscale(p=0.2)
        # )

        self.hor_flip_prob = hor_flip_prob
        self.jit_prob = jit_prob
        self.gs_prob = gs_prob
        self.s = strength 
        
        # self.crop = K.RandomResizedCrop(size=(32, 32), interpolation=Resample.BILINEAR, same_on_batch=False)
        self.crop = RandomResizedCrop(size=32)
        self.hor_flip = K.RandomHorizontalFlip(p=self.hor_flip_prob, same_on_batch=False)
        self.jit = K.ColorJitter(brightness=0.8 * self.s, contrast=0.8 * self.s, saturation=0.8 * self.s, hue= 0.2 * self.s, p=self.jit_prob, same_on_batch=False)
        self.rand_grayscale =  K.RandomGrayscale(p=self.gs_prob, same_on_batch=False)
        
        self.aff = K.RandomAffine(360)
        self.normalize = K.Normalize(self.mu, self.sigma)

    @torch.no_grad()
    # Note that I should only normalize in test mode; no other type of augmentation should be performed
    def forward(self, x, params=None, mode='train', visualize=False, augment_type='orig'):
        B = x.shape[0]

        if visualize:
            pil_img = FT.to_pil_image(x[0])
            pil_img.show()

        if mode == 'train':
            if augment_type == 'orig':
                x = self.crop(x, params['crop_params'])
                x = self.hor_flip(x, params['hor_flip_params'])
                x[params['jit_batch_probs']] = self.jit(x[params['jit_batch_probs']], params['jit_params'])
                x = self.rand_grayscale(x, params['grayscale_params'])

            elif augment_type == 'rot-jit':
                x = self.aff(x, params['aff_params'])
                x = self.jit(x, params['jit_params'])

            elif augment_type == 'no_params':
                x = self.crop(x)
                x = self.hor_flip(x)
                x = self.jit(x)
                x = self.rand_grayscale(x)

            if visualize:
                pil_img_bright = FT.to_pil_image(x[0])
                pil_img_bright.show()

        x = self.normalize(x)
        
        # Used to check if normalization above gives same value as inbuilt normalization
        # x_test = torch.zeros_like(x)
        # for b in range(B):
        #     x_test[b] = FT.normalize(x[b], self.mu, self.sigma)

        if visualize:
            pil_img_normalized = FT.to_pil_image(x[0])
            pil_img_normalized.show()
        return x

class ProposedModel(nn.Module):
    def __init__(self, feature_dim=128, norm_type='layer', output_norm='layer', model='resnet18', dataset='imagenet'):
        super(ProposedModel, self).__init__()
        
        resnet = None 
        if model == 'resnet50': 
            resnet = resnet50
        elif model == 'resnet34':
            resnet = resnet34
        elif model == 'resnet18':
            resnet = resnet18
        else:
            raise ValueError(f"Specified resnet model {model} not supported.")

        self.output_norm = output_norm
        self.f = []
        for name, module in resnet().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)

        if model == 'resnet18' or model == 'resnet34':
            proj_layers = [nn.Linear(512, 512, bias=False)]
        elif model == 'resnet50':
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

        self.augment = KorniaAugmentationModule(dataset=dataset) #AugmentationModule()

    def forward(self, x, affine_params=None, jit_params=None, mode='train'):
        if mode == 'train':
            assert affine_params is not None and jit_params is not None
        x = self.augment(x, affine_params, jit_params, mode=mode)
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        if self.output_norm is None:
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        # else we are getting normalized output from projection head
        return F.normalize(feature, dim=-1), out

class SimCLRJacobianModel(nn.Module):
    def __init__(self, feature_dim=256, model='resnet18', dataset='imagenet'):
        super(SimCLRJacobianModel, self).__init__()

        resnet = None 
        if model == 'resnet50': 
            resnet = resnet50
        elif model == 'resnet34':
            resnet = resnet34
        elif model == 'resnet18':
            resnet = resnet18
        else:
            raise ValueError(f"Specified resnet model {model} not supported.")

        self.f = []
        for name, module in resnet().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)

        if model == 'resnet18' or model == 'resnet34':
            # projection head
            self.g = nn.Sequential( nn.Linear(512, 512, bias=False), 
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(512, feature_dim, bias=False))

        else:
            # projection head
            self.g = nn.Sequential( nn.Linear(2048, 2048, bias=False), 
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(2048, feature_dim, bias=False))

        self.augment = KorniaAugmentationModule(dataset=dataset)

    def forward(self, x, params=None, mode='train'):

        if mode == 'train':
            assert params is not None

        x = self.augment(x, params=params, mode=mode)
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

