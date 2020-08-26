import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet34, resnet18

import torchvision
import torchvision.transforms.functional as FT
import numpy as np

class OriginalModel(nn.Module):
    def __init__(self, feature_dim=128, model='resnet18'):
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
                                    nn.Linear(512, feature_dim, bias=True))

        else:
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
        # These are standard values for CIFAR10. We will have to change this for imagenet
        self.mu = torch.Tensor([0.4914, 0.4822, 0.4465])
        self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010])

    # Note that I should only normalize in test mode; no other type of augmentation should be performed
    def forward(self, x, rot_mat, brightness, mode='train', visualize=False):
        import pdb; pdb.set_trace()
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

            # # Color jitter
            # x = x + brightness
            # x = torch.clamp(x, 0.0, 1.0)

            # if visualize:
            #     pil_img_bright = FT.to_pil_image(x[0])
            #     pil_img_bright.show()

        #  Normalize - implementing this because inbuilt normalize doesn't seem to support batch normalization
        mean = self.mu.repeat(B, 1, 1, 1).view(B, 3, 1, 1)
        std = self.sigma.repeat(B, 1, 1 ,1).view(B, 3, 1, 1)
        if torch.cuda.is_available():
            mean = mean.cuda()
            std = std.cuda()
        x_norm = (x - mean)/std
        
        # Used to check if normalization above gives same value as inbuilt normalization
        # x_test = torch.zeros_like(x)
        # for b in range(B):
        #     x_test[b] = FT.normalize(x[b], self.mu, self.sigma)

        if visualize:
            pil_img_normalized = FT.to_pil_image(x[0])
            pil_img_normalized.show()
        #     pil_img_normalized_test = FT.to_pil_image(x_test[0])
        #     pil_img_normalized_test.show()

        return x

import kornia.augmentation as K
class KorniaAugmentationModule(nn.Module):
    def __init__(self, batch_size=512):
        super().__init__()
        # These are standard values for CIFAR10. We will have to change this for imagenet
        self.mu = torch.Tensor([0.4914, 0.4822, 0.4465])
        self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010])
        self.augment = nn.Sequential(
            K.RandomResizedCrop(size=(32, 32)),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1),
            K.RandomGrayscale(p=0.2)
        )

        self.aff = K.RandomAffine(360)
        self.jit = K.ColorJitter(0.2, 0.3, 0.2, 0.3)
        
        self.jit_prob = 0.8
        self.normalize = K.Normalize(self.mu, self.sigma)

    # Note that I should only normalize in test mode; no other type of augmentation should be performed
    def forward(self, x, rot_mat, brightness, mode='train', visualize=False):
        # import pdb; pdb.set_trace()
        B = x.shape[0]

        ###### Uncomment and use following code to visualize images
        if visualize:
            pil_img = FT.to_pil_image(x[0])
            pil_img.show()

        if mode == 'train':
            # Rotation and translation
            if torch.rand(1) < 0.8:
                x = self.augment(x)

            # aff_params = self.aff.generate_parameters(x.shape)
            # aff_params['translations'] = torch.randint(low=-6, high=6, size=aff_params['translations'].shape)
            # aff_params['angle'] = torch.zeros_like(aff_params['angle'])
            # x = self.aff(x, aff_params)
            

            # jit_params = self.jit.generate_parameters(x.shape)
            # x = self.jit(x, jit_params)

            if visualize:
                pil_img_bright = FT.to_pil_image(x[0])
                pil_img_bright.show()

        #  Normalize - implementing this because inbuilt normalize doesn't seem to support batch normalization
        # mean = self.mu.repeat(B, 1, 1, 1).view(B, 3, 1, 1)
        # std = self.sigma.repeat(B, 1, 1 ,1).view(B, 3, 1, 1)
        # if torch.cuda.is_available():
        #     mean = mean.cuda()
        #     std = std.cuda()
        # import pdb; pdb.set_trace()
        # x_norm = (x - mean)/std
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
    def __init__(self, feature_dim=128, norm_type='layer', output_norm='layer', model='resnet18'):
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

        self.augment = KorniaAugmentationModule() #AugmentationModule()

    def forward(self, x, rot_mat=None, brightness=None, mode='train'):
        if mode == 'train':
            assert rot_mat is not None and brightness is not None
        x = self.augment(x, rot_mat, brightness, mode=mode)
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        if self.output_norm is None:
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        # else we are getting normalized output from projection head
        return F.normalize(feature, dim=-1), out


