import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np 

from data_aug.augmentation_utils import GaussianBlur

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR10Data(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos, target

def load_cifar_data(batch_size, num_workers, seed, input_shape, use_augmentation=False, load_pair=False):
    # For workers in dataloaders
    def _init_fn(worker_id):
        np.random.seed(int(seed))

    if use_augmentation:
        train_transform, test_transform = get_augmented_transforms(input_shape)
    else:
        train_transform, test_transform = get_tensor_transforms(input_shape)

    if load_pair:
        dataloader_class = CIFAR10Pair
    else:
        dataloader_class = CIFAR10Data
        
    # data prepare
    train_data = dataloader_class(root='data', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn,
                            drop_last=True)
    memory_data = dataloader_class(root='data', train=True, transform=test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)
    test_data = dataloader_class(root='data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)
    return train_loader, memory_loader, test_loader

def get_augmented_transforms(input_shape):
    train_orig_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # Note: No Gaussian blurring is used for CIFAR10 experiments in SimCLR paper 
        # (ref: Appendix Section B.9 in https://arxiv.org/pdf/2002.05709.pdf)
        # GaussianBlur(kernel_size=int(0.1 * input_shape[1])),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
    test_orig_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    return test_orig_transform, test_orig_transform

def get_tensor_transforms():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Proposed model uses differentiable normalization - no need to normalize here
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, test_transform