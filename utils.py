from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

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
    # def __init__(self):
    #     super(CIFAR10Data, self).__init__()
    #     # create a dictionar

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # img.save(f"Image_{index}.jpg")

        if self.transform is not None:
            pos = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            return pos, target

        return img, target

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

train_normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_orig_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# Proposed model uses differentiable normalization - no need to normalize here
test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
