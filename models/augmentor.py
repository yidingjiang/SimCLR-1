import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.style_transfer_model import ConvLayer
from models.style_transfer_model import ResidualBlock
from models.style_transfer_model import UpsampleConvLayer
from models.conv_layers import SNConv2d
from models.transformer import ExemplarTransformer


def get_augmentor(args):
    augmentor = None
    if args.aug_type == "convnet":
        augmentor = LpAugmentor(clip=args.aug_clip_output, radius=args.aug_radius)
    elif args.aug_type == "convnet_specnorm":
        augmentor = LpAugmentorSpecNorm(clip=args.aug_clip_output, radius=args.aug_radius)
    elif args.aug_type == "style_transfer":
        augmentor = LpAugmentorStyleTransfer(clip=args.aug_clip_output, radius=args.aug_radius)
    elif args.aug_type == "transformer":
        augmentor = LpAugmentorTransformer(clip=args.aug_clip_output, radius=args.aug_radius)
    else:
        raise ValueError("Unrecognized augmentor type: {}".format(args.aug_type))
    return augmentor


class LpAugmentor(nn.Module):
    def __init__(self, p=1, noise_dim=3, clip=True, radius=0.05):
        super(LpAugmentor, self).__init__()
        self.noise_dim = noise_dim
        self.p = p
        self.clip = clip
        self.radius = radius

        self.l_1 = nn.Conv2d(self.noise_dim + 3, 64, 3, padding=1)
        self.l_2 = nn.Conv2d(self.noise_dim + 64, 64, 3, padding=1)
        self.l_3 = nn.Conv2d(self.noise_dim + 64, 64, 3, padding=1)
        self.l_4 = nn.Conv2d(self.noise_dim + 64, 3, 3, padding=1)

    def noise_shapes(self, input_dim):
        return [[3, input_dim, input_dim]] * 4

    def sample_noise(self, x, batch_size, input_dim):
        noise = [
            torch.from_numpy(np.float32(np.random.normal(size=[batch_size] + s))).cuda()
            for s in self.noise_shapes(input_dim)
        ]
        return noise

    def forward(self, x, noise=None):
        shape = x.size()
        total_size = shape[1] * shape[2] * shape[3]
        if not noise:
            noise = self.sample_noise(x, shape[0], shape[2])
        h1 = F.relu(self.l_1(torch.cat((x, noise[0]), 1)))
        h2 = F.relu(self.l_2(torch.cat((h1, noise[1]), 1)))
        h3 = F.relu(self.l_3(torch.cat((h2, noise[2]), 1)))
        h4 = self.l_4(torch.cat((h3, noise[3]), 1))
        if self.clip:
            norm = h4.norm(p=self.p, dim=(1, 2, 3), keepdim=True)
            h4 = total_size * h4.div(norm)
        out = x + self.radius * h4
        return torch.clamp(out, 0.0, 1.0)


class LpAugmentorStyleTransfer(nn.Module):
    def __init__(self, p=1, noise_dim=3, clip=True, radius=0.05):
        super(LpAugmentorStyleTransfer, self).__init__()
        self.p = p
        self.clip = clip
        self.radius = radius
        # Initial convolution layers
        self.conv1 = ConvLayer(4, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128 + 1)
        self.res2 = ResidualBlock(128 + 2)
        self.res3 = ResidualBlock(128 + 3)
        # self.res4 = ResidualBlock(128)
        # self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(131, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def noise_shapes(self, input_dim):
        shape = [[1, input_dim, input_dim]]
        shape += [[1, input_dim // 4, input_dim // 4]] * 3
        return shape

    def sample_noise(self, x, batch_size, input_dim):
        noise = [
            torch.from_numpy(np.float32(np.random.uniform(size=[batch_size] + s))).cuda()
            for s in self.noise_shapes(input_dim)
        ]
        return noise

    def forward(self, x, noise=None):
        x_o = x
        shape = x.size()
        total_size = shape[1] * shape[2] * shape[3]
        if not noise:
            noise = self.sample_noise(x, shape[0], shape[2])
        x = torch.cat((x, noise[0]), 1)
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(torch.cat((y, noise[1]), 1))
        y = self.res2(torch.cat((y, noise[2]), 1))
        y = self.res3(torch.cat((y, noise[3]), 1))
        # y = self.res4(y)
        # y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        if self.clip:
            norm = y.norm(p=1, dim=(1, 2, 3), keepdim=True).detach()
            y = total_size * y.div(norm)
        out = x_o + self.radius * y
        return torch.clamp(out, 0.0, 1.0)


class LpAugmentorSpecNorm(nn.Module):
    def __init__(self, p=1, noise_dim=3, clip=False, radius=0.05):
        super(LpAugmentorSpecNorm, self).__init__()
        self.noise_dim = noise_dim
        self.p = p
        self.clip = clip
        self.radius = radius

        self.l_1 = SNConv2d(self.noise_dim + 3, 64, 3, padding=1)
        self.l_2 = SNConv2d(self.noise_dim + 64, 64, 3, padding=1)
        self.l_3 = SNConv2d(self.noise_dim + 64, 64, 3, padding=1)
        self.l_4 = SNConv2d(self.noise_dim + 64, 3, 3, padding=1)

    def noise_shapes(self, input_dim):
        return [[3, input_dim, input_dim]] * 4

    def sample_noise(self, x, batch_size, input_dim):
        noise = [
            torch.from_numpy(np.float32(np.random.normal(size=[batch_size] + s)))
            for s in self.noise_shapes(input_dim)
        ]
        for n in noise:
            n.to(x.device)
        return noise

    def forward(self, x, noise=None):
        shape = x.size()
        total_size = shape[1] * shape[2] * shape[3]
        if not noise:
            noise = self.sample_noise(x, shape[0], shape[2])
        h1 = F.relu(self.l_1(torch.cat((x, noise[0]), 1)))
        h2 = F.relu(self.l_2(torch.cat((h1, noise[1]), 1)))
        h3 = F.relu(self.l_3(torch.cat((h2, noise[2]), 1)))
        h4 = self.l_4(torch.cat((h3, noise[3]), 1))
        if self.clip:
            norm = h4.norm(p=1, dim=(1, 2, 3), keepdim=True).detach()
            h4 = h4.div(norm)
        out = x + self.radius * total_size * h4
        return torch.clamp(out, 0.0, 1.0)


class LpAugmentorTransformer(nn.Module):
    def __init__(self, p=1, noise_dim=3, num_noise_token=2, clip=False, radius=0.05):
        super(LpAugmentorTransformer, self).__init__()
        self.noise_dim = noise_dim
        self.num_noise_token = num_noise_token
        self.p = p
        self.clip = clip
        self.radius = radius

        self.noise_to_embedding = nn.Linear(
            128 * num_noise_token, 128 * num_noise_token
        )
        self.conv_down = nn.Sequential(
            ConvLayer(3, 64, kernel_size=5, stride=2),
            torch.nn.InstanceNorm2d(64, affine=True),
            torch.nn.ReLU(),
            ConvLayer(64, 16, kernel_size=3, stride=1),
            torch.nn.InstanceNorm2d(16, affine=True),
            torch.nn.ReLU(),
        )
        self.transformer = ExemplarTransformer(
            image_size=48,
            patch_size=6,
            dim=128,
            depth=8,
            heads=8,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.05,
            channels=16,
        )
        self.conv_up = nn.Sequential(
            UpsampleConvLayer(16, 64, kernel_size=3, stride=1, upsample=2),
            # torch.nn.InstanceNorm2d(128, affine=True),
            torch.nn.ReLU(),
            UpsampleConvLayer(64, 64, kernel_size=3, stride=1, upsample=1),
            # torch.nn.InstanceNorm2d(128, affine=True),
            torch.nn.ReLU(),
            ConvLayer(64, 3, kernel_size=1, stride=1),
        )

    def noise_shapes(self, input_dim):
        return [[3, input_dim, input_dim]] * 4

    def sample_noise(self, x, batch_size, input_dim):
        noise = [
            torch.from_numpy(np.float32(np.random.uniform(size=[batch_size] + s))).cuda()
            for s in self.noise_shapes(input_dim)
        ]
        for n in noise:
            n.to(x.device)
        return noise

    def forward(self, x, noise=None):
        shape = x.size()
        total_size = shape[1] * shape[2] * shape[3]
        if not noise:
            noise = self.sample_noise(x, shape[0], shape[2])
        x_o = x
        x = self.conv_down(x)
        shape = x.size()
        total_size = shape[1] * shape[2] * shape[3]
        noise = noise[0]
        shape = noise.size()
        noise = torch.reshape(noise, [shape[0], -1])[:, : 128 * self.num_noise_token]
        noise = self.noise_to_embedding(noise)
        noise = torch.reshape(noise, [shape[0], self.num_noise_token, 128])
        # y = self.conv_proj(self.transformer(x, noise))
        y = self.transformer(x, noise)
        y = self.conv_up(y)
        if self.clip:
            norm = y.norm(p=1, dim=(1, 2, 3), keepdim=True).detach()
            y = y.div(norm)
        out = x_o + self.radius * total_size * y
        return torch.clamp(out, 0.0, 1.0)
