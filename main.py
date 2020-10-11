import argparse
import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from tqdm import tqdm

from dataloader.cifar_dataloader import load_cifar_data
from dataloader.imagenet_dataloader import load_imagenet_data
from utils.utils import *
from models.model import OriginalModel
import wandb
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import random

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        
        if cuda_available:
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    all_data = []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            if cuda_available:
                data = data.cuda(non_blocking=True)
            feature, out = net(data)
            feature_bank.append(feature)
            all_data.append(data)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()

        # [N, 3, 32, 32]
        all_data = torch.cat(all_data, dim=0).contiguous()

        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            if cuda_available:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=256, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model_type', default='original', type=str, help='Type of model to train - original SimCLR or Proposed')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers to load data')
    parser.add_argument('--use_wandb', default=False, type=bool, help='Log results to wandb')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='learning rate')
    parser.add_argument('--resnet', default='resnet18', type=str, help='Type of resnet: 1. resnet18, resnet34, resnet50')
    parser.add_argument('--use_seed', default=False, type=bool, help='Should we make the process deterministic and use seeds?')
    parser.add_argument('--seed', default=1, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--exp_name', required=True, type=str, help="name of experiment")
    parser.add_argument('--exp_group', default='grid_search', type=str, help='exp_group that can be used to filter results.')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to train the model on. Current choices: 1. cifar10 2. imagenet')
    parser.add_argument('--data_path', default='data', type=str, help='Path to dataset')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    if args.use_seed:
        seed = args.seed

        # Make the process deterministic
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.use_wandb:
        wandb.init(project="contrastive learning", config=args)

    cuda_available = torch.cuda.is_available()

    print("Preparing data...")

    if args.dataset == 'cifar10':
        train_loader, memory_loader, test_loader = load_cifar_data('data', batch_size, args.num_workers, args.use_seed, args.seed, input_shape=(3,32,32), 
                                                    use_augmentation=True, load_pair=True)
    elif args.dataset == 'imagenet':
        train_loader, memory_loader, test_loader = load_imagenet_data('data/imagenet', batch_size, args.num_workers, args.use_seed, args.seed, input_shape=(3,224,224), 
                                                    use_augmentation=True, load_pair=True)
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    print("Data prepared. Now initializing out Model...")

    # model setup and optimizer config
    model = OriginalModel(feature_dim, model=args.resnet, dataset=args.dataset)
    
    if args.dataset == 'cifar10':
        inputs = torch.randn(1, 3, 32, 32)
    else:
        inputs = torch.randn(1, 3, 224, 224)

    if cuda_available:
        model = model.cuda()
        inputs = inputs.cuda()

    flops, params = profile(model, inputs=(inputs,))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}_seed_{}'.format(args.exp_name, args.model_type, feature_dim, k, batch_size, epochs, args.seed)

    dirname = "results-{}".format(args.dataset)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    output_dir = '{}/{}'.format(dirname, datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Starting training")
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)

        if args.dataset == 'cifar10':
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch)

            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
            if args.use_wandb:
                wandb.log({"epoch": epoch, "train loss": train_loss, "test_acc@1": test_acc_1, "test_acc@5": test_acc_5})

            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv('{}/{}_statistics.csv'.format(dirname, save_name_pre), index_label='epoch')
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                torch.save(model.state_dict(), '{}/{}_model_best.pth'.format(output_dir, save_name_pre))

        else:
            # k-NN based testing is memory intensive, not possible for ImageNet
            if args.use_wandb:
                wandb.log({"epoch": epoch, "train loss": train_loss})
        
            if epoch % 10 == 0:
                torch.save(model.state_dict(), '{}/{}_model_{}.pth'.format(output_dir, save_name_pre, epoch))
    
    torch.save(model.state_dict(), '{}/{}_model_best.pth'.format(output_dir, save_name_pre))
