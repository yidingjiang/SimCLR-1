import argparse
import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd.gradcheck import zero_gradients

import dataloader
from utils import *
from model import SimCLRJacobianModel
import wandb

import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import copy

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):

    net.train()

    avg_jtxy = 0.0
    avg_jitter = 0.0
    avg_crop = 0.0

    avg_contr_loss, avg_grad_loss, total_itr = 0, 0, 0

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        #if cuda_available:
        #    pos = pos.cuda(non_blocking=True)
        if cuda_available:
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

        params1, params2 = None, None
        if args.model_type == 'proposed' and args.grad_compute_type == 'centered':
            params1, params_delta_r1, params_delta_l1 = get_batch_augmentation_centered_params(net, shape=pos_1.shape, eps=args.eps)
        elif args.model_type == 'proposed' and args.grad_compute_type == 'default':
            params1, params_delta1 = get_batch_augmentation_params(net, shape=pos_1.shape, eps=args.eps)
        
        # [B, D]
        feature_1, out_1 = net(pos_1, params=params1, model_type=args.model_type)

        if args.model_type == 'proposed' and args.grad_compute_type == 'centered':
            params2, params_delta_r2,  params_delta_l2 = get_batch_augmentation_centered_params(net, shape=pos_2.shape, eps=args.eps)
        elif args.model_type == 'proposed' and args.grad_compute_type == 'default':
            params2, params_delta2 = get_batch_augmentation_params(net, shape=pos_2.shape, eps=args.eps)

        # [B, D]
        feature_2, out_2 = net(pos_2, params=params2, model_type=args.model_type)

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

        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        avg_contr_loss += loss.item() * args.batch_size
        
        if args.model_type == 'proposed':
            if args.grad_compute_type == 'default' and args.plot_jac:
                j_djitter1 = get_jitter_norm_loss(net, pos, out_1, params1, params_delta1, args.eps)
                j_djitter2 = get_jitter_norm_loss(net, pos, out_2, params2, params_delta2, args.eps)
                avg_jitter += (j_djitter1 + j_djitter2).item() * args.batch_size
                if args.use_jitter_norm:
                    loss += args.lamda1 * (j_djitter1 + j_djitter2)

                j_dcrop1 = get_crop_norm_loss(net, pos, out_1, params1, params_delta1, args.eps)
                j_dcrop2 = get_crop_norm_loss(net, pos, out_2, params2, params_delta2, args.eps)
                avg_crop += (j_dcrop1 + j_dcrop2).item() * args.batch_size
                if args.use_crop_norm:
                    loss += args.lamda2 * (j_dcrop1 + j_dcrop2)

            elif args.grad_compute_type == 'centered' and args.plot_jac:
                j_djitter1 = get_jitter_norm_loss_centered(net, pos, params1, params_delta_r1, params_delta_l1, args.eps)
                j_djitter2 = get_jitter_norm_loss_centered(net, pos, params2, params_delta_r2, params_delta_l2, args.eps)

                avg_jitter += (j_djitter1 + j_djitter2).item() * args.batch_size
                if args.use_jitter_norm:
                    loss += args.lamda1 * (j_djitter1 + j_djitter2)
                
                j_dcrop1 = get_crop_norm_loss_centered(net, pos, params1, params_delta_r1, params_delta_l1, args.eps)
                j_dcrop2 = get_crop_norm_loss_centered(net, pos, params2, params_delta_r2, params_delta_l2, args.eps)
                avg_crop += (j_dcrop1 + j_dcrop2).item() * args.batch_size
                if args.use_crop_norm:
                    loss += args.lamda2 * (j_dcrop1 + j_dcrop2)

        train_optimizer.zero_grad()
        loss.backward()

        train_optimizer.step()
        total_num += batch_size

        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    if args.use_wandb:
        wandb.log({"jitter_norm" : avg_jitter / total_num, "contrastive loss" : avg_contr_loss / total_num, "crop_norm" : avg_crop / total_num})
    
    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch, plot_img=True):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            if cuda_available:
                data = data.cuda(non_blocking=True)
            feature, out = net(data, mode='test')
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            if cuda_available:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            feature, out = net(data, mode='test')

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
            if plot_img:
                fig, axs = plt.subplots(10, 7)
                target = target.cpu().detach().numpy()
                sim_indices = sim_indices.cpu().detach().numpy()
                for label in range(10):
                    cur_index = np.where(target == label)[0][0]
                    cur_img = data[cur_index].cpu().detach().numpy()
                    cur_features = feature[cur_index].cpu().detach().numpy()

                    ax[label][0].imshow(cur_img)
                    ax[label][1].imshow(cur_features)

                    for offset, index in enumerate(sim_indices[cur_index][:5]):
                        ax[label][2 + offset].imshow(feature_bank[:, index].cpu().detach().numpy())
                
                fig.savefig('plot.png')
                wandb.log({"Features of k-NN; k=5": [wandb.Image(Image.open('plot.png'), caption=f"feature @epoch {epoch}")]})

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=150, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model_type', default='proposed', type=str, help='Type of model to train - original SimCLR (original) or Proposed (proposed)')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers to load data')
    parser.add_argument('--use_wandb', default=False, type=bool, help='Log results to wandb')
    parser.add_argument('--norm_type', default='batch', type=str, help="Type of norm to use in between FC layers of the projection head")
    parser.add_argument('--output_norm', default=None, type=str, help="Norm to use at the output")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='learning rate')
    parser.add_argument('--resnet', default='resnet18', type=str, help='Type of resnet: 1. resnet18, resnet34, resnet50')
    parser.add_argument('--eps', default=1e-4, type=float, help='epsilon to compute jacobian')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    
    parser.add_argument('--lamda1', default=5e-3, type=float, help='weight for jacobian of color jitter')
    parser.add_argument('--lamda2', default=5e-3, type=float, help='weight for jacobian of crop')

    parser.add_argument('--exp_name', required=True, type=str, help="name of experiment")
    parser.add_argument('--exp_group', default='grid_search', type=str, help='exp_group that can be used to filter results.')
    
    parser.add_argument('--save_interval', default=25, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--use_jitter_norm', default=False, type=bool, help='Should we add norm of gradients wrt jitter to loss?')
    parser.add_argument('--use_crop_norm', default=False, type=bool, help='Should we add norm of gradients wrt jitter to loss?')
    parser.add_argument('--grad_compute_type', default='default', type=str, help='Should we add norm of gradients wrt jitter to loss? (default/centered)')
    parser.add_argument('--seed', default=0, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--plot_jac', default=False, type=bool, help='Should the jacobian be plotted?')
    parser.add_argument('--use_augment', default=True, type=str2bool, help='Should we use augmentation')
    
    # args parse
    args = parser.parse_args()
    feature_dim, k = args.feature_dim, args.k
    batch_size, epochs = args.batch_size, args.epochs
    temperature = args.temperature

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.use_wandb:
        wandb.init(project="contrastive learning", config=args)

    cuda_available = torch.cuda.is_available()
    print("Preparing data...")

    train_transform = dataloader.train_transform if args.model_type == 'proposed' else dataloader.train_orig_transform
    test_transform = dataloader.test_transform if args.model_type == 'proposed' else dataloader.test_orig_transform

    # data prepare
    train_data = dataloader.CIFAR10Pair(root='data', train=True,
                                    transform=train_transform,
                                    download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                            drop_last=True)

    memory_data = dataloader.CIFAR10Pair(root='data', train=True, 
                                    transform=test_transform, 
                                    download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_data = dataloader.CIFAR10Pair(root='data', train=False, 
                                    transform=test_transform, 
                                    download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print("Data prepared. Now initializing out Model...")
    # model setup and optimizer config
    model = SimCLRJacobianModel(feature_dim=feature_dim, model=args.resnet, use_augment=args.use_augment)
    inputs = torch.randn(1, 3, 32, 32)

    if cuda_available:
        model = model.cuda()
        inputs = inputs.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    c = len(memory_data.classes)

    # Some initial setup for saving checkpoints and results
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}'.format(args.exp_name, args.model_type, feature_dim, k, batch_size, epochs)

    if not os.path.exists('results'):
        os.mkdir('results')
    
    output_dir = 'results/{}'.format(datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # training loop
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        plot_img = False
        # if (epoch-1) % 20 == 0 or epoch == epochs - 1:
        #     plot_img = True
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch, plot_img=plot_img)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train loss": train_loss, "test_acc@1": test_acc_1, "test_acc@5": test_acc_5})

        # save statistics
        # data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), '{}/{}_model_best.pth'.format(output_dir, save_name_pre))
        
        # if epoch % args.save_interval == 0:
        #     torch.save(model.state_dict(), '{}/{}_model_{}.pth'.format(output_dir, save_name_pre, str(epoch)))