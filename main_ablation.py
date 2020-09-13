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

import utils
from model import SimCLRJacobianModel
import wandb

import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import copy

def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)

# get color jitter tensors
def get_batch_color_jitter_tensors(net, shape, eps=1e-3):
    jit_params = net.augment.jit.generate_parameters(shape)
    jit_params_delta = net.augment.jit.generate_parameters(shape)
    for k in jit_params.keys():
        if k is 'order':
            jit_params_delta[k] = jit_params[k]
        else:
            jit_params_delta[k] = jit_params[k] + eps

    if cuda_available:
        for k in jit_params.keys():
            jit_params[k] = jit_params[k].cuda()
            jit_params_delta[k] = jit_params_delta[k].cuda()

    return  jit_params, jit_params_delta

def get_batch_op_augment_params(op, shape, eps, only_keys=None, clamp_low=0, clamp_hi=31):
    params = op.generate_parameters(shape)
    params_delta = op.generate_parameters(shape)
    if type(params) == dict:
        for k in params.keys():
            if only_keys is not None and k in only_keys:
                params_delta[k] = params[k] + eps
                #  Needs to be generalized, currently using this for only crop
                params_delta[k] = params_delta[k].clamp(clamp_low, clamp_hi)
            else:
                if params[k].dtype == torch.float64 or params[k].dtype == torch.float32:
                    # if k is 'order':
                    #     params_delta[k] = params[k]
                    # else:
                    params_delta[k] = params[k] + eps
                else:
                    params_delta[k] = params[k]

        if cuda_available:
            for k in params.keys():
                params[k] = params[k].cuda()
                params_delta[k] = params_delta[k].cuda()

    return  params, params_delta

def get_batch_op_augment_params_centered(op, shape, eps, only_keys=None, clamp_low=0, clamp_hi=31):
    params = op.generate_parameters(shape)
    params_delta_r = op.generate_parameters(shape)
    params_delta_l = op.generate_parameters(shape)
    if type(params) == dict:
        for k in params.keys():
            if only_keys is not None and k in only_keys:
                params_delta_r[k] = params[k] + eps
                params_delta_l[k] = params[k] - eps
                params_delta_r[k] = params_delta_r[k].clamp(clamp_low, clamp_hi)
                params_delta_l[k] = params_delta_l[k].clamp(clamp_low, clamp_hi)
            else:
                if params[k].dtype == torch.float64 or params[k].dtype == torch.float32:
                    # if k is 'order':
                    #     params_delta[k] = params[k]
                    # else:
                    params_delta_r[k] = params[k] + eps
                    params_delta_l[k] = params[k] - eps
                else:
                    params_delta_r[k] = params[k]
                    params_delta_l[k] = params[k]

        if cuda_available:
            for k in params.keys():
                params[k] = params[k].cuda()
                params_delta_r[k] = params_delta_r[k].cuda()
                params_delta_l[k] = params_delta_l[k].cuda()

    return  params, params_delta_r, params_delta_l

def get_batch_augmentation_centered_params(net, shape, eps=1e-3):
    params = {}
    params_delta_r, params_delta_l = {}, {}
    # Generate params for cropping
    params['crop_params'], params_delta_r['crop_params_delta'], params_delta_l['crop_params_delta'] = get_batch_op_augment_params_centered(
                                                net.augment.crop, shape, eps=1, only_keys=['src'])
    # Generate params for horizontal flip
    params['hor_flip_params'], params_delta_r['hor_flip_params_delta'], params_delta_l['hor_flip_params_delta'] = get_batch_op_augment_params_centered(
                                                net.augment.hor_flip, shape, eps)
    # Generate params for color jitter
    # Probability of color jitter
    params['jit_prob'] = torch.rand(shape[0])
    params['jit_threshold'] = 0.8

    B = (params['jit_prob'] < params['jit_threshold']).sum()
    jit_params_shape = (B, 3, 32, 32) # assuming that images are of shape 3, 32, 32

    # parameters for color jitter
    params['jit_params'], params_delta_r['jit_params_delta'], params_delta_l['jit_params_delta'] = get_batch_op_augment_params_centered(
                                                net.augment.jit, jit_params_shape, eps)
                                    
    # Generate params for random grayscaling
    params['grayscale_params'], params_delta_r['grayscale_params_delta'], params_delta_l['grayscale_params_delta'] = get_batch_op_augment_params_centered(
                                                net.augment.rand_grayscale, shape, eps)    
    return params, params_delta_r, params_delta_l

# get color jitter tensors
def get_batch_augmentation_params(net, shape, eps=1e-3):
    params = {}
    params_delta = {}
    # Generate params for cropping
    params['crop_params'], params_delta['crop_params_delta'] = get_batch_op_augment_params(net.augment.crop, shape, eps=1, only_keys=['src'])
    # Generate params for horizontal flip
    params['hor_flip_params'], params_delta['hor_flip_params_delta'] = get_batch_op_augment_params(net.augment.hor_flip, shape, eps)
    # Generate params for color jitter
    # Probability of color jitter
    # Probability of color jitter
    params['jit_prob'] = torch.rand(shape[0])
    params['jit_threshold'] = 0.8

    B = (params['jit_prob'] < params['jit_threshold']).sum()
    jit_params_shape = (B, 3, 32, 32) # assuming that images are of shape 3, 32, 32
    
    # parameters for color jitter
    params['jit_params'], params_delta['jit_params_delta'] = get_batch_op_augment_params(net.augment.jit, shape, eps)
    # Generate params for random grayscaling
    params['grayscale_params'], params_delta['grayscale_params_delta'] = get_batch_op_augment_params(net.augment.rand_grayscale, shape, eps)    
    return params, params_delta

def get_jitter_norm_loss(net, pos, out, params, params_delta, eps):
    # compute approximate derivative wrt jitter
    jitter_params = copy.deepcopy(params)
    jitter_params['jit_params'] = params_delta['jit_params_delta']
    _, out_djitter = net(pos, jitter_params)   
    j_djitter = torch.mean(torch.norm((out_djitter - out)/eps, dim=1))
    return j_djitter

def get_jitter_norm_loss_centered(net, pos, params, params_delta_r, params_delta_l, eps):
    # compute approximate derivative wrt jitter
    jitter_params = copy.deepcopy(params)
    jitter_params['jit_params'] = params_delta_r['jit_params_delta']
    _, out_djitter_r = net(pos, jitter_params)
    jitter_params['jit_params'] = params_delta_l['jit_params_delta']
    _, out_djitter_l = net(pos, jitter_params)
    j_djitter = torch.mean(torch.norm((out_djitter_r - out_djitter_l)/(2*eps), dim=1))
    return j_djitter

def get_crop_norm_loss(net, pos, out, params, params_delta, eps):
    # compute approximate derivative wrt crop params
    crop_params = copy.deepcopy(params)
    crop_params['crop_params'] = params_delta['crop_params_delta']
    _, out_dcrop = net(pos, crop_params)   
    j_dcrop = torch.mean(torch.norm((out_dcrop - out)/eps, dim=1))
    return j_dcrop

def get_crop_norm_loss_centered(net, pos, params, params_delta_r, params_delta_l, eps):
    # compute approximate derivative wrt crop params
    crop_params = copy.deepcopy(params)
    crop_params['crop_params'] = params_delta_r['crop_params_delta']
    _, out_dcrop_r = net(pos, crop_params)
    crop_params['crop_params'] = params_delta_l['crop_params_delta']
    _, out_dcrop_l = net(pos, crop_params)
    j_dcrop = torch.mean(torch.norm((out_dcrop_r - out_dcrop_l)/(2*eps), dim=1))
    return j_dcrop

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):

    net.train()

    avg_jtxy = 0.0
    avg_jitter = 0.0
    avg_crop = 0.0

    avg_contr_loss, avg_grad_loss, total_itr = 0, 0, 0

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos, target in train_bar:
        if cuda_available:
            pos = pos.cuda(non_blocking=True)

        if args.grad_compute_type == 'centered':
            params1, params_delta_r1, params_delta_l1 = get_batch_augmentation_centered_params(net, shape=pos.shape, eps=args.eps)
        else:
            params1, params_delta1 = get_batch_augmentation_params(net, shape=pos.shape, eps=args.eps)
        
        # [B, D]
        feature_1, out_1 = net(pos, params1)

        if args.grad_compute_type == 'centered':
            params2, params_delta_r2,  params_delta_l2 = get_batch_augmentation_centered_params(net, shape=pos.shape, eps=args.eps)
        else:
            params2, params_delta2 = get_batch_augmentation_params(net, shape=pos.shape, eps=args.eps)

        # [B, D]
        feature_2, out_2 = net(pos, params2)

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
        
        if args.grad_compute_type == 'default':
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

        elif args.grad_compute_type == 'centered':
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

        else:
            raise ValueError("Unknown grad compute type {}".format(args.grad_compute_type))

        train_optimizer.zero_grad()
        loss.backward()

        train_optimizer.step()
        total_num += batch_size

        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    wandb.log({"jitter_norm" : avg_jitter / total_num, "contrastive loss" : avg_contr_loss / total_num, "crop_norm" : avg_crop / total_num})
    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch, plot_img=True):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
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
        for data, target in test_bar:
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
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=150, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model_type', default='simclr_ablation', type=str, help='Type of model to train - original SimCLR (original) or Proposed (proposed)')
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
    parser.add_argument('--grad_compute_type', default="default", type=str, help='Should we add norm of gradients wrt jitter to loss?')
    parser.add_argument('--seed', default=0, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, k = args.feature_dim, args.k
    batch_size, epochs = args.batch_size, args.epochs
    temperature = args.temperature

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.use_wandb:
        wandb.init(project="contrlearning-gridsearch", config=args)

    cuda_available = torch.cuda.is_available()
    print("Preparing data...")

    # data prepare
    train_data = utils.CIFAR10Data(root='data', train=True,
                                    transform=utils.train_normalize_transform,
                                    download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                            drop_last=True)

    memory_data = utils.CIFAR10Data(root='data', train=True, 
                                    transform=utils.test_transform, 
                                    download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_data = utils.CIFAR10Data(root='data', train=False, 
                                    transform=utils.test_transform, 
                                    download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print("Data prepared. Now initializing out Model...")
    # model setup and optimizer config
    model = SimCLRJacobianModel(feature_dim=feature_dim, norm_type=args.norm_type, output_norm=args.output_norm, model=args.resnet)
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
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), '{}/{}_model_best.pth'.format(output_dir, save_name_pre))
        
        # if epoch % args.save_interval == 0:
        #     torch.save(model.state_dict(), '{}/{}_model_{}.pth'.format(output_dir, save_name_pre, str(epoch)))