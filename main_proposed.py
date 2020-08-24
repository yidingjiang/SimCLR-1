import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd.gradcheck import zero_gradients

import utils
from model import ProposedModel
import wandb

import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image


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

def compute_jacobian_norm(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]
    
    jacobian_norm = 0.0

    grad_output = torch.zeros(*output.size())
    if cuda_available:
        inputs = inputs.cuda()
        grad_output = grad_output.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian_norm += torch.norm(inputs.grad.data)**2

    return torch.sqrt(jacobian_norm/inputs.shape[0])

# def get_affine_transform_tensors(batch_size):
#     torch_pi = torch.acos(torch.zeros(1)).item() * 2
#     theta = 2 * torch_pi * torch.rand(1, requires_grad=True) - torch_pi

#     if cuda_available:
#         theta = theta.cuda()

#     # [2,3]
#     if cuda_available:
#         rot_mat = torch.stack(( torch.cat((torch.cos(theta).cuda(), torch.sin(theta).cuda(), torch.zeros(1).cuda())), 
#                                 torch.cat((-torch.sin(theta).cuda(), torch.cos(theta).cuda(), torch.zeros(1).cuda()))
#                                 ))
#     else:
#         rot_mat = torch.stack(( torch.cat((torch.cos(theta), torch.sin(theta), torch.zeros(1))), 
#                                 torch.cat((-torch.sin(theta), torch.cos(theta), torch.zeros(1)))
#                                 ))
#     # replicate it [B, 2, 3]
#     # grad = torch.autograd.grad( outputs=rot_mat, inputs=theta, 
#     #                             grad_outputs=torch.ones_like(rot_mat).cuda() if cuda_available else torch.ones_like(rot_mat) , 
#     #                             create_graph=True)
#     rot_mat = rot_mat.repeat(batch_size, 1, 1)

#     if cuda_available:
#         rot_mat = rot_mat.cuda()
#     return theta, rot_mat

def get_batch_rot_mat(theta, tx, ty):
    rot_mat = torch.zeros((batch_size, 2, 3), requires_grad=True)
    if torch.cuda.is_available():
        rot_mat = rot_mat.cuda()

    mask1 = torch.zeros_like(rot_mat, dtype=torch.bool)
    mask1[:, 0, 0] = True
    mask2 = torch.zeros_like(rot_mat, dtype=torch.bool)
    mask2[:, 0, 1] = True
    mask3 = torch.zeros_like(rot_mat, dtype=torch.bool)
    mask3[:, 1, 0] = True
    mask4 = torch.zeros_like(rot_mat, dtype=torch.bool)
    mask4[:, 1, 1] = True

    # add some translation too
    # small horizontal translation
    mask5 = torch.zeros_like(rot_mat, dtype=torch.bool)
    mask5[:, 0, 2] = True

    mask6 = torch.zeros_like(rot_mat, dtype=torch.bool)
    mask6[:, 0, 2] = True

    if cuda_available:
        mask1 = mask1.cuda()
        mask2 = mask2.cuda()
        mask3 = mask3.cuda()
        mask4 = mask4.cuda()
        mask5 = mask5.cuda()
        mask6 = mask6.cuda()

    rot_mat = rot_mat.masked_scatter(mask1, theta.cos())
    rot_mat = rot_mat.masked_scatter(mask2, -theta.sin())
    rot_mat = rot_mat.masked_scatter(mask3, theta.sin())
    rot_mat = rot_mat.masked_scatter(mask4, theta.cos())
    rot_mat = rot_mat.masked_scatter(mask5, tx)
    rot_mat = rot_mat.masked_scatter(mask6, ty)
    return rot_mat

# Checked; works correctly
def get_batch_affine_transform_tensors(batch_size):
    # Rotate image by a random angle
    torch_pi = torch.acos(torch.zeros(1, requires_grad=True)).item() * 2
    theta = 2 * torch_pi * torch.rand((batch_size, 1), requires_grad=True) - torch_pi #torch.rand((batch_size, 1), requires_grad=True)
    
    tx = torch.rand((batch_size, 1), requires_grad=True)
    ty = torch.rand((batch_size, 1), requires_grad=True)

    theta_delta = theta + torch.rand_like(theta) * 1e-3

    tx_delta = tx + torch.rand_like(tx) * 1e-3
    ty_delta = ty + torch.rand_like(ty) * 1e-3

    if cuda_available:
        theta = theta.cuda()
        tx = tx.cuda()
        ty = ty.cuda()
        theta_delta = theta_delta.cuda()
        tx_delta = tx_delta.cuda()
        ty_delta = ty_delta.cuda()

    rot_mat = get_batch_rot_mat(theta, tx, ty)
    rot_mat_dtheta = get_batch_rot_mat(theta_delta, tx, ty)
    rot_mat_dtx = get_batch_rot_mat(theta, tx_delta, ty)
    rot_mat_dty = get_batch_rot_mat(theta, tx, ty_delta)

    return theta, tx, ty, rot_mat, {'r_dtheta': rot_mat_dtheta, 'r_dtx': rot_mat_dtx, 'r_dty': rot_mat_dty, 
                                    'dtheta': theta_delta, 'dtx': tx_delta, 'dty': ty_delta}

# get color jitter tensors
def get_batch_color_jitter_tensors(batch_size):
    # Generate a value between (0, 0.5)
    constant = torch.tensor([0.5], requires_grad=True)
    brightness = constant * torch.rand((batch_size, 1, 1, 1), requires_grad=True)
    brightness_delta = brightness + torch.rand_like(brightness) * 1e-3

    if cuda_available:
        brightness = brightness.cuda()
        brightness_delta = brightness_delta.cuda()

    return brightness, brightness_delta

def simclr_based_loss(out):
    sim_matrix = torch.mm(out, out.t().contiguous())
    mask = (torch.ones_like(sim_matrix) - torch.eye(batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(batch_size, -1)
    loss = sim_matrix.sum(dim=-1).mean()
    return loss

def corrected_loss(out, target):
    # Find dot product of first example with the rest
    sim_vals = torch.mm(out[0].view(1, -1), out.t())

    # I will consider class of the first example as positive
    # pos_mask = (target == target[0])
    # pos_vals = sim_vals.masked_select(pos_mask)
    
    neg_mask = (target != target[0])
    if torch.cuda.is_available():
        neg_mask = neg_mask.cuda()
    neg_vals = sim_vals.masked_select(neg_mask)

    loss = torch.mean(neg_vals)
    return loss

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    # for name, param in  net.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos, target in train_bar:
        if cuda_available:
            pos = pos.cuda(non_blocking=True)

        theta, tx, ty, rot_mat, delta_dict = get_batch_affine_transform_tensors(args.batch_size)
        brightness, brightness_delta = get_batch_color_jitter_tensors(args.batch_size)

        # theta.retain_grad()
        # tx.retain_grad()
        # ty.retain_grad()
        # brightness.retain_grad()

        # [B, D]
        feature, out = net(pos, rot_mat, brightness)
        loss = corrected_loss(out, target)

        # compute exact derivative wrt theta, tx and ty and jitter
        # loss += compute_jacobian_norm(theta, out)
        # loss += compute_jacobian_norm(tx, out)
        # loss += compute_jacobian_norm(ty, out)
        # compute derivative wrt brightness
        # loss += compute_jacobian_norm(brightness, out)
        
        # compute approximate derivative wrt theta, tx and ty and jitter
        _, out_dtheta = net(pos, delta_dict['r_dtheta'], brightness)
        _, out_dtx = net(pos, delta_dict['r_dtx'], brightness)
        _, out_dty = net(pos, delta_dict['r_dty'], brightness)
        _, out_djitter = net(pos, rot_mat, brightness_delta)

        j_dtheta = torch.norm((out_dtheta - out)/ delta_dict['dtheta'])/args.batch_size
        j_dtx = torch.norm((out_dtx - out)/ delta_dict['dtx'])/args.batch_size
        j_dty = torch.norm((out_dty - out)/ delta_dict['dty'])/args.batch_size
        j_djitter = torch.norm((out_djitter)/ brightness_delta)/args.batch_size
        
        loss += j_dtheta + j_dtx + j_dty + j_djitter

        train_optimizer.zero_grad()
        loss.backward()

        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
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
            feature_bank.append(out)
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
            sim_matrix = torch.mm(out, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)

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

    # args parse
    args = parser.parse_args()
    feature_dim, k = args.feature_dim, args.k
    batch_size, epochs = args.batch_size, args.epochs
    
    if args.use_wandb:
        wandb.init(project="contrastive learning", config=args)

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
    model = ProposedModel(feature_dim=feature_dim, norm_type=args.norm_type, output_norm=args.output_norm, model=args.resnet)
    inputs = torch.randn(1, 3, 32, 32)

    if cuda_available:
        model = model.cuda()
        inputs = inputs.cuda()

    # flops, params = profile(model, inputs=(inputs, theta))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    c = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = 'proposed_{}_{}_{}_{}_{}_{}'.format(feature_dim, k, batch_size, epochs, args.norm_type, args.output_norm)

    if not os.path.exists('results'):
        os.mkdir('results')
    
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
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))

torch.autograd.grad(outputs=out[:, i], inputs=theta, grad_outputs=torch.ones(len(rot_mat)), retain_graph=True, create_graph=True)[0]