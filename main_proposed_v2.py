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

def get_batch_affine_transform_tensors(net, shape, eps=1e-3):
    # Generate some random affine params
    # No rotations; only translations
    aff_params = net.augment.aff.generate_parameters(shape)
    aff_params['angle'] = torch.zeros_like(aff_params['angle'])
    aff_params['translations'] = torch.randint(low=-6, high=6, size=aff_params['translations'].shape)

    trans_delta =  aff_params['translations'] + eps
    aff_params_delta = net.augment.aff.generate_parameters(shape)
    aff_params_delta['translations'] = trans_delta
    aff_params_delta['angle'] = torch.zeros_like(aff_params_delta['angle'])

    if cuda_available:
        for k in aff_params.keys():
            aff_params[k] = aff_params[k].cuda()
            aff_params_delta[k] = aff_params_delta[k].cuda()
    return aff_params, aff_params_delta

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
    pos_mask = (target == target[0])
    pos_mask[0] = False # Don't use the first tensor to compute similarity
    pos_vals = sim_vals.masked_select(pos_mask)
    
    neg_mask = (target != target[0])
    if torch.cuda.is_available():
        neg_mask = neg_mask.cuda()
    neg_vals = sim_vals.masked_select(neg_mask)

    loss = torch.mean(neg_vals) - torch.mean(pos_vals)
    return loss

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):

    net.train()
    avg_jtxy = 0.0
    avg_jitter = 0.0
    avg_sim_loss = 0.0

    avg_contr_loss, avg_grad_loss, total_itr = 0, 0, 0

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos, target in train_bar:
        if cuda_available:
            pos = pos.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        affine_params, affine_params_delta = get_batch_affine_transform_tensors(net, shape=pos.shape, eps=args.eps)
        jit_params, jit_params_delta = get_batch_color_jitter_tensors(net, shape=pos.shape, eps=args.eps)

        # [B, D]
        feature, out = net(pos, affine_params, jit_params)
        contr_loss = corrected_loss(out, target)
        avg_sim_loss += contr_loss

        # compute approximate derivative wrt theta, tx and ty and jitter
        _, out_dtxy = net(pos, affine_params_delta, jit_params)
        _, out_djitter = net(pos, affine_params, jit_params_delta)    

        j_dtxy = torch.norm((out_dtxy - out)/ args.eps)
        j_djitter = torch.norm((out_djitter - out)/args.eps)
        
        avg_jtxy += j_dtxy/args.batch_size
        avg_jitter += j_djitter/args.batch_size

        grad_loss = (j_dtxy + j_djitter)/args.batch_size

        loss = contr_loss + grad_loss
        train_optimizer.zero_grad()
        loss.backward()

        train_optimizer.step()
        total_itr += 1
        total_num += batch_size

        avg_contr_loss += contr_loss.item()
        avg_grad_loss += grad_loss.item() * args.batch_size

        total_loss += avg_contr_loss + avg_grad_loss/total_num

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss))
    
    wandb.log({"txy_norm" : avg_jtxy, "jitter_norm" : avg_jitter, "contr loss": avg_sim_loss})
    total_loss = avg_contr_loss/total_itr + avg_grad_loss/total_num
    return total_loss


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
