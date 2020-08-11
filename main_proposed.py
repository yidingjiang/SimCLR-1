import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import ProposedModel
import wandb

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos, target in train_bar:
        if cuda_available:
            pos = pos.cuda(non_blocking=True)
        # [B, D]
        feature, out = net(pos)

        # [B, B]
        sim_matrix = torch.mm(out, out.t().contiguous())
        mask = (torch.ones_like(sim_matrix) - torch.eye(batch_size, device=sim_matrix.device)).bool()

        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, -1)

        loss = sim_matrix.sum(dim=-1).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            if cuda_available:
                data = data.cuda(non_blocking=True)
            feature, out = net(data)
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
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model_type', default='proposed', type=str, help='Type of model to train - original SimCLR (original) or Proposed (proposed)')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers to load data')
    parser.add_argument('--use_wandb', default=True, type=bool, help='Log results to wandb')
    parser.add_argument('--norm_type', default='batch', type=str, help="Type of norm to use in between FC layers of the projection head")
    parser.add_argument('--output_norm', default=None, type=str, help="Norm to use at the output")

    # args parse
    args = parser.parse_args()
    feature_dim, k = args.feature_dim, args.k
    batch_size, epochs = args.batch_size, args.epochs
    
    wandb.init(project="contrastive learning", config=args)

    cuda_available = torch.cuda.is_available()
    print("Preparing data...")
    if args.model_type == 'proposed':
        # data prepare
        train_data = utils.CIFAR10Data(root='data', train=True, transform=utils.train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                                drop_last=True)
        memory_data = utils.CIFAR10Data(root='data', train=True, transform=utils.test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_data = utils.CIFAR10Data(root='data', train=False, transform=utils.test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    elif args.model_type == 'original':
        # data prepare
        train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                                drop_last=True)
        memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Data prepared. Now initializing out Model...")
    # model setup and optimizer config
    model = ProposedModel(feature_dim=feature_dim, norm_type=args.norm_type, output_norm=args.output_norm)
    inputs = torch.randn(1, 3, 32, 32)
    if cuda_available:
        model = model.cuda()
        inputs = inputs.cuda()
    flops, params = profile(model, inputs=(inputs,))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
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
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train loss": total_loss, "test_acc@1": test_acc_1, "test_acc@5": test_acc_5})

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
