import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from datetime import datetime

from utils import *
import dataloader
from model import OriginalModel, ProposedModel
import wandb

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, use_original=False):
        super(Net, self).__init__()

        # encoder
        if use_original:
            self.f = OriginalModel().f
        else:
            self.f = SimCLRJacobianModel().f

        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default=None, required=True,
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model_type', type=str, default='proposed', help='Type of model to train - original SimCLR (original) or Proposed (proposed)')
    parser.add_argument('--exp_name', required=True, type=str, help="name of experiment")
    parser.add_argument('--exp_group', default='linear-classification', type=str, help='exp_group that can be used to filter results.')
    parser.add_argument('--seed', default=0, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--use_wandb', default=False, type=bool, help='Log results to wandb')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    
    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.use_wandb:
        wandb.init(project="contrastive learning", config=args)

    train_transform = dataloader.train_transform if args.model_type == 'proposed' else dataloader.train_orig_transform
    train_data = CIFAR10(root='data', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    test_transform = dataloader.test_transform if args.model_type == 'proposed' else dataloader.test_orig_transform
    test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    use_original = False
    if args.model_type == 'original':
        use_original = True
    model = Net(num_class=len(train_data.classes), pretrained_path=model_path, use_original=use_original).cuda()

    for param in model.f.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    if not os.path.exists('results-linear'):
        os.mkdir('results-linear')

    output_dir = 'results-linear/{}'.format(datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_name_pre = '{}_{}_{}_{}_{}_{}'.format(args.exp_name, args.model_type, feature_dim, k, batch_size, epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if args.use_wandb:
            wandb.log({'train_loss': train_loss, 'train_acc@1': train_acc_1, 'train_acc@5': train_acc_5, 
                        'test_loss': test_loss, 'test_acc@1': test_acc_1, 'test_acc@5': test_acc_5})

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/linear_statistics.csv'.format(output_dir), index_label='epoch')
        
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), '{}/{}_linear_model.pth'.format(output_dir, save_name_pre))
