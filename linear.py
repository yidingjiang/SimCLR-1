import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageNet
from tqdm import tqdm
from datetime import datetime

from dataloader.cifar_dataloader import load_cifar_data
from dataloader.imagenet_dataloader import load_imagenet_data

from models.model import OriginalModel, SimCLRJacobianModel
import wandb

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, model_type, feature_dim, encoder, dataset, input_shape):
        super(Net, self).__init__()

        # encoder
        if model_type == 'original':
            self.f = OriginalModel().f
        else:
            self.f = SimCLRJacobianModel(feature_dim=feature_dim, model=encoder, dataset=dataset, input_shape=input_shape).f

        encoder_out_dim = 512
        if encoder == 'resnet50':
            encoder_out_dim = 2048
            
        # classifier
        self.fc = nn.Linear(encoder_out_dim, num_class, bias=True)
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
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--model_path', type=str, default=None, required=True,
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=4096, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model_type', type=str, default='proposed', help='Type of model to train - original SimCLR (original) or Proposed (proposed)')
    parser.add_argument('--exp_name', required=True, type=str, help="name of experiment")
    parser.add_argument('--exp_group', default='linear-classification', type=str, help='exp_group that can be used to filter results.')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers to load data')
    parser.add_argument('--resnet', default='resnet18', type=str, help='Type of resnet: 1. resnet18, resnet34, resnet50')
    
    parser.add_argument('--use_seed', default=False, type=bool, help='Should we make the process deterministic and use seeds?')
    parser.add_argument('--seed', default=0, type=int, help='Number of sweeps over the dataset to train')
    
    parser.add_argument('--use_wandb', default=False, type=bool, help='Log results to wandb')

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to train the model on. Current choices: 1. cifar10 2. imagenet')
    parser.add_argument('--data_path', default='data', type=str, help='Path to dataset')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes in dataset')
    
    parser.add_argument('--optimizer', default='nestorov', type=str, help='Optimizer to use for optimizing the training objective')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    
    if args.use_seed:
        # set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.use_wandb:
        wandb.init(project="contrastive learning", config=args)

    input_shape = None
    if args.dataset == 'cifar10':
        input_shape = (3, 32, 32)
    elif args.dataset == 'imagenet':
        input_shape = (3, 224, 224)

    train_transform, test_transform = None, None
    if args.model_type == 'original':
        if args.dataset == 'cifar10':
            train_loader, memory_loader, test_loader = load_cifar_data(args.datapath, batch_size, args.num_workers, args.use_seed, args.seed, 
                                                        input_shape=input_shape,
                                                        use_augmentation=True, 
                                                        load_pair=False,
                                                        linear_eval=True)
        elif args.dataset == 'imagenet':
            train_loader, memory_loader, test_loader = load_imagenet_data(args.data_path, batch_size, args.num_workers, args.use_seed, args.seed, 
                                                        input_shape=input_shape, 
                                                        use_augmentation=True, 
                                                        load_pair=False,
                                                        linear_eval=True)
    elif args.model_type == 'proposed':
        if args.dataset == 'cifar10':
            train_loader, memory_loader, test_loader = load_cifar_data(args.datapath, batch_size, args.num_workers, args.use_seed, args.seed, 
                                                    input_shape=input_shape, 
                                                    use_augmentation=False, 
                                                    load_pair=False,
                                                    linear_eval=True)
        elif args.dataset == 'imagenet':
            train_loader, memory_loader, test_loader = load_imagenet_data(args.data_path, batch_size, args.num_workers, args.use_seed, args.seed, 
                                                    input_shape=input_shape, 
                                                    use_augmentation=False, 
                                                    load_pair=False,
                                                    linear_eval=True)

    ### Model to use for training
    ### Load weights of pretrained model
    model = Net(num_class=args.num_classes, pretrained_path=model_path, model_type=args.model_type, feature_dim=args.feature_dim, encoder=args.resnet, 
            dataset=args.dataset, input_shape=input_shape).cuda()

    ### Freeze the weights of the encoder
    for param in model.f.parameters():
        param.requires_grad = False

    ### Optimizer to use for traning the objective
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    elif args.optimizer == 'nestorov':
        # Hyperparams as specific in SimCLR v1 Appendix section B.5
        optimizer = optim.SGD(model.fc.parameters(), lr=0.8, momentum=0.9, nesterov=True)

    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    if not os.path.exists('results-linear'):
        os.mkdir('results-linear')

    output_dir = 'results-linear/{}'.format(datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_name_pre = '{}_{}_{}_{}_{}_{}'.format(args.exp_name, args.exp_group, args.model_type, args.feature_dim, batch_size, epochs)

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
