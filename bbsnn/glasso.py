import torch.optim as optim
import torch.nn.functional as F
import argparse
import os

from utils import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--batch_size', default=180, type=int)
parser.add_argument('--valid_batch_size', type=int, default=5000)
parser.add_argument('--test_batch_size', type=int, default=10000)
parser.add_argument('--run_name', default="glasso", type=str)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
set_random_seed(args.seed)

# Data load
print('==> Preparing data..')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
trainset = Mytrainset()
testset = Mytestset()
train_loader, valid_loader, test_loader = get_dataloader(trainset, testset, args, kwargs)

# Build net
print('==> Building model..')
net = VGG('VGG11')
net.to(device)
cent_fn = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

save_dir = os.path.join('../results/', str(args.run_name))

def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for v in vars(args):
            f.write('{}: {}\n'.format(v, getattr(args, v)))

    net.train()
    for epoch in range(args.num_epochs):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, idx) in enumerate(train_loader):
            #sample_width_mult = width_sampler(width_mult_list, args.num_samples)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            log_soft_targets = F.log_softmax(outputs.detach(), dim=1)
            penalty = get_glasso()
            loss = cent_fn(outputs, targets) + 0.001*penalty[0] + 0.001*penalty[1]
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            loss.backward()

            optimizer.step()
            total += targets.size(0)

            progress_bar(batch_idx, len(train_loader), 'Epoch: %d | Loss: %.3f | %.3f%%'
                % (epoch+1, train_loss/(batch_idx+1), 100.*correct/total))

        _validation()

    torch.save({'state_dict':net.state_dict()},
        os.path.join(save_dir, 'model.tar'))


def _validation():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, idx) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            print('Val Acc: %.3f%%'
                % (100.*correct/total))


def test():
    ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, idx) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            print('Test Acc: %.3f%%'
                % (100.*correct/total))

def get_glasso():
    in_channels_penalty = 0
    out_channels_penalty = 0
    for layer in net.features:
        if isinstance(layer, USConv2d):
            in_channels_penalty += layer.in_channels_lasso.sum()
            out_channels_penalty += layer.out_channels_lasso.sum()

    return [in_channels_penalty, out_channels_penalty]



if __name__ == '__main__':
    if args.test:
        test()
    else:
        train()
