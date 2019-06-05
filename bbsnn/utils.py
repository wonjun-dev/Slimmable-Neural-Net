import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

class Mytrainset(Dataset):
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.cifar10 = datasets.CIFAR10(root='../data',
                                        download=True,
                                        train=True,
                                        transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)

class Mytestset(Dataset):
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.cifar10 = datasets.CIFAR10(root='../data',
                                        download=True,
                                        train=False,
                                        transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)

def get_dataloader(trainset, testset, args, kwargs):
    train_sampler, valid_sampler = _split_train_val(trainset, args)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    valid_loader = DataLoader(trainset, batch_size=args.valid_batch_size, sampler=valid_sampler, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, valid_loader, test_loader


def _split_train_val(dataset, args):
    idx = list(range(len(dataset)))
    np.random.shuffle(idx)
    train_idx, valid_idx = idx[args.valid_batch_size:50000], idx[:args.valid_batch_size]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def width_sampler(width_mult_list, num):
    width = np.array([0.25, 1.])
    width = np.append(width, np.random.choice(width_mult_list, num-2))

    return list(width)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
