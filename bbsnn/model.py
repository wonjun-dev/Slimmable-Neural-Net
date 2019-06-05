'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = USLinear(512, 10, us=[True, False])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for order, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if order == 0:
                    # head
                    layers += [USConv2d(in_channels, x, kernel_size=3, padding=1, us=[False, True]),
                               USBatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                    in_channels = x
                else:
                    # body
                    layers += [USConv2d(in_channels, x, kernel_size=3, padding=1),
                               USBatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                    in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

#test()
