import torch.nn as nn
import torch

width_mult_list = [0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.]

def make_divisible(v, divisor=8, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=False,
                 us=[True, True], ratio=[1, 1]):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = 1
        self.us = us
        self.ratio = ratio
        self.in_channels_lasso = 0
        self.out_channels_lasso = 0

    def _get_glasso(self):
        self.in_channels_lasso = torch.sqrt(torch.sum(self.weight[:self.out_channels, :self.in_channels, :, :]**2, (1,2,3)))
        self.out_channels_lasso = torch.sqrt(torch.sum(self.weight[:self.out_channels, :self.in_channels, :, :]**2, (0,2,3)))

    def _get_masked_weight(self, weight):
        z = weight>0.001
        z = torch.sum(z, (1,2,3))
        z =  (z != 0).float()
        return weight * z.view(weight.shape[0],1,1,1).expand_as(weight)

    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if not self.training:
            weight = _get_masked_weight(weight)
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)

        self._get_glasso()
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.width_mult = 1
        self.us = us

    def forward(self, input):
        if self.us[0]:
            self.in_features = make_divisible(
                self.in_features_max * self.width_mult)
        if self.us[1]:
            self.out_features = make_divisible(
                self.out_features_max * self.width_mult)
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # for tracking performance during training
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(i, affine=False)
             for i in [
                     make_divisible(
                         self.num_features_max * width_mult / ratio) * ratio
                     for width_mult in width_mult_list]
             ]
        )
        self.ratio = ratio
        self.width_mult = 1
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = make_divisible(
            self.num_features_max * self.width_mult / self.ratio) * self.ratio
        if self.width_mult in width_mult_list:
            idx = width_mult_list.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y
