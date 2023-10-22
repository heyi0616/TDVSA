from torch import nn
import torch.nn.functional as F
import torch

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


class VGG(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        output = self.features(x)
        pool_output = self.pool(output)
        pool_output = pool_output.squeeze(3).squeeze(2)
        pool_output = self.dropout(pool_output)
        feature_map = {"pooled_feature": pool_output, "mid_feature": output}
        return feature_map


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))
