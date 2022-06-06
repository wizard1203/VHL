import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, input_channels=3, output_dim=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, input_channels=3, batch_norm=False):
    layers = []
    input_channels = input_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(input_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            input_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(input_channels=3, output_dim=10):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], input_channels=input_channels), output_dim=output_dim)


def vgg11_bn(input_channels=3, output_dim=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], input_channels=input_channels, batch_norm=True), output_dim=output_dim)


def vgg13(input_channels=3, output_dim=10):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], input_channels=input_channels), output_dim=output_dim)


def vgg13_bn(input_channels=3, output_dim=10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], input_channels=input_channels, batch_norm=True), output_dim=output_dim)


def vgg16(input_channels=3, output_dim=10):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], input_channels=input_channels), output_dim=output_dim)


def vgg16_bn(input_channels=3, output_dim=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], input_channels=input_channels, batch_norm=True), output_dim=output_dim)


def vgg19(input_channels=3, output_dim=10):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], input_channels=input_channels), output_dim=output_dim)


def vgg19_bn(input_channels=3, output_dim=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], input_channels=input_channels, batch_norm=True), output_dim=output_dim)


