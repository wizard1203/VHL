import logging

import torch.nn as nn
import torch.nn.functional as F

class SVCCAConvNet(nn.Module):
    def __init__(self):
        super(SVCCAConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1)      #  32 - 5 + 1 = 28
        self.conv2 = nn.Conv2d(64, 128, 5, 1)     #  28 - 5 + 1 = 24
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(3, 2, 1)       #  24 / 2 = 12
        self.conv3 = nn.Conv2d(128, 128, 3, 1)    #  12 - 3 + 1 = 10
        self.conv4 = nn.Conv2d(128, 128, 3, 1)    #  10 - 3 + 1 = 8
        self.conv5 = nn.Conv2d(128, 128, 3, 1)    #  8 - 3 + 1 = 6
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(3, 2, 1)      #  6 / 2 = 3
        self.fc1 = nn.Linear(3*3*128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.bn4 = nn.BatchNorm1d(10)

        self.name = 'SVCCAConvNet'

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # logging.info("After conv2, shape: {}".format(x.shape))
        x = self.bn1(x)
        x = self.pool1(x)
        # logging.info("After pool1, shape: {}".format(x.shape))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # logging.info("After conv5, shape: {}".format(x.shape))
        x = self.bn2(x)
        x = self.pool2(x)
        # logging.info("After pool2, shape: {}".format(x.shape))
        x = x.view(-1, 3*3*128)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.bn4(x)
        # logging.info("After bn4, shape: {}".format(x.shape))
        return F.softmax(x)













