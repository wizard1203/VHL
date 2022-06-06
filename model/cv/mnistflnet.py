import torch.nn as nn
import torch.nn.functional as F

class MnistFLNet(nn.Module):
    def __init__(self, input_channels=1, output_dim=10):
        super(MnistFLNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.name = 'mnistflnet'

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # x = F.relu(self.fc2(x))
        # return F.softmax(x)



class MnistFLNet_feat_out(nn.Module):
    def __init__(self, input_channels=1, output_dim=10):
        super(MnistFLNet_feat_out, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.name = 'mnistflnet'

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 3136)
        feat = F.relu(self.fc1(x))
        x = self.fc2(feat)
        return x, feat








