import logging
import math
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024, h=1, w=1):
        return input.view(input.size(0), size, h, w)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar



class Cifar_Conv_decoder(nn.Module):
    def __init__(self, image_channels=3, ngf=64, h_dim=1024, z_dim=64):
        super(Cifar_Conv_decoder, self).__init__()

        self.ngf = ngf

        self.h_dim = h_dim
        self.hidden_channels = int(self.h_dim/64)
        self.z_dim = z_dim
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     Flatten()
        # )

        # self.fc1 = nn.Linear(h_dim, z_dim)
        # self.fc2 = nn.Linear(h_dim, z_dim)
        # self.fc3 = nn.Linear(z_dim, 512)
        # self.bn_fc_3 = nn.BatchNorm1d(512)
        # self.fc4 = nn.Linear(512, h_dim)
        # self.bn_fc_4 = nn.BatchNorm1d(h_dim)

        self.fc3 = nn.Linear(z_dim, h_dim)
        self.bn_fc_3 = nn.BatchNorm1d(h_dim)

        # self.ConvTrans1 = nn.ConvTranspose2d(self.hidden_channels, 128, kernel_size=3, stride=1)
        # self.bn_conv1 = nn.BatchNorm2d(32)
        # self.ConvTrans2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1)
        # self.bn_conv2 = nn.BatchNorm2d(32)
        # self.ConvTrans3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1)
        # self.bn_conv3 = nn.BatchNorm2d(32)
        # self.ConvTrans4 = nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=1)

        self.ConvTrans1 = nn.ConvTranspose2d(8, ngf, kernel_size=3, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(ngf)
        self.ConvTrans2 = nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1)
        self.bn_conv2 = nn.BatchNorm2d(ngf)
        self.ConvTrans3 = nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1)
        self.bn_conv3 = nn.BatchNorm2d(ngf)
        self.ConvTrans4 = nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1)
        self.bn_conv4 = nn.BatchNorm2d(ngf)

        self.conv_5 = nn.Conv2d(ngf, ngf, kernel_size=5, stride=1)
        self.bn_conv5 = nn.BatchNorm2d(ngf)
        self.conv_6 = nn.Conv2d(ngf, ngf, kernel_size=5, stride=1)
        self.bn_conv6 = nn.BatchNorm2d(ngf)
        self.conv_7 = nn.Conv2d(ngf, ngf, kernel_size=5, stride=1)
        self.bn_conv7 = nn.BatchNorm2d(ngf)
        self.conv_8 = nn.Conv2d(ngf, 3, kernel_size=5, stride=1)
        # bn_conv8 = nn.BatchNorm2d(3)


        self.unflat = UnFlatten()
        # upsamp = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        # upsamp = nn.Upsample(scale_factor=2, mode='linear')
        self.act = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, input):
        # h = self.encoder(x)
        # z, mu, logvar = self.bottleneck(h)

        # input = x, y
        # z = self.act(self.bn_fc_3(self.fc3(input)))
        # z = self.act(self.bn_fc_4(self.fc4(z)))

        z = self.act(self.bn_fc_3(self.fc3(input)))

        x = self.unflat(z, self.hidden_channels, 8, 8)
        # print(x.shape)
        x = self.act(self.bn_conv1(self.upsamp(self.ConvTrans1(x))))
        # print(x.shape)
        x = self.act(self.bn_conv2(self.upsamp(self.ConvTrans2(x))))
        # print(x.shape)
        x = self.act(self.bn_conv3(self.ConvTrans3(x)))
        # print(x.shape)
        x = self.act(self.bn_conv4(self.ConvTrans4(x)))
        # print(x.shape)

        x = self.act(self.bn_conv5(self.conv_5(x)))
        # print(x.shape)

        x = self.act(self.bn_conv6(self.conv_6(x)))
        # print(x.shape)

        x = self.act(self.bn_conv7(self.conv_7(x)))
        # print(x.shape)

        x = self.conv_8(x)
        # print(x.shape)

        # return torch.sigmoid(x5)
        return torch.tanh(x)



class MNIST_Conv_decoder(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=64):
        super(MNIST_Conv_decoder, self).__init__()

        self.h_dim = h_dim
        self.hidden_channels = int(self.h_dim/64)
        self.z_dim = z_dim
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     Flatten()
        # )
        
        # self.fc1 = nn.Linear(h_dim, z_dim)
        # self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, 512)
        self.bn_fc_3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, h_dim)
        self.bn_fc_4 = nn.BatchNorm1d(h_dim)

        self.ConvTrans1 = nn.ConvTranspose2d(self.hidden_channels, 128, kernel_size=3, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(128)
        self.ConvTrans2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.ConvTrans3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1)
        self.bn_conv3 = nn.BatchNorm2d(32)
        self.ConvTrans4 = nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=1)

        self.unflat = UnFlatten()
        # upsamp = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        # upsamp = nn.Upsample(scale_factor=2, mode='linear')
        self.act = nn.ReLU(True)


    def forward(self, input):
        # h = self.encoder(x)
        # z, mu, logvar = self.bottleneck(h)

        # input = x, y
        z = self.act(self.bn_fc_3(self.fc3(input)))
        z = self.act(self.bn_fc_4(self.fc4(z)))

        # x1 = self.unflat(z, 64, 4, 4)
        x1 = self.unflat(z, self.hidden_channels, 8, 8)
        # print(x1.shape)
        x2 = self.act(self.bn_conv1(self.ConvTrans1(x1)))
        # print(x2.shape)
        x3 = self.upsamp(self.act(self.bn_conv2(self.ConvTrans2(x2))))
        # print(x3.shape)
        x4 = self.act(self.bn_conv3(self.ConvTrans3(x3)))
        # print(x4.shape)
        x5 = self.ConvTrans4(x4)
        # print(x5.shape)

        # return torch.sigmoid(x5)
        return torch.tanh(x5)










class Mini_Generator_Out_32(nn.Module):
    def __init__(self, image_channels=3, ngf=64, h_dim=1024, z_dim=64):
        super(Mini_Generator_Out_32, self).__init__()

        self.ngf = ngf

        self.h_dim = h_dim
        self.hidden_channels = int(self.h_dim/64)
        self.z_dim = z_dim

        self.fc3 = nn.Linear(z_dim, h_dim)
        self.bn_fc_3 = nn.BatchNorm1d(h_dim)

        self.ConvTrans1 = nn.ConvTranspose2d(8, ngf, kernel_size=5, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(ngf)
        self.ConvTrans2 = nn.ConvTranspose2d(ngf, ngf, kernel_size=5, stride=1)
        self.bn_conv2 = nn.BatchNorm2d(ngf)
        self.ConvTrans3 = nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1)
        self.bn_conv3 = nn.BatchNorm2d(ngf)
        self.ConvTrans4 = nn.ConvTranspose2d(ngf, 3, kernel_size=3, stride=1)
        self.bn_conv4 = nn.BatchNorm2d(ngf)

        self.unflat = UnFlatten()
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        self.act = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # init.kaiming_normal(m.weight)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, input):

        z = self.act(self.bn_fc_3(self.fc3(input)))

        x = self.unflat(z, self.hidden_channels, 8, 8)
        # print(x.shape)
        x = self.act(self.bn_conv1(self.upsamp(self.ConvTrans1(x))))
        # print(x.shape)
        x = self.act(self.bn_conv2(self.ConvTrans2(x)))
        # print(x.shape)
        x = self.act(self.bn_conv3(self.ConvTrans3(x)))
        # print(x.shape)
        # x = self.act(self.bn_conv4(self.ConvTrans4(x)))
        x = self.ConvTrans4(x)
        # print(x.shape)

        # return torch.sigmoid(x5)
        return torch.tanh(x)























































































