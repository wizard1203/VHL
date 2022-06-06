import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from loss_fn.losses import SupConLoss

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024, h=1, w=1):
        return input.view(input.size(0), size, h, w)



class DecoderLabel(nn.Module):
    def __init__(self, inplanes, z_length,
                outplanes, image_size, 
                net_config='0f2c', interpolate_mode='bilinear', widen=1):
        super(DecoderLabel, self).__init__()

        self.inplanes = inplanes
        self.z_length = z_length

        self.outplanes = outplanes
        self.image_size = image_size
        self.net_config = net_config

        self.unflat = UnFlatten()

        assert interpolate_mode in ['bilinear', 'nearest']
        self.interpolate_mode = interpolate_mode

        self.bce_loss = nn.BCELoss()

        hidden_channels = int((inplanes + self.outplanes) / 2)
        if net_config == '0f2c':
            self.decoder = nn.Sequential(
                nn.Conv2d(inplanes, int(hidden_channels * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(hidden_channels * widen)),
                nn.ReLU(),
                nn.Conv2d(int(hidden_channels * widen), self.outplanes, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
        elif net_config == '0f3c':
            self.decoder = nn.Sequential(
                nn.Conv2d(inplanes, int(hidden_channels * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(hidden_channels * widen)),
                nn.ReLU(),
                nn.Conv2d(int(hidden_channels * widen), int(hidden_channels * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(hidden_channels * widen)),
                nn.ReLU(),
                nn.Conv2d(int(hidden_channels * widen), self.outplanes, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
        elif net_config == '2f3c':

            z_dim = self.inplanes * self.z_length * self.z_length

            self.linear = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.ReLU(),
                nn.BatchNorm1d(z_dim),
                nn.Linear(z_dim, z_dim),
                nn.ReLU(),
                nn.BatchNorm1d(z_dim),
            )

            self.decoder = nn.Sequential(
                nn.Conv2d(inplanes, int(hidden_channels * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(hidden_channels * widen)),
                nn.ReLU(),
                nn.Conv2d(int(hidden_channels * widen), int(hidden_channels * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(hidden_channels * widen)),
                nn.ReLU(),
                nn.Conv2d(int(hidden_channels * widen), self.outplanes, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError


    def forward(self, prior_z):
        # logging.info(f"prior_z.shape:{prior_z.shape}. image_ori.shape:{image_ori.shape}")
        if int(self.net_config.split('f')[0]) > 0:
            prior_z = self.linear(prior_z)
        prior_z = prior_z.reshape(-1, self.inplanes, self.z_length, self.z_length)

        if self.interpolate_mode == 'bilinear':
            features = F.interpolate(prior_z, size=[self.image_size, self.image_size],
                                    mode='bilinear', align_corners=True)
        elif self.interpolate_mode == 'nearest':   # might be faster
            features = F.interpolate(prior_z, size=[self.image_size, self.image_size],
                                    mode='nearest')
        else:
            raise NotImplementedError

        fake_feat = self.decoder(features)

        return fake_feat









class Decoder(nn.Module):
    def __init__(self, inplanes, image_size, z_dim=64, h_dim=512, 
                net_config='0f2c', interpolate_mode='bilinear', widen=1):
        super(Decoder, self).__init__()

        self.image_size = image_size

        self.unflat = UnFlatten()

        assert interpolate_mode in ['bilinear', 'nearest']
        self.interpolate_mode = interpolate_mode

        self.bce_loss = nn.BCELoss()

        if net_config == '0f2c':
            self.decoder = nn.Sequential(
                nn.Conv2d(inplanes, int(12 * widen), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(12 * widen)),
                nn.ReLU(),
                nn.Conv2d(int(12 * widen), 3, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
        elif net_config == '1f2c':
            pass
            # self.fc3 = nn.Linear(z_dim, h_dim)
            # self.bn_fc_3 = nn.BatchNorm1d(h_dim)
        else:
            raise NotImplementedError


    def forward(self, features, image_ori):
        if self.interpolate_mode == 'bilinear':
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='bilinear', align_corners=True)
        elif self.interpolate_mode == 'nearest':   # might be faster
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='nearest')
        else:
            raise NotImplementedError
        decode_img = self.decoder(features)

        return self.bce_loss(decode_img, image_ori), decode_img


class AuxClassifier(nn.Module):
    def __init__(self, inplanes, net_config='1c2f', loss_mode='contrast',
                class_num=10, widen=1, feature_dim=128, sup_con_temp=0.07, device=None):
        super(AuxClassifier, self).__init__()

        # assert inplanes in [16, 32, 64]
        # assert inplanes in [16, 32, 64]
        assert net_config in ['0c1f', '0c2f', '1c1f', '1c2f', '1c3f', '2c2f']
        # assert loss_mode in ['contrast', 'cross_entropy']

        self.loss_mode = loss_mode
        self.feature_dim = feature_dim

        self.device = device

        if loss_mode == 'contrast':
            self.criterion = SupConLoss(contrast_mode='all', base_temperature=sup_con_temp, device=self.device)
            self.fc_out_channels = feature_dim
        elif loss_mode == 'SimCLR':
            self.criterion = SupConLoss(contrast_mode='all', base_temperature=sup_con_temp, device=self.device)
            self.fc_out_channels = feature_dim
        elif loss_mode == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
            self.fc_out_channels = class_num
        else:
            raise NotImplementedError

        if net_config == '0c1f':  # Greedy Supervised Learning (Greedy SL)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(inplanes, self.fc_out_channels),
            )

        if net_config == '0c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(16, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(32, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            else:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(inplanes, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '1c1f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), self.fc_out_channels),
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), self.fc_out_channels),
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), self.fc_out_channels),
                )
            else:
                self.head = nn.Sequential(
                    nn.Conv2d(inplanes, int(inplanes * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(inplanes * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(inplanes * widen), self.fc_out_channels),
                )

        if net_config == '1c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            else:
                self.head = nn.Sequential(
                    nn.Conv2d(inplanes, int(inplanes * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(inplanes * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(inplanes * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '1c3f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            else:
                self.head = nn.Sequential(
                    nn.Conv2d(inplanes, int(inplanes * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(inplanes * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(inplanes * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '2c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(32 * widen), int(32 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(64 * widen), int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(64 * widen), int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            else:
                self.head = nn.Sequential(
                    nn.Conv2d(inplanes, int(inplanes * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(inplanes * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(inplanes * widen), int(inplanes * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(inplanes * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(inplanes * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

    def forward(self, x, target, temperature=0.07):

        features = self.head(x)

        if self.loss_mode == 'contrast':
            assert features.size(1) == self.feature_dim
            features = F.normalize(features, dim=1)
            features = features.unsqueeze(1)
            loss = self.criterion(features, target, temperature=temperature)
        elif self.loss_mode == 'SimCLR':

            feat1, feat2 = torch.split(features, 
                    [features.size(0)//2, features.size(0)//2], dim=0)
            cat_feat = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
            cat_feat = F.normalize(cat_feat, dim=1)

            # logging.info(f"In auxclassifier, cat_feat.device: {cat_feat.device}")
            loss = self.criterion(cat_feat, temperature=temperature)
            # logging.info(f"feat.shape: {feat.shape}, flat_feat.shape: {flat_feat.shape}, \
            #     self.split_hidden_loss:{self.split_hidden_loss}, hidden_loss: {hidden_loss}")
            # logging.info(f"cat_feat.shape: {cat_feat.shape}, \
            #     loss:{loss}")

        elif self.loss_mode == 'cross_entropy':
            loss = self.criterion(features, target)
        else:
            raise NotImplementedError

        return loss




















