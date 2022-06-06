from data_preprocessing.utils.stats import get_dataset_image_size
from model.generative.unet import UNet
import logging

import torch
import torchvision.models as models

from model.linear.lr import LogisticRegression
from model.cv.cnn import CNN_DropOut
from model.cv.simplecnn import (SimpleCNN, SimpleCNNMNIST)
from model.cv.resnet_gn import resnet18
from model.cv.resnet_b import resnet20
from model.cv.mobilenet import mobilenet
from model.cv.resnet import resnet56
# from model.cv.resnetcifar import ResNet18_cifar10, ResNet50_cifar10
from model.cv.resnetcifar import ResNet_cifar
from model.cv.resnet_v2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet10
from model.cv.inceptionresnetv2 import inceptionresnetv2
from model.cv import resnet_torch
from model.cv.mobilenet_v3 import MobileNetV3
from model.cv.efficientnet import EfficientNet
from model.cv.mnistflnet import MnistFLNet, MnistFLNet_feat_out
from model.cv.cifar10flnet import Cifar10FLNet
from model.cv.svcca_conv import SVCCAConvNet
from model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow
from model.nlp import lstm as lstmpy
from model.nlp.lstman4 import create_net as LSTMAN4

from model.cv.others import (ModerateCNNMNIST, ModerateCNN)
from model.cv.vggmodel import (
    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn)
from model.cv.wrn import WideResNet

from model.cv.swd_cnn import build_SWD_CNN

vgg_dict = {
    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_bn": vgg19_bn,
}


CV_MODEL_LIST = []
RNN_MODEL_LIST = ["rnn"]


def create_model(args, model_name, output_dim, pretrained=False, device=None, **kwargs):
    logging.info("create_model. model_name = %s, output_dim = %s" %
                 (model_name, output_dim))
    model = None
    logging.info(f"model name: {model_name}")

    if args.VHL:
        if args.VHL_label_style == "extra":
            output_dim = output_dim + args.VHL_num
            logging.info(
                f"Model output dim is changed into {output_dim}, original is {output_dim - args.VHL_num}")
        else:
            pass

    if model_name in RNN_MODEL_LIST:
        pass
    else:
        image_size = get_dataset_image_size(args.dataset)

    if not args.gate_layer:
        if model_name == "lr" and args.dataset == "mnist":
            logging.info("LogisticRegression + MNIST")
            model = LogisticRegression(28 * 28, output_dim)
        elif model_name == "simple-cnn" and args.dataset in ["cifar10"]:
            logging.info("simplecnn + CIFAR10")
            model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=output_dim,
                              input_channels=args.model_input_channels)
        elif model_name == "simple-cnn-mnist" and args.dataset in ["mnist", "fmnist"]:
            logging.info("simplecnn_mnist + MNIST or FMNIST")
            model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=output_dim,
                                   input_channels=args.model_input_channels)
        elif model_name == "mnistflnet" and args.dataset in ["mnist", "fmnist", "femnist"]:
            logging.info("MnistFLNet + MNIST or FMNIST")
            if args.model_out_feature:
                model = MnistFLNet_feat_out(
                    input_channels=args.model_input_channels, output_dim=output_dim)
            else:
                model = MnistFLNet(
                    input_channels=args.model_input_channels, output_dim=output_dim)
        elif model_name == "cifar10flnet" and args.dataset == "cifar10":
            logging.info("Cifar10FLNet + CIFAR-10")
            model = Cifar10FLNet()
        elif model_name == "SVCCAConvNet" and args.dataset == "cifar10":
            logging.info("SVCCAConvNet + CIFAR-10")
            model = SVCCAConvNet()
        elif model_name == "cnn" and args.dataset == "femnist":
            logging.info("CNN + FederatedEMNIST")
            model = CNN_DropOut(False)
        elif model_name == "vgg-9":
            if args.dataset in ("mnist", 'femnist', 'fmnist'):
                model = ModerateCNNMNIST(output_dim=output_dim,
                                         input_channels=args.model_input_channels)
            elif args.dataset in ("cifar10", "cifar100", "cinic10", "svhn"):
                # print("in moderate cnn")
                model = ModerateCNN(output_dim=output_dim)
            elif args.dataset == 'celeba':
                model = ModerateCNN(output_dim=2)
        elif "vgg" in model_name:
            logging.info(f"{model_name}.........")
            # exec('model = model_name(input_channels=args.model_input_channels, output_dim=output_dim)')
            model = vgg_dict[model_name](
                input_channels=args.model_input_channels, output_dim=output_dim)
        elif model_name == "resnet18_gn" or model_name == "resnet18":
            logging.info("ResNet18_GN or resnet18")
            model = resnet18(
                pretrained=pretrained, num_classes=output_dim, group_norm=args.group_norm_num)
        elif model_name == "resnet18_v2":
            logging.info("ResNet18_v2")
            model = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                             model_input_channels=args.model_input_channels, device=device)
        elif model_name == "resnet34_v2":
            logging.info("ResNet34_v2")
            model = ResNet34(args=args, num_classes=output_dim, image_size=image_size,
                             model_input_channels=args.model_input_channels, device=device)
        elif model_name == "resnet50_v2":
            model = ResNet50(args=args, num_classes=output_dim, image_size=image_size,
                             model_input_channels=args.model_input_channels, device=device)
        elif model_name == "resnet10_v2":
            logging.info("ResNet10_v2")
            model = ResNet10(args=args, num_classes=output_dim, image_size=image_size,
                             model_input_channels=args.model_input_channels, device=device)
        elif "swdcnn" in model_name:
            logging.info(f"{model_name}")
            model = build_SWD_CNN(model_name=model_name,
                                  output_dim=output_dim, input_channels=3)
        elif model_name in ["resnet8_cifar", "resnet20_cifar", "resnet32_cifar", "resnet54_cifar"]:
            resnet_size = int(model_name.split("_")[0].split("resnet")[1])
            logging.info(f"{model_name}")
            model = ResNet_cifar(num_classes=output_dim, args=args, image_size=image_size,
                                 model_input_channels=args.model_input_channels, resnet_size=resnet_size)
        elif model_name == "resnet18_torch":
            logging.info("ResNet18_torch")
            model = resnet_torch.resnet18(pretrained=pretrained, num_classes=output_dim,
                                          args=args, model_input_channels=args.model_input_channels)
        elif model_name == "resnet50_torch":
            logging.info("ResNet50_torch")
            model = resnet_torch.resnet50(pretrained=pretrained, num_classes=output_dim,
                                          args=args, model_input_channels=args.model_input_channels)
        elif model_name == "rnn" and args.dataset == "shakespeare":
            logging.info("RNN + shakespeare")
            model = RNN_OriginalFedAvg(
                embedding_dim=args.lstm_embedding_dim, hidden_size=args.lstm_hidden_size)
        elif model_name == "rnn" and args.dataset == "fed_shakespeare":
            logging.info("RNN + fed_shakespeare")
            model = RNN_OriginalFedAvg(
                embedding_dim=args.lstm_embedding_dim, hidden_size=args.lstm_hidden_size)
        elif model_name == "lr" and args.dataset == "stackoverflow_lr":
            logging.info("lr + stackoverflow_lr")
            model = LogisticRegression(10004, output_dim)
        elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
            logging.info("CNN + stackoverflow_nwp")
            model = RNN_StackOverFlow()
        elif model_name == "mobilenet":
            model = mobilenet(num_classes=output_dim)
        # elif model_name == "resnet50-cifar10" or model_name == "resnet50-cifar100" or model_name == "resnet50-smallkernel" or model_name == "resnet50":
        #     model = ResNet50_cifar10(num_classes=output_dim, args=args)
        elif model_name == "wideres40-2":
            model = WideResNet(args=args, depth=40, num_classes=output_dim, widen_factor=2,
                               dropRate=0.0)
        elif model_name == "inceptionresnetv2":
            model = inceptionresnetv2(args=args, num_classes=output_dim, image_size=image_size,
                                      model_input_channels=args.model_input_channels, pretrained=pretrained)
        # TODO
        elif model_name == 'mobilenet_v3':
            '''model_mode \in {LARGE: 5.15M, SMALL: 2.94M}'''
            model = MobileNetV3(model_mode='LARGE', num_classes=output_dim)
        elif model_name == 'efficientnet':
            # model = EfficientNet()
            efficientnet_dict = {
                # Coefficients:   width,depth,res,dropout
                'efficientnet-b0': (1.0, 1.0, 224, 0.2),
                'efficientnet-b1': (1.0, 1.1, 240, 0.2),
                'efficientnet-b2': (1.1, 1.2, 260, 0.3),
                'efficientnet-b3': (1.2, 1.4, 300, 0.3),
                'efficientnet-b4': (1.4, 1.8, 380, 0.4),
                'efficientnet-b5': (1.6, 2.2, 456, 0.4),
                'efficientnet-b6': (1.8, 2.6, 528, 0.5),
                'efficientnet-b7': (2.0, 3.1, 600, 0.5),
                'efficientnet-b8': (2.2, 3.6, 672, 0.5),
                'efficientnet-l2': (4.3, 5.3, 800, 0.5),
            }
            # default is 'efficientnet-b0'
            model = EfficientNet.from_name(
                model_name='efficientnet-b0', num_classes=output_dim)
            # model = EfficientNet.from_pretrained(model_name='efficientnet-b0')
        elif model_name == 'lstman4':
            model = LSTMAN4(datapath=args.an4_audio_path)
        elif model_name == 'lstm':
            model = lstmpy.lstm(vocab_size=kwargs["vocab_size"], embedding_dim=args.lstm_embedding_dim,
                                batch_size=args.batch_size,
                                num_steps=args.lstm_num_steps, dp_keep_prob=0.3)
        elif model_name == 'lstmwt2':
            model = lstmpy.lstmwt2(
                vocab_size=kwargs["vocab_size"], batch_size=args.batch_size, dp_keep_prob=0.5)
        elif model_name == "unet":
            model = UNet(3)
        else:
            logging.info(f"model name is {model_name}")
            raise NotImplementedError
    else:
        raise NotImplementedError

    return model
