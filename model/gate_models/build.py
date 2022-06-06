from .lenet import LeNet
from .vgg_bn import slimmingvgg as vgg11_bn
from .preact_resnet import *

from .resnet import (
    resnet50, resnet34, resnet101, DenseNet201, DenseNet121
)

def create_gate_model(args, model_name, output_dim, **kargs):
    if args.model == "lenet3":
        model = LeNet(dataset=args.dataset)
    elif args.model == "vgg":
        model = vgg11_bn(pretrained=True)
    elif args.model == "resnet18":
        model = PreActResNet18()
    elif (args.model == "resnet50") or (args.model == "resnet50_noskip"):
        if args.dataset == "cifar10":
            model = PreActResNet50(dataset=args.dataset)
        else:
            # from models.resnet import resnet50
            skip_gate = True
            if "noskip" in args.model:
                skip_gate = False

            if args.pruning_method not in [22, 40]:
                skip_gate = False
            model = resnet50(skip_gate=skip_gate)
    elif args.model == "resnet34":
        if not (args.dataset == "cifar10"):
            # from models.resnet import resnet34
            model = resnet34()
    elif "resnet101" in args.model:
        if not (args.dataset == "cifar10"):
            # from models.resnet import resnet101
            if args.dataset == "Imagenet":
                classes = 1000
            if "noskip" in args.model:
                model = resnet101(num_classes=classes, skip_gate=False)
            else:
                model = resnet101(num_classes=classes)
    elif args.model == "resnet20":
        if args.dataset == "cifar10":
            NotImplementedError("resnet20 is not implemented in the current project")
            # from models.resnet_cifar import resnet20
            # model = resnet20()
    elif args.model == "resnet152":
        model = PreActResNet152()
    elif args.model == "densenet201_imagenet":
        # from models.densenet_imagenet import DenseNet201
        model = DenseNet201(gate_types=['output_bn'], pretrained=True)
    elif args.model == "densenet121_imagenet":
        # from models.densenet_imagenet import DenseNet121
        model = DenseNet121(gate_types=['output_bn'], pretrained=True)
    else:
        print(args.model, "model is not supported")

    return model












