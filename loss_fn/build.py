import torch.nn as nn

from .losses import LDAMLoss, FocalLoss, SupConLoss


"""
    args.loss_fn in 
    ["nll_loss", "CrossEntropy"]

"""

def create_loss(args, device=None, **kwargs):
    if "client_index" in kwargs:
        client_index = kwargs["client_index"]
    else:
        client_index = args.client_index

    if args.loss_fn == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_fn == "nll_loss":
        loss_fn = nn.NLLLoss()
    elif args.loss_fn == "LDAMLoss":
        if "selected_cls_num_list" in kwargs and kwargs["selected_cls_num_list"] is not None:
            cls_num_list = kwargs["selected_cls_num_list"]
        else:
            raise NotImplementedError
        loss_fn = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, imbalance_beta=args.imbalance_beta, args=args)
    elif args.loss_fn == "FocalLoss":
        if "selected_cls_num_list" in kwargs and kwargs["selected_cls_num_list"] is not None:
            cls_num_list = kwargs["selected_cls_num_list"]
        else:
            raise NotImplementedError
        loss_fn = FocalLoss(cls_num_list=cls_num_list, gamma=1, imbalance_beta=args.imbalance_beta, args=args)
    elif args.loss_fn == "local_LDAMLoss":
        if "local_cls_num_list_dict" in kwargs and kwargs["local_cls_num_list_dict"] is not None:
            cls_num_list = kwargs["local_cls_num_list_dict"][client_index]
        else:
            raise NotImplementedError
        loss_fn = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, imbalance_beta=args.imbalance_beta, args=args)
    elif args.loss_fn == "local_FocalLoss":
        if "local_cls_num_list_dict" in kwargs and kwargs["local_cls_num_list_dict"] is not None:
            cls_num_list = kwargs["local_cls_num_list_dict"][client_index]
        else:
            raise NotImplementedError
        loss_fn = FocalLoss(cls_num_list=cls_num_list, gamma=1, imbalance_beta=args.imbalance_beta, args=args)
    elif args.loss_fn in ["SimCLR", "SupCon"]:
        loss_fn = SupConLoss(device=device)
    else:
        raise NotImplementedError

    return loss_fn















