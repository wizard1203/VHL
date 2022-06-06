import torch.nn as nn

"""
    args.loss_fn in 
    ["nll_loss", "CrossEntropy"]

"""

def create_loss(args):
    if args.loss_fn == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_fn == "nll_loss":
        # loss_fn = nn.functional.nll_loss
        loss_fn = nn.NLLLoss()
    else:
        raise NotImplementedError

    return loss_fn

