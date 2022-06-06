import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from .normal_trainer import NormalTrainer

from optim.build import create_optimizer
from loss_fn.build import create_loss
from lr_scheduler.build import create_scheduler


from optim.group_lasso_optimizer import group_lasso_decay



def create_trainer(args, device, model=None, **kwargs):

    params = None
    optimizer = create_optimizer(args, model, params=params, **kwargs)


    criterion = create_loss(args, device, **kwargs)
    lr_scheduler = create_scheduler(args, optimizer, **kwargs)
    if args.trainer_type == 'normal':
        model_trainer = NormalTrainer(model, device, criterion, optimizer, lr_scheduler, args, **kwargs)
    else:
        raise NotImplementedError

    return model_trainer



def make_initial_param_groups(args, model):
    """
        used in Gossip algorithms.
    """
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": args.wd if "bn" not in key else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]
    return params












