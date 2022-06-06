import os
import logging

import torch
from torchvision.utils import save_image


def setup_checkpoint_config(args):
    # assert args.checkpoint_save is True
    save_checkpoints_config = {}
    save_checkpoints_config["model_state_dict"] = True if args.checkpoint_save_model else False
    save_checkpoints_config["optimizer_state_dict"] = True if args.checkpoint_save_optim else False
    save_checkpoints_config["train_metric_info"] = True if args.checkpoint_save_train_metric else False
    save_checkpoints_config["test_metric_info"] = True if args.checkpoint_save_test_metric else False
    save_checkpoints_config["checkpoint_root_path"] = args.checkpoint_root_path
    save_checkpoints_config["checkpoint_epoch_list"] = args.checkpoint_epoch_list
    save_checkpoints_config["checkpoint_file_name_save_list"] = args.checkpoint_file_name_save_list
    save_checkpoints_config["checkpoint_file_name_prefix"] = setup_checkpoint_file_name_prefix(args)
    return save_checkpoints_config


def setup_checkpoint_file_name_prefix(args):
    checkpoint_file_name_prefix = ""
    for i, name in enumerate(args.checkpoint_file_name_save_list):
        checkpoint_file_name_prefix += str(getattr(args, name))
        if i != len(args.checkpoint_file_name_save_list) - 1:
            checkpoint_file_name_prefix += "-"
    return checkpoint_file_name_prefix

def setup_save_checkpoint_common_name(save_checkpoints_config, extra_name=None):
    if extra_name is not None:
        checkpoint_common_name = "checkpoint-" + extra_name + "-" \
            + save_checkpoints_config["checkpoint_file_name_prefix"]
    else:
        checkpoint_common_name = "checkpoint-" \
            + save_checkpoints_config["checkpoint_file_name_prefix"]

    return checkpoint_common_name


def setup_save_checkpoint_path(save_checkpoints_config, extra_name=None, epoch="init", postfix=None):
    # if extra_name is not None:
    #     checkpoint_path = save_checkpoints_config["checkpoint_root_path"] \
    #         + "checkpoint-" + extra_name + "-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
    #         + "-epoch-"+str(epoch) + ".pth"
    # else:
    #     checkpoint_path = save_checkpoints_config["checkpoint_root_path"] \
    #         + "checkpoint-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
    #         + "-epoch-"+str(epoch) + ".pth"
    if postfix is not None:
        postfix_str = "-" + postfix
    else:
        postfix_str = ""

    checkpoint_common_name = setup_save_checkpoint_common_name(save_checkpoints_config, extra_name=extra_name)
    checkpoint_path = save_checkpoints_config["checkpoint_root_path"] + checkpoint_common_name \
        + "-epoch-"+str(epoch) + postfix_str + ".pth"
    return checkpoint_path 


def save_checkpoint(args, save_checkpoints_config, extra_name=None, epoch="init",
                    model_state_dict=None, optimizer_state_dict=None,
                    train_metric_info=None, test_metric_info=None, check_epoch_require=True,
                    postfix=None):
    if save_checkpoints_config is None:
        logging.info("WARNING: Not save checkpoints......")
        return
    if (check_epoch_require and epoch in save_checkpoints_config["checkpoint_epoch_list"]) \
        or (check_epoch_require is False):
        checkpoint_path = setup_save_checkpoint_path(save_checkpoints_config, extra_name, epoch, postfix)
        if not os.path.exists(save_checkpoints_config["checkpoint_root_path"]):
            os.makedirs(save_checkpoints_config["checkpoint_root_path"])
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict if save_checkpoints_config["optimizer_state_dict"] else None,
            'train_metric_info': train_metric_info if save_checkpoints_config["train_metric_info"] else None,
            'test_metric_info': test_metric_info if save_checkpoints_config["test_metric_info"] else None,
            }, checkpoint_path)
        logging.info("WARNING: Saving checkpoints {} at epoch {}......".format(
            checkpoint_path, epoch))
    else:
        logging.info("WARNING: Not save checkpoints......")


def save_checkpoint_without_check(args, save_checkpoints_config, extra_name=None, epoch="init",
                    model_state_dict=None, optimizer_state_dict=None,
                    train_metric_info=None, test_metric_info=None, check_epoch_require=True,
                    postfix=None):
    checkpoint_path = setup_save_checkpoint_path(save_checkpoints_config, extra_name, epoch, postfix=postfix)
    if not os.path.exists(save_checkpoints_config["checkpoint_root_path"]):
        os.makedirs(save_checkpoints_config["checkpoint_root_path"])
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict if save_checkpoints_config["optimizer_state_dict"] else None,
        'train_metric_info': train_metric_info if save_checkpoints_config["train_metric_info"] else None,
        'test_metric_info': test_metric_info if save_checkpoints_config["test_metric_info"] else None,
        }, checkpoint_path)
    logging.info("WARNING: Saving checkpoints {} at epoch {}......".format(
        checkpoint_path, epoch))


def load_checkpoint(args, save_checkpoints_config, extra_name, epoch, postfix=None):
    checkpoint_path = setup_save_checkpoint_path(save_checkpoints_config, extra_name, epoch, postfix=postfix)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logging.info(f"checkpoint['model_state_dict'].keys():\
            {list(checkpoint['model_state_dict'].keys())}")
        # model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = None
        logging.info(f"path: {checkpoint_path} Not exists!!!!!!!")
    return checkpoint, checkpoint_path


def load_checkpoint_dict(args, epoch_list_name, extra_name):

    epoch_list = getattr(args, epoch_list_name, [0]) 
    save_checkpoints_config = setup_checkpoint_config(args)
    checkpoint_with_epoch = {}
    checkpoint_with_epoch_paths = {}
    for epoch in epoch_list:
        logging.info("Getting epoch %s model ..." % (epoch))
        checkpoint, checkpoint_path = load_checkpoint(args, save_checkpoints_config, extra_name, epoch)
        checkpoint_with_epoch[epoch] = checkpoint
        checkpoint_with_epoch_paths[epoch] = checkpoint_path
    return save_checkpoints_config, checkpoint_with_epoch, checkpoint_with_epoch_paths




def save_images(args, data, nrow=8, epoch=0, extra_name=None, postfix=None):
    extra_name_str = "images-" + extra_name if extra_name is not None else "images-"
    postfix_str = "-" + postfix if postfix is not None else ""
    image_path = args.checkpoint_root_path + \
        extra_name_str + setup_checkpoint_file_name_prefix(args) + \
        "-epoch-"+str(epoch) + postfix_str + '.jpg'
    save_image(tensor=data, fp=image_path, nrow=nrow)





