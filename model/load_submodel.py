import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from utils.data_utils import scan_model_with_depth
from utils.data_utils import scan_model_dict_with_depth
from utils.model_utils import build_param_groups



def choose_params(param_groups, layer_sub_name, name_match):
    filter_params = []
    for param_group in param_groups:
        layer_name = param_group["layer_name"]
        param_name = param_group["param_name"]
        if (layer_sub_name == layer_name and name_match == "fullname") or \
            (layer_sub_name in layer_name and name_match == "subname") :
            filter_params.append(param_name)
            # logging.info(f'Detect layer, Freezed Layer: {layer_name},, sub_name:{layer_sub_name}, param name: {param_group["param_name"]} \
            #     Status: {self.freeze_status[param_group["param_name"]]}')
        else:
            pass

    return filter_params


def load_submodel(model_name, model, pretrain_model_state_dict, submodel_config):

    # named_parameters, named_modules, layers_depth = scan_model_with_depth(model)
    # param_groups = build_param_groups(named_parameters, named_modules, layers_depth)
    named_modules, layers_depth = scan_model_dict_with_depth(model, pretrain_model_state_dict)
    param_groups = build_param_groups(pretrain_model_state_dict, named_modules, layers_depth)

    if model_name in ["resnet18_v2", "resnet34_v2", "resnet50_v2"]:
        layer_sub_name_list = ["conv1", "bn1"]

        chosen_params = []
        for layer_sub_name in layer_sub_name_list:
            filter_params = choose_params(param_groups, layer_sub_name, name_match="fullname")
            chosen_params += filter_params

        part = submodel_config.split("-")[0]
        if part == "Before":
            pass
        elif part == "After":
            raise NotImplementedError
        else:
            raise NotImplementedError
        submodel_name = submodel_config.split("-")[1]
        layer = int(submodel_name[-1])

        layer_sub_name_list = []
        for layer in range(1, layer+1):
            layer_sub_name_list.append(f"layer{layer}")

        for layer_sub_name in layer_sub_name_list:
            filter_params = choose_params(param_groups, layer_sub_name, name_match="subname")
            chosen_params += filter_params

        logging.info(f"In load_submodel, chosen_params: {chosen_params}")
        model_dict =  model.state_dict()
        filter_pretrain_dict = {}
        # for layer_name in pretrain_model_state_dict.items():
        for param_name in chosen_params:
            filter_pretrain_dict[param_name] = pretrain_model_state_dict[param_name]

        # state_dict = {k:v for k,v in filter_pretrain_dict.items() if k in model_dict.keys()}
        state_dict = {k:v for k,v in filter_pretrain_dict.items()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    else:
        raise NotImplementedError
































