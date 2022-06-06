import os
import logging
import copy

from numpy.lib.function_base import kaiser

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import (
    get_tensors_norm,
    average_named_params,
    get_name_params_div,
    get_name_params_sum,
    get_name_params_difference,
    get_name_params_difference_norm,
    get_name_params_difference_abs,
    get_named_tensors_rotation,
    calculate_metric_for_named_tensors,
    get_diff_tensor_norm,
    get_tensor_rotation,
    calculate_metric_for_whole_model,
    calc_client_divergence,
    calc_client_layer_divergence
)

from utils.model_utils import (
    build_layer_params
)

from utils.checkpoint import setup_checkpoint_config, setup_save_checkpoint_path, save_checkpoint



def asign_type(tensor):
    if tensor.dim() == 4:
        tensor_type = "Conv2d"
    elif tensor.dim() == 2:
        tensor_type = "Linear"
    else:
        logging.info("tensor.shape: {}".format(
            tensor.shape
        ))
        raise NotImplementedError
    return tensor_type



class model_dif_tracker(object):
    def __init__(self, args=None, initial_weight={}):
        self.things_to_track = ["model_diff_norm"]
        self.num = 0
        self.args = args
        self.device = torch.device("cpu")
        self.initial_weight = copy.deepcopy(initial_weight)
        if (args.model_dif_seq_FO_track or args.model_dif_seq_SO_track):
            self.seq_start_weight = self.initial_weight
            self.seq_current_time = 0
            # self.model_dif_seq_FO_dict = {}
            # self.model_dif_seq_SO_dict = {}
            self.model_dif_seq_SO_acc = None
            self.model_dif_seq_SO_total = 0
            self.save_checkpoints_config = setup_checkpoint_config(args) if True else None


    def set_initial_weight(self, initial_weight):
        self.initial_weight = copy.deepcopy(initial_weight)
        if (self.args.model_dif_seq_FO_track or self.args.model_dif_seq_SO_track):
            self.seq_start_weight = self.initial_weight
            self.seq_current_time = 0
            # self.model_dif_seq_FO_dict = {}
            # self.model_dif_seq_SO_dict = {}
            self.model_dif_seq_SO_acc = None
            self.model_dif_seq_SO_total = 0
            self.save_checkpoints_config = setup_checkpoint_config(self.args) if True else None



    def check_config(self, args, **kwargs):
        if args.model_dif_track:
            pass
        else:
            return False
        return True

    def generate_record(self, args, **kwargs):
        """ Here args means the overall args, not the *args """

        info_dict = {}

        # ==================================================================  model_dif_divergence_track

        if 'model_dif_divergence_track' in kwargs["track_thing"]:
            assert "global_model_weights" in kwargs and "model_list" in kwargs and "selected_clients" in kwargs
            if args.model_dif_divergence_track:
                divergence_list, average_divergence, max_divergence, min_divergence = \
                    calc_client_divergence(kwargs["global_model_weights"],
                                            kwargs["model_list"], p=2, rotation=False)
                # for i, client in enumerate(kwargs["selected_clients"]):
                #     info_dict["model_divergence_dif_whole_norm/client"+str(client)] = divergence_list[i]
                info_dict["model_divergence_dif_whole_norm/avg"] = average_divergence
                info_dict["model_divergence_dif_whole_norm/max"] = max_divergence
                info_dict["model_divergence_dif_whole_norm/min"] = min_divergence

                # divergence_list, average_divergence, max_divergence, min_divergence = \
                #     calc_client_divergence(kwargs["global_model_weights"],
                #                             kwargs["model_list"], p=2, rotation=True)
                # for i, client in enumerate(kwargs["selected_clients"]):
                #     info_dict["model_divergence_rotation_whole/client"+str(client)] = divergence_list[i]
                # info_dict["model_divergence_rotation_whole/avg"] = average_divergence
                # info_dict["model_divergence_rotation_whole/max"] = max_divergence
                # info_dict["model_divergence_rotation_whole/min"] = min_divergence


        # ==================================================================  model_dif_divergence_track

        if 'model_layer_dif_divergence_track' in kwargs["track_thing"]:
            assert "global_model_weights" in kwargs and "model_list" in kwargs and "selected_clients" in kwargs \
                and "global_named_modules" in kwargs
            if args.model_layer_dif_divergence_track:
                global_layer_params = build_layer_params(named_parameters=kwargs["global_model_weights"], named_modules=kwargs["global_named_modules"],
                                param_types=["Conv2d","Linear","BatchNorm2d"])

                layer_params_list = []
                for i, model_num_i in enumerate(kwargs["model_list"]):
                    model_i = model_num_i[1]
                    layer_params = build_layer_params(named_parameters=model_i, named_modules=kwargs["global_named_modules"],
                                    param_types=["Conv2d","Linear","BatchNorm2d"])
                    layer_params_list.append(layer_params)

                layer_divergence_dimnorm_list, layer_average_divergence_dimnorm, \
                layer_divergence_originnorm_list, layer_average_divergence_originnorm = \
                    calc_client_layer_divergence(global_layer_params,
                                            layer_params_list, p=2, rotation=False)

                # divergence_list, layer_average_divergence, layer_max_divergence, layer_min_divergence = \
                #     calc_client_layer_divergence(global_layer_params,
                #                             layer_params_list, p=2, rotation=False)
                # for i, client in enumerate(kwargs["selected_clients"]):
                #     info_dict["model_divergence_dif_whole_norm/client"+str(client)] = divergence_list[i]

                for layer_name, average_divergence in layer_average_divergence_dimnorm.items():
                    info_dict[f"model_divergence_dif_norm_dimnorm/avg/{layer_name}"] = average_divergence

                for layer_name, average_divergence in layer_average_divergence_originnorm.items():
                    info_dict[f"model_divergence_dif_norm_scaled/avg/{layer_name}"] = average_divergence

        logging.debug('Model Dif TRACK::::   {}'.format(
            info_dict
        ))
        return info_dict


    def get_things_to_track(self):
        return self.things_to_track












