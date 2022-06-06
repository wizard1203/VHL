import copy
import logging
import time

import torch
import torch.nn.functional as F

import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import platform

from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.wandb_util import wandb_log
from utils.data_utils import(
    get_data,
    filter_parameters,
    mean_std_online_estimate,
    retrieve_mean_std,
    get_tensors_norm,
    average_named_params,
    idv_average_named_params,
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
    check_device
)

from utils.tensor_buffer import (
    TensorBuffer
)



class Averager(object):
    """
        Responsible to implement average.
        There maybe some history information need to be memorized.
    """
    def __init__(self, args, model):
        self.args = args

    def get_average_weight(self, sample_num_list, avg_weight_type=None, global_outer_epoch_idx=0,
            inplace=True):
        # balance_sample_number_list = []
        average_weights_dict_list = []
        sum = 0
        inv_sum = 0 

        sample_num_list = copy.deepcopy(sample_num_list)
        # for i in range(0, len(sample_num_list)):
        #     sample_num_list[i] * np.random.random(1)

        for i in range(0, len(sample_num_list)):
            local_sample_number = sample_num_list[i]
            if avg_weight_type == 'inv_datanum':
                inv_local_sample_number = 1 / local_sample_number
                inv_sum += inv_local_sample_number
                sum = None
            elif avg_weight_type == 'inv_datanum2datanum':
                inv_local_sample_number = 1 / local_sample_number
                inv_sum += inv_local_sample_number
                sum += local_sample_number
            else:
                inv_sum = None
                sum += local_sample_number

        for i in range(0, len(sample_num_list)):
            local_sample_number = sample_num_list[i]

            if avg_weight_type == 'datanum':
                weight_by_sample_num = local_sample_number / sum
            elif avg_weight_type == 'even':
                weight_by_sample_num = 1 / len(sample_num_list)
            elif avg_weight_type == 'inv_datanum':
                inv_local_sample_number = 1 / local_sample_number
                weight_by_sample_num = inv_local_sample_number / inv_sum
            elif avg_weight_type == 'inv_datanum2datanum':
                # time_coefficient = global_comm_round / ............
                time_coefficient = global_outer_epoch_idx / self.args.max_epochs
                inv_local_sample_number = 1 / local_sample_number
                inv_weight = inv_local_sample_number / inv_sum
                normal_weight = local_sample_number / sum

                weight_by_sample_num = inv_weight*(1 - time_coefficient) + normal_weight*time_coefficient
                # Note that ``Sum w_i'' should be equal to 1.
            elif avg_weight_type == 'even2datanum':
                time_coefficient = global_outer_epoch_idx / self.args.max_epochs
                even_weight = 1 / len(sample_num_list)
                normal_weight = local_sample_number / sum

                weight_by_sample_num = even_weight*(1 - time_coefficient) + normal_weight*time_coefficient
                # Note that ``Sum w_i'' should be equal to 1.
            else:
                raise NotImplementedError

            average_weights_dict_list.append(weight_by_sample_num)

        homo_weights_list = average_weights_dict_list
        return average_weights_dict_list, homo_weights_list











