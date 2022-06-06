import logging
import copy

import torch
from algorithms_standalone.basePS.client import Client

from algorithms.fedavg.fedavg_client_timer import FedAVGClientTimer

from utils.data_utils import (
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations,
    get_num_iterations,
)

from utils.checkpoint import save_images

from model.build import create_model



class FedNovaClient(Client):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                train_data_num, device, args, model_trainer, perf_timer=None, metrics=None):
        super().__init__(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                train_data_num, device, args, model_trainer, perf_timer, metrics)
        local_num_iterations_dict = {}
        local_num_iterations_dict[self.client_index] = self.local_num_iterations

        self.global_epochs_per_round = self.args.global_epochs_per_round
        local_num_epochs_per_comm_round_dict = {}
        local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round

        # override the PSClientManager's timer
        self.client_timer = FedAVGClientTimer(
            self.args,
            self.local_num_iterations,
            local_num_iterations_dict,
            self.global_epochs_per_round,
            local_num_epochs_per_comm_round_dict,
            client_index=self.client_index 
        )
        self.train_tracker.timer = self.client_timer
        self.test_tracker.timer = self.client_timer

    # override
    def lr_schedule(self, num_iterations, warmup_epochs):
        epoch = None
        iteration = None
        round_idx = self.client_timer.local_comm_round_idx 
        if self.args.sched == "no":
            pass
        else:
            if round_idx < warmup_epochs:
                # Because gradual warmup need iterations updates
                self.trainer.warmup_lr_schedule(round_idx*num_iterations)
            else:
                self.trainer.lr_schedule(round_idx)


    def fednova_train(self, round_idx=None, global_other_params=None, 
            tracker=None, metrics=None, traininig_start=False, shared_params_for_simulation=None,
            **kwargs):
        if self.args.if_get_diff:
            raise NotImplementedError
        previous_model = copy.deepcopy(self.trainer.get_model_params())
        client_other_params = {}
        self.move_to_gpu(self.device)

        if self.args.VHL:
            if self.args.VHL_label_from == "dataset":
                if self.args.generative_dataset_shared_loader:
                    self.trainer.train_generative_dl_dict = shared_params_for_simulation["train_generative_dl_dict"]
                    self.trainer.test_generative_dl_dict = shared_params_for_simulation["test_generative_dl_dict"]
                    self.trainer.train_generative_ds_dict = shared_params_for_simulation["train_generative_ds_dict"]
                    self.trainer.test_generative_ds_dict = shared_params_for_simulation["test_generative_ds_dict"]
                    self.trainer.noise_dataset_label_shift = shared_params_for_simulation["noise_dataset_label_shift"]
                    # These two dataloader iters are shared
                    self.trainer.train_generative_iter_dict = shared_params_for_simulation["train_generative_iter_dict"]
                    self.trainer.test_generative_iter_dict = shared_params_for_simulation["test_generative_iter_dict"]
                else:
                    self.trainer.train_generative_dl_dict = global_other_params["train_generative_dl_dict"]
                    self.trainer.test_generative_dl_dict = global_other_params["test_generative_dl_dict"]
                    self.trainer.train_generative_ds_dict = global_other_params["train_generative_ds_dict"]
                    self.trainer.test_generative_ds_dict = global_other_params["test_generative_ds_dict"]
                    self.trainer.noise_dataset_label_shift = global_other_params["noise_dataset_label_shift"]



            if self.args.VHL_inter_domain_mapping:
                self.trainer.set_VHL_mapping_matrix(global_other_params["VHL_mapping_matrix"])

        if self.args.fed_align:
            self.trainer.set_feature_align_means(global_other_params["feature_align_means"])
        tau = 0
        for epoch in range(self.args.global_epochs_per_round):
            self.epoch_init()
            for iteration in range(len(self.train_local)):
                train_batch_data = self.get_train_batch_data()
                loss, output, labels = \
                    self.trainer.train_one_step(
                        train_batch_data, device=self.device, args=self.args,
                        epoch=epoch, iteration=iteration,
                        tracker=tracker, metrics=metrics)
                tau = tau + 1

        a_i = (tau - self.args.momentum * (1 - pow(self.args.momentum, tau)) / (1 - self.args.momentum)) / (1 - self.args.momentum)
        global_model_para = previous_model
        net_para = self.trainer.get_model_params()
        norm_grad = copy.deepcopy(previous_model)
        for key in norm_grad:
            # logging.debug(global_model_para[key])
            # logging.debug(net_para[key])
            #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
            norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)


        if self.args.data_save_memory_mode:
            del self.train_local_iter
            # del self.trainer.train_local_iter
            self.train_local_iter = None
            self.trainer.train_local_iter = None


        if self.args.if_get_diff:
            raise NotImplementedError
            # compressed_weights_diff, model_indexes = self.get_model_diff_params(previous_model)
        else:
            pass
            # compressed_weights_diff, model_indexes = self.get_model_params()
        if self.args.record_dataframe:
            """
                Now, the client timer need to be updated by each iteration, 
                so as to record the track infomation.
            """
            pass
        else:
            if traininig_start:
                self.client_timer.past_epochs(epochs=self.global_epochs_per_round-1)
            else:
                self.client_timer.past_epochs(epochs=self.global_epochs_per_round)

        self.move_to_cpu()
        client_other_params["a_i"] = a_i
        client_other_params["norm_grad"] = norm_grad
        # return None, None, self.local_sample_number, a_i, norm_grad
        return None, None, self.local_sample_number, client_other_params, shared_params_for_simulation


    def algorithm_on_train(self, update_state_kargs, 
            client_index, named_params, params_type='model', traininig_start=False,
            shared_params_for_simulation=None):
        pass
















