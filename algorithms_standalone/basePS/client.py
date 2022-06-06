import logging
import os
import sys
from abc import ABC, abstractmethod

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from utils.perf_timer import Perf_Timer
from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.wandb_util import wandb_log

from algorithms.basePS.ps_client_trainer import PSTrainer

from utils.data_utils import optimizer_to

from timers.client_timer import ClientTimer


class Client(PSTrainer):
    
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                train_data_num, device, args, model_trainer, perf_timer=None, metrics=None):
        super().__init__(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                train_data_num, device, args, model_trainer, perf_timer, metrics)
        self.metrics = metrics
        # ================================================
        local_num_iterations_dict = {}
        local_num_iterations_dict[self.client_index] = self.local_num_iterations
        self.client_timer = ClientTimer(
            self.args,
            self.global_num_iterations,
            local_num_iterations_dict,
            client_index=self.client_index
        )
        # ================================================
        self.train_tracker = RuntimeTracker(
            mode='Train',
            things_to_metric=self.metrics.metric_names,
            timer=self.client_timer,
            args=args
        )
        self.test_tracker = RuntimeTracker(
            mode='Test',
            things_to_metric=self.metrics.metric_names,
            timer=self.client_timer,
            args=args
        )


    def check_end_epoch(self):
        return (self.client_timer.local_outer_iter_idx > 0 and self.client_timer.local_outer_iter_idx % self.local_num_iterations == 0)

    def check_test_frequency(self):
        return self.client_timer.local_outer_epoch_idx % self.args.frequency_of_the_test == 0 \
            or self.client_timer.local_outer_epoch_idx == self.args.max_epochs - 1


    def get_max_comm_round(self):
        pass

    def move_to_cpu(self):
        if str(next(self.trainer.model.parameters()).device) == 'cpu':
            pass
        else:
            self.trainer.model = self.trainer.model.to('cpu')
            # optimizer_to(self.trainer.optimizer, 'cpu')

        if len(list(self.trainer.optimizer.state.values())) > 0:
            optimizer_to(self.trainer.optimizer, 'cpu')


    def move_to_gpu(self, device):
        if str(next(self.trainer.model.parameters()).device) == 'cpu':
            self.trainer.model = self.trainer.model.to(device)
        else:
            pass

        # logging.info(self.trainer.optimizer.state.values())
        if len(list(self.trainer.optimizer.state.values())) > 0:
            optimizer_to(self.trainer.optimizer, device)


    def lr_schedule(self, num_iterations, warmup_epochs):
        epochs = self.client_timer.local_outer_epoch_idx
        iterations = self.client_timer.local_outer_iter_idx
        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.trainer.lr_schedule(epochs)

    def train(self, update_state_kargs, global_time_info, 
            client_index, named_params, params_type='model',
            global_other_params=None,
            traininig_start=False,
            shared_params_for_simulation=None):
        self.client_timer.update_time_info(global_time_info)
        self.client_timer.past_comm_round(comm_round=1)
        time_info = self.client_timer.get_local_and_global_time_info()
        client_update_state_kargs = dict(list(update_state_kargs.items()) + list(time_info.items()))
        client_update_state_kargs["progress"] = self.client_timer.global_comm_round_idx
        self.update_state(**client_update_state_kargs)

        self.update_dataset(int(client_index),
                            self.client_timer.local_outer_epoch_idx)

        if self.args.instantiate_all:
            self.move_to_gpu(self.device)
        named_params, params_indexes, local_sample_number, other_client_params, shared_params_for_simulation = \
            self.algorithm_on_train(update_state_kargs, 
                client_index, named_params, params_type,
                global_other_params,
                traininig_start,
                shared_params_for_simulation)
        local_train_tracker_info = self.train_tracker.encode_local_info(
            client_index, if_reset=True, metrics=self.metrics)
        if self.args.instantiate_all:
            self.move_to_cpu()

        if self.args.data_save_memory_mode:
            del self.train_local_iter
            del self.trainer.train_local_iter
            self.train_local_iter = None
            self.trainer.train_local_iter = None

        # Because if not instantiate all clients, The new training instantiations will
        # make use of old history information (w.r.t. old datasets) like momentum buffer 
        # to train current clients (new datasets). 
        # Thus the performance increase, but this may be not true in real FL scenario.
        if not self.args.instantiate_all:
            self.clear_buffer()

        local_time_info = self.client_timer.get_time_info_to_send()
        return named_params, params_indexes, local_sample_number, other_client_params, \
            local_train_tracker_info, local_time_info, shared_params_for_simulation



    @abstractmethod
    def algorithm_on_train(self, update_state_kargs, 
            client_index, named_params, params_type='model',
            global_other_params=None,
            traininig_start=False,
            shared_params_for_simulation=None):
        named_params, params_indexes, local_sample_number, other_client_params = None, None, None, None
        return named_params, params_indexes, local_sample_number, other_client_params, shared_params_for_simulation











