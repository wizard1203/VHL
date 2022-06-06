import copy
import logging
import random
from re import M

import numpy as np
import torch
import wandb

from .client import FedAVGClient
from .aggregator import FedAVGAggregator

from utils.data_utils import (
    get_avg_num_iterations,
    get_label_distribution,
    get_selected_clients_label_distribution
)
from utils.checkpoint import setup_checkpoint_config, save_checkpoint


from algorithms_standalone.basePS.basePSmanager import BasePSManager
from model.build import create_model
from trainers.build import create_trainer

from algorithms.fedavg.fedavg_server_timer import FedAVGServerTimer

class FedAVGManager(BasePSManager):
    def __init__(self, device, args):
        super().__init__(device, args)

        self.global_epochs_per_round = self.args.global_epochs_per_round
        local_num_epochs_per_comm_round_dict = {}
        # local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round

        self.server_timer = FedAVGServerTimer(
            self.args,
            self.global_num_iterations,
            None,
            self.global_epochs_per_round,
            local_num_epochs_per_comm_round_dict
        )
        self.total_train_tracker.timer = self.server_timer
        self.total_test_tracker.timer = self.server_timer



    def _setup_server(self):
        logging.info("############_setup_server (START)#############")
        model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=self.device, **self.other_params)
        num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)
        init_state_kargs = self.get_init_state_kargs()
        model_trainer = create_trainer(
            self.args, self.device, model, num_iterations=num_iterations,
            train_data_num=self.train_data_num_in_total, test_data_num=self.test_data_num_in_total,
            train_data_global=self.train_global, test_data_global=self.test_global,
            train_data_local_num_dict=self.train_data_local_num_dict, train_data_local_dict=self.train_data_local_dict,
            test_data_local_dict=self.test_data_local_dict, class_num=self.class_num, other_params=self.other_params,
            server_index=0, role='server',
            **init_state_kargs
        )
        # model_trainer = create_trainer(self.args, self.device, model)
        self.aggregator = FedAVGAggregator(self.train_global, self.test_global, self.train_data_num_in_total,
                self.train_data_local_dict, self.test_data_local_dict,
                self.train_data_local_num_dict, self.args.client_num_in_total, self.device,
                self.args, model_trainer, perf_timer=self.perf_timer, metrics=self.metrics, traindata_cls_counts=self.traindata_cls_counts)

        # self.aggregator.traindata_cls_counts = self.traindata_cls_counts
        logging.info("############_setup_server (END)#############")

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        init_state_kargs = self.get_init_state_kargs()
        # for client_index in range(self.args.client_num_in_total):
        for client_index in range(self.number_instantiated_client):
            model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=self.device, **self.other_params)
            num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)
            model_trainer = create_trainer(
                self.args, self.device, model, num_iterations=num_iterations,
                train_data_num=self.train_data_num_in_total, test_data_num=self.test_data_num_in_total,
                train_data_global=self.train_global, test_data_global=self.test_global,
                train_data_local_num_dict=self.train_data_local_num_dict, train_data_local_dict=self.train_data_local_dict,
                test_data_local_dict=self.test_data_local_dict, class_num=self.class_num, other_params=self.other_params,
                client_index=client_index, role='client',
                **init_state_kargs
            )
            # model_trainer = create_trainer(self.args, self.device, model)
            c = FedAVGClient(client_index, self.train_data_local_dict, self.train_data_local_num_dict, 
                    self.test_data_local_dict, self.train_data_num_in_total,
                    self.device, self.args, model_trainer,
                    perf_timer=self.perf_timer, metrics=self.metrics)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")



    # override
    def check_end_epoch(self):
        return True

    # override
    def check_test_frequency(self):
        return ( self.server_timer.global_comm_round_idx % self.args.frequency_of_the_test == 0 \
            or self.server_timer.global_comm_round_idx == self.max_comm_round - 1)

    # override
    def check_and_test(self):
        if self.check_test_frequency():
            self.test()
        else:
            self.reset_train_test_tracker()



    def algorithm_train(self, client_indexes, named_params, params_type,
                        global_other_params,
                        update_state_kargs, global_time_info,
                        shared_params_for_simulation):
        for i, client_index in enumerate(client_indexes):

            copy_global_other_params = copy.deepcopy(global_other_params)
            if self.args.exchange_model == True:
                copy_named_model_params = copy.deepcopy(named_params)

            if self.args.instantiate_all:
                client = self.client_list[client_index]
            else:
                # WARNING! All instantiated clients are only used in current round.
                # The history information saved may cause BUGs in the realistic FL scenario.
                client = self.client_list[i]

            if global_time_info["global_time_info"]["global_comm_round_idx"] == 0:
                traininig_start = True
            else:
                traininig_start = False

            # client training.............
            model_params, model_indexes, local_sample_number, client_other_params, \
            local_train_tracker_info, local_time_info, shared_params_for_simulation = \
                    client.train(update_state_kargs, global_time_info, 
                    client_index, copy_named_model_params, params_type,
                    copy_global_other_params,
                    traininig_start=traininig_start,
                    shared_params_for_simulation=shared_params_for_simulation)

            self.total_train_tracker.decode_local_info(client_index, local_train_tracker_info)
            # self.total_test_tracker.decode_local_info(client_index, local_test_tracker_info)

            self.server_timer.update_time_info(local_time_info)
            self.aggregator.add_local_trained_result(
                client_index, model_params, model_indexes, local_sample_number, client_other_params)

        # update global weights and return them
        # global_model_params = self.aggregator.aggregate()
        global_model_params, global_other_params, shared_params_for_simulation = self.aggregator.aggregate(
            global_comm_round=self.server_timer.global_comm_round_idx,
            global_outer_epoch_idx=self.server_timer.global_outer_epoch_idx,
            tracker=self.total_train_tracker,
            metrics=self.metrics)

        params_type = 'model'
        self.check_and_test()

        save_checkpoints_config = setup_checkpoint_config(self.args) if self.args.checkpoint_save else None
        save_checkpoint(
            self.args, save_checkpoints_config, extra_name="server", 
            epoch=self.server_timer.global_outer_epoch_idx,
            model_state_dict=self.aggregator.get_global_model_params(),
            optimizer_state_dict=self.aggregator.trainer.optimizer.state_dict(),
            train_metric_info=self.total_train_tracker.get_metric_info(self.metrics),
            test_metric_info=self.total_test_tracker.get_metric_info(self.metrics),
            postfix=self.args.checkpoint_custom_name
        )
        self.server_timer.past_epochs(epochs=1*self.global_epochs_per_round)
        self.server_timer.past_comm_round(comm_round=1)

        return global_model_params, params_type, global_other_params, shared_params_for_simulation






