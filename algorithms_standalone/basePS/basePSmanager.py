import copy
import logging
import os
import sys
from abc import ABC, abstractmethod

import random

import numpy as np
import torch
import wandb


from utils.perf_timer import Perf_Timer
from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.logger import Logger
from utils.data_utils import (
    get_avg_num_iterations,
    get_label_distribution,
    get_selected_clients_label_distribution
)
from utils.checkpoint import setup_checkpoint_config, save_checkpoint

# from data_preprocessing.build import load_data
from data_preprocessing.build import load_data

from timers.server_timer import ServerTimer


track_time = True


class BasePSManager(object):
    def __init__(self, device, args):
        self.device = device
        self.args = args
        # ================================================
        self._setup_datasets()
        self.perf_timer = Perf_Timer(
            verbosity_level=1 if track_time else 0,
            log_fn=Logger.log_timer
        )
        self.selected_clients = None
        self.client_list = []
        self.metrics = Metrics([1], task=self.args.task)
        # ================================================
        if self.args.instantiate_all:
            self.number_instantiated_client = self.args.client_num_in_total
        else:
            self.number_instantiated_client = self.args.client_num_per_round
        self._setup_clients()
        self._setup_server()
        # aggregator will be initianized in _setup_server()
        self.max_comm_round = self.aggregator.get_max_comm_round()
        self.global_num_iterations = self.aggregator.global_num_iterations
        # ================================================
        self.server_timer = ServerTimer(
            self.args,
            self.global_num_iterations,
            local_num_iterations_dict=None
        )
        self.total_train_tracker = RuntimeTracker(
            mode='Train',
            things_to_metric=self.metrics.metric_names,
            timer=self.server_timer,
            args=args
        )
        self.total_test_tracker = RuntimeTracker(
            mode='Test',
            things_to_metric=self.metrics.metric_names,
            timer=self.server_timer,
            args=args
        )
        # ================================================


    def _setup_datasets(self):
        # dataset = load_data(self.args, self.args.dataset)

        dataset = load_data(
                load_as="training", args=self.args, process_id=0, mode="standalone", task="federated", data_efficient_load=True,
                dirichlet_balance=False, dirichlet_min_p=None,
                dataset=self.args.dataset, datadir=self.args.data_dir,
                partition_method=self.args.partition_method, partition_alpha=self.args.partition_alpha,
                client_number=self.args.client_num_in_total, batch_size=self.args.batch_size, num_workers=self.args.data_load_num_workers,
                data_sampler=self.args.data_sampler,
                resize=self.args.dataset_load_image_size, augmentation=self.args.dataset_aug)

        [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params] = dataset
        self.other_params = other_params
        self.train_global = train_data_global
        self.test_global = test_data_global

        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.class_num = class_num

        if self.args.task in ["classification"] and \
            self.args.loss_fn in ["LDAMLoss", "FocalLoss", "local_FocalLoss", "local_LDAMLoss"]:
            self.local_cls_num_list_dict, self.total_cls_num = get_label_distribution(self.train_data_local_dict, class_num)
        else:
            self.local_cls_num_list_dict = None
            self.total_cls_num = None
        if "traindata_cls_counts" in self.other_params:
            self.traindata_cls_counts = self.other_params["traindata_cls_counts"]
            # Adding missing classes to list
            classes = list(range(self.class_num))
            for key in self.traindata_cls_counts:
                if len(classes) != len(self.traindata_cls_counts[key]):
                    # print(len(classes))
                    # print(len(train_data_cls_counts[key]))
                    add_classes = set(classes) - set(self.traindata_cls_counts[key])
                    # print(add_classes)
                    for e in add_classes:
                        self.traindata_cls_counts[key][e] = 0
        else:
            self.traindata_cls_counts = None


    def _setup_server(self):
        pass

    def _setup_clients(self):
        pass

    def check_end_epoch(self):
        return (self.server_timer.global_outer_iter_idx > 0 and \
            self.server_timer.global_outer_iter_idx % self.global_num_iterations == 0)

    def check_test_frequency(self):
        return self.server_timer.global_outer_epoch_idx % self.args.frequency_of_the_test == 0 or \
            self.server_timer.global_outer_epoch_idx == self.args.max_epochs - 1


    def check_and_test(self):
        if self.check_end_epoch():
            if self.check_test_frequency():
                if self.args.exchange_model:
                    self.test()
                else:
                    self.test_all_clients_model(
                        self.epoch, self.aggregator.model_dict,
                        tracker=self.total_test_tracker, metrics=self.metrics)
            else:
                self.total_train_tracker.reset()
                self.total_test_tracker.reset()


    def test(self):
        logging.info("################test_on_server_for_all_clients : {}".format(
            self.server_timer.global_outer_epoch_idx))
        self.aggregator.test_on_server_for_all_clients(
            self.server_timer.global_outer_epoch_idx, self.total_test_tracker, self.metrics)

        self.total_train_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)
        self.total_test_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)


    def test_all_clients_model(self, epoch, model_dict, tracker=None, metrics=None):
        logging.info("################test_on_server_for_all_clients : {}".format(epoch))
        for idx in model_dict.keys():
            self.aggregator.set_global_model_params(model_dict[idx])
            self.aggregator.test_on_server_for_all_clients(
                epoch, self.total_test_tracker, self.metrics)
        self.total_train_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)
        self.total_test_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)


    def get_init_state_kargs(self):
        self.selected_clients = [i for i in range(self.args.client_num_in_total)]
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss", "local_FocalLoss", "local_LDAMLoss"]:
            self.selected_clients_label_distribution = get_selected_clients_label_distribution(
                self.local_cls_num_list_dict, self.class_num, self.selected_clients, min_limit=1)
            init_state_kargs = {"weight": None, "selected_cls_num_list": self.selected_clients_label_distribution,
                                "local_cls_num_list_dict": self.local_cls_num_list_dict}
        else:
            init_state_kargs = {}
        return init_state_kargs


    def get_update_state_kargs(self):
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss", "local_FocalLoss", "local_LDAMLoss"]:
            self.selected_clients_label_distribution = get_selected_clients_label_distribution(
                self.local_cls_num_list_dict, self.class_num, self.selected_clients, min_limit=1)
            update_state_kargs = {"weight": None, "selected_cls_num_list": self.selected_clients_label_distribution,
                                "local_cls_num_list_dict": self.local_cls_num_list_dict}
        else:
            update_state_kargs = {}
        return update_state_kargs


    def lr_schedule(self, num_iterations, warmup_epochs):
        epochs = self.server_timer.global_outer_epoch_idx
        iterations = self.server_timer.global_outer_iter_idx

        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.aggregator.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.aggregator.trainer.lr_schedule(epochs)



    # ==============train clients and add results to aggregator ===================================
    def train(self):
        for _ in range(self.max_comm_round):

            logging.info("################Communication round : {}".format(self.server_timer.global_comm_round_idx))
            # w_locals = []

            # Note in the first round, something of global_other_params is not constructed by algorithm_train(),
            # So care about this.
            if self.server_timer.global_comm_round_idx == 0:
                named_params = self.aggregator.get_global_model_params() 
                if self.args.VHL and self.args.VHL_server_retrain:
                    self.aggregator.server_train_on_noise(max_iterations=50,
                        global_comm_round=0, move_to_gpu=True, dataset_name="Noise Data")
                    named_params = self.aggregator.trainer.get_model_params()

                params_type = 'model'
                global_other_params = {}
                shared_params_for_simulation = {}


                if self.args.VHL:



                    if self.args.VHL_label_from == "dataset":
                        if self.args.generative_dataset_shared_loader:
                            shared_params_for_simulation["train_generative_dl_dict"] = self.aggregator.trainer.train_generative_dl_dict
                            shared_params_for_simulation["test_generative_dl_dict"] = self.aggregator.trainer.test_generative_dl_dict
                            shared_params_for_simulation["train_generative_ds_dict"] = self.aggregator.trainer.train_generative_ds_dict
                            shared_params_for_simulation["test_generative_ds_dict"] = self.aggregator.trainer.test_generative_ds_dict
                            shared_params_for_simulation["noise_dataset_label_shift"] = self.aggregator.trainer.noise_dataset_label_shift
                            # These two dataloader iters are shared
                            shared_params_for_simulation["train_generative_iter_dict"] = self.aggregator.trainer.train_generative_iter_dict
                            shared_params_for_simulation["test_generative_iter_dict"] = self.aggregator.trainer.test_generative_iter_dict
                        else:
                            global_other_params["train_generative_dl_dict"] = self.aggregator.trainer.train_generative_dl_dict
                            global_other_params["test_generative_dl_dict"] = self.aggregator.trainer.test_generative_dl_dict
                            global_other_params["train_generative_ds_dict"] = self.aggregator.trainer.train_generative_ds_dict
                            global_other_params["test_generative_ds_dict"] = self.aggregator.trainer.test_generative_ds_dict
                            global_other_params["noise_dataset_label_shift"] = self.aggregator.trainer.noise_dataset_label_shift

                    if self.args.VHL_inter_domain_mapping:
                        global_other_params["VHL_mapping_matrix"] = self.aggregator.trainer.VHL_mapping_matrix

                if self.args.fed_align:
                    global_other_params["feature_align_means"] = self.aggregator.trainer.feature_align_means


                if self.args.scaffold:
                    c_global_para = self.aggregator.c_model_global.state_dict()
                    global_other_params["c_model_global"] = c_global_para

            client_indexes = self.aggregator.client_sampling(
                self.server_timer.global_comm_round_idx, self.args.client_num_in_total,
                self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            global_time_info = self.server_timer.get_time_info_to_send()
            update_state_kargs = self.get_update_state_kargs()

            named_params, params_type, global_other_params, shared_params_for_simulation = self.algorithm_train(
                client_indexes,
                named_params,
                params_type,
                global_other_params,
                update_state_kargs,
                global_time_info,
                shared_params_for_simulation
            )
        self.total_train_tracker.upload_record_to_wandb()
        self.total_test_tracker.upload_record_to_wandb()


    @abstractmethod
    def algorithm_train(self, client_indexes, named_params, params_type,
                        global_other_params,
                        update_state_kargs, global_time_info,
                        shared_params_for_simulation):
        pass


    # ===========================================================================

















