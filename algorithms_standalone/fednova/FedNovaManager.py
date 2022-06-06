import copy
import logging
import random

import numpy as np
import torch
import wandb

from .client import FedNovaClient
from .aggregator import FedNovaAggregator

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

class FedNovaManager(BasePSManager):
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
        self.aggregator = FedNovaAggregator(self.train_global, self.test_global, self.train_data_num_in_total,
                self.train_data_local_dict, self.test_data_local_dict,
                self.train_data_local_num_dict, self.args.client_num_in_total, self.device,
                self.args, model_trainer, perf_timer=self.perf_timer, metrics=self.metrics)
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
            c = FedNovaClient(client_index, self.train_data_local_dict, self.train_data_local_num_dict, 
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


    def train(self):
        for _ in range(self.max_comm_round):

            logging.debug("################Communication round : {}".format(self.server_timer.global_comm_round_idx))
            # w_locals = []

            if self.server_timer.global_comm_round_idx == 0 and \
				self.args.VHL and self.args.VHL_server_retrain:
                self.aggregator.server_train_on_noise(max_iterations=50,
                    global_comm_round=self.server_timer.global_comm_round_idx,
                    move_to_gpu=True, dataset_name="Noise Data")
                global_model_params = self.aggregator.trainer.get_model_params()
            else:
                global_model_params = self.aggregator.get_global_model_params() 


            client_indexes = self.aggregator.client_sampling(
                self.server_timer.global_comm_round_idx, self.args.client_num_in_total,
                self.args.client_num_per_round)
            logging.debug("client_indexes = " + str(client_indexes))

            a_list = {}
            d_list = {}
            n_list = {}

            global_time_info = self.server_timer.get_time_info_to_send()
            update_state_kargs = self.get_update_state_kargs()
            # for client_index in client_indexes:
            #     client = self.client_list[client_index]
            for i, client_index in enumerate(client_indexes):
                if self.args.instantiate_all:
                    client = self.client_list[client_index]
                else:
                    # WARNING! All instantiated clients are only used in current round.
                    # The history information saved may cause BUGs in the realistic FL scenario.
                    client = self.client_list[i]

                client.client_timer.update_time_info(global_time_info)
                client.client_timer.past_comm_round(comm_round=1)
                time_info = client.client_timer.get_local_and_global_time_info()
                client_update_state_kargs = dict(list(update_state_kargs.items()) + list(time_info.items()))
                client.update_state(**client_update_state_kargs)
                # client.trainer.update_state(**update_state_kargs)
                if self.args.exchange_model == True:
                    copy_global_model_params = copy.deepcopy(global_model_params)
                    client.set_model_params(copy_global_model_params)
                client.move_to_gpu(self.device)
                # train on new dataset

                client.lr_schedule(self.global_num_iterations, self.args.warmup_epochs)

                if global_time_info["global_time_info"]["global_comm_round_idx"] == 0:
                    traininig_start = True
                else:
                    traininig_start = False

                global_other_params = {}
                shared_params_for_simulation = {}
                if self.args.VHL:
                    if self.args.VHL_label_from == "dataset":
                        # TODO, this maybe improved in the future.
                        # We may dynamicly allocate noise dataset sampling of clients.
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

                model_params, model_indexes, local_sample_number, client_other_params, shared_params_for_simulation = \
                    client.fednova_train(round_idx=client.client_timer.global_comm_round_idx,
                                        global_other_params=global_other_params,
                                        tracker=client.train_tracker,
                                        metrics=client.metrics,
                                        traininig_start=traininig_start,
                                        shared_params_for_simulation=shared_params_for_simulation)
                a_i, d_i = client_other_params["a_i"], client_other_params["norm_grad"]
                local_train_tracker_info = client.train_tracker.encode_local_info(
                    client_index, if_reset=True, metrics=self.metrics)
                local_time_info = client.client_timer.get_time_info_to_send()
                self.total_train_tracker.decode_local_info(client_index, local_train_tracker_info)
                # self.total_test_tracker.decode_local_info(client_index, local_test_tracker_info)

                self.server_timer.update_time_info(local_time_info)

                client.move_to_cpu()
                a_list[client_index] = a_i
                d_list[client_index] = d_i
                n_i = local_sample_number
                n_list[client_index] = n_i

            total_n = sum(n_list.values())
            d_total_round = copy.deepcopy(global_model_params)
            for key in d_total_round:
                d_total_round[key] = 0.0

            for client_index in client_indexes:
                d_para = d_list[client_index]
                for key in d_para:
                    d_total_round[key] += d_para[key] * n_list[client_index] / total_n

            # update global model
            coeff = 0.0
            for client_index in client_indexes:
                coeff = coeff + a_list[client_index] * n_list[client_index] / total_n

            # global_model_params = global_model.state_dict()
            for key in global_model_params:
                #print(updated_model[key])
                if global_model_params[key].type() == 'torch.LongTensor':
                    global_model_params[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif global_model_params[key].type() == 'torch.cuda.LongTensor':
                    global_model_params[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    #print(updated_model[key].type())
                    #print((coeff*d_total_round[key].type()))
                    global_model_params[key] -= coeff * d_total_round[key]
            self.aggregator.set_global_model_params(global_model_params)
            if self.args.model_dif_track:
                # global_model_weights = self.trainer.get_model_params()
                if self.args.model_dif_divergence_track:
                    self.total_train_tracker.update_local_record(
                        'model_dif_track',
                        server_index=0, 
                        summary_n_samples=self.global_num_iterations*1,
                        args=self.args,
                        choose_layers=True,
                        track_thing='model_dif_divergence_track',
                        global_model_weights=global_model_params,
                        model_list=[ (n_list[i], d_list[i])  for i in client_indexes],
                        selected_clients=client_indexes,
                    )
            self.check_and_test()
            # self.server_timer.past_iterations(iterations=1)
            self.server_timer.past_comm_round(comm_round=1)
        self.total_train_tracker.upload_record_to_wandb()
        self.total_test_tracker.upload_record_to_wandb()


    def algorithm_train(self, client_indexes, named_params, params_type,
                        update_state_kargs, global_time_info, shared_params_for_simulation):
        pass




