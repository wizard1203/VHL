import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

# from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

from algorithms.basePS.ps_client_manager import PSClientManager


from .message_define import MyMessage
from .fedavg_client_timer import FedAVGClientTimer

class FedAVGClientManager(PSClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI", perf_timer=None, metrics=None):
        super().__init__(args, trainer, comm, rank, size, backend, perf_timer, metrics)

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
        )
        self.train_tracker.timer = self.client_timer
        self.test_tracker.timer = self.client_timer


    def run(self):
        super().run()

    def get_max_comm_round(self):
        # return self.args.max_comm_round + 1
        return self.args.max_epochs % self.args.global_epochs_per_round + 1


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                            self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                            self.handle_message_receive_model_from_server)



    def lr_schedule(self, num_iterations, warmup_epochs):
        epoch = None
        iteration = None
        round_idx = self.client_timer.local_comm_round_idx 
        if self.args.sched == "no":
            pass
        else:
            if round_idx < warmup_epochs:
                self.trainer.warmup_lr_schedule(round_idx)
            else:
                self.trainer.lr_schedule(round_idx)



    def client_train(self, client_index, named_params, params_type='model', global_other_params={}, traininig_start=False):
        self.trainer.set_model_params(named_params)
        self.trainer.update_dataset(int(client_index), self.client_timer.local_outer_epoch_idx)
        if traininig_start:
            pass
        else:
            self.client_timer.past_epochs(epochs=self.global_epochs_per_round)
        self.lr_schedule(self.global_num_iterations, self.args.warmup_epochs)
        weights_diff, model_indexes, local_sample_num = \
            self.trainer.fedavg_train(
                self.client_timer.global_comm_round_idx, 
                global_other_params,
                self.train_tracker,
                self.metrics)

        client_other_params = {}

        # will be sent to server for uploading
        train_tracker_info = self.train_tracker.encode_local_info(
            self.client_index, if_reset=True, metrics=self.metrics)
        time_info = self.client_timer.get_time_info_to_send()
        self.send_model_to_server(
            0, weights_diff, local_sample_num, model_indexes,
            client_other_params=client_other_params,
            time_info=time_info,
            train_tracker_info=train_tracker_info,
            test_tracker_info=None
        )


    def algorithm_on_handle_message_init(
        self, client_index, model_params, global_other_params, time_info,
        global_train_tracker_info, global_test_tracker_info
    ):
        self.client_train(
            client_index, 
            model_params, 
            params_type='model', 
            global_other_params=global_other_params, 
            traininig_start=True
        )


    def algorithm_on_handle_message_receive_model_from_server(
        self, client_index, model_params, global_other_params, time_info,
        global_train_tracker_info, global_test_tracker_info
    ):
        self.client_train(
            client_index, 
            model_params, 
            params_type='model', 
            global_other_params=global_other_params, 
            traininig_start=False
        )




    def algorithm_on_handle_message_receive_grad_from_server(
        self, client_index, grad_params, global_other_params, time_info,
        global_train_tracker_info, global_test_tracker_info
    ):
        raise NotImplementedError




