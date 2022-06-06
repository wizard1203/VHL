import logging
import os
import sys
import numpy as np

from .message_define import MyMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_core.distributed.communication.message import Message
# from fedml_core.distributed.server.server_manager import ServerManager

from algorithms.basePS.ps_server_manager import PSServerManager

from .message_define import MyMessage

from .fedavg_server_timer import FedAVGServerTimer

class FedAVGServerManager(PSServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", perf_timer=None, metrics=None):
        super().__init__(args, aggregator, comm, rank, size, backend, perf_timer, metrics)
        # assert args.client_num_in_total == self.size - 1
        assert args.client_num_per_round == self.size - 1

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


    def run(self):
        super().run()

    def get_max_comm_round(self):
        # return self.args.max_comm_round + 1
        return self.args.max_epochs // self.args.global_epochs_per_round + 1

    # @overrid
    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.server_timer.global_comm_round_idx, self.args.client_num_in_total,
            self.args.client_num_per_round)

        global_other_params = {}

        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id, global_model_params, client_indexes[process_id - 1],
                global_other_params=global_other_params,
                time_info=self.server_timer.get_time_info_to_send())

    # override
    def choose_clients_and_send(self, global_model_params, params_type='model', global_other_params=None):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.server_timer.global_comm_round_idx, 
            self.args.client_num_in_total,
            self.args.client_num_per_round
        )

        time_info = self.server_timer.get_time_info_to_send()
        logging.debug("size = %d" % self.size)
        for receiver_id in range(1, self.size):
            if params_type == 'grad':
                self.send_message_sync_grad_to_client(
                    receiver_id, global_model_params, client_indexes[receiver_id - 1],
                    global_other_params=global_other_params,
                    time_info=time_info
                )
            elif params_type == 'model':
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, client_indexes[receiver_id - 1],
                    global_other_params=global_other_params,
                    time_info=time_info
                )

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


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                            self.handle_message_receive_model_from_client)


    def algorithm_on_handle_message_receive_model_from_client(
        self, sender_id, client_index, model_params, model_indexes, local_sample_number, client_other_params, time_info,
        local_train_tracker_info, local_test_tracker_info
    ):
        self.aggregator.add_local_trained_result(client_index, model_params, model_indexes, local_sample_number, client_other_params)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.debug("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.Failure_chance is not None and np.random.rand(1) < self.args.Failure_chance:
                logging.info("Communication Failure happens on worker: {}, Failure_chance: {}".format(
                    self.client_index, self.args.Failure_chance))
                global_model_params = self.aggregator.get_global_model_params()
            else:
                global_model_params, global_other_params, shared_params_for_simulation = self.aggregator.aggregate(
                    global_comm_round=self.server_timer.global_comm_round_idx,
                    global_outer_epoch_idx=self.server_timer.global_outer_epoch_idx,
                    tracker=self.total_train_tracker,
                    metrics=self.metrics)

            # Test on server
            self.check_and_test()

            # start the next round
            self.server_timer.past_epochs(epochs=1*self.global_epochs_per_round)
            self.server_timer.past_comm_round(comm_round=1)
            self.check_end_training()

            self.choose_clients_and_send(
                global_model_params, params_type='model', global_other_params=global_other_params)

    def algorithm_on_handle_message_receive_grad_from_client(
        self, sender_id, client_index, grad_params, grad_indexes, local_sample_number, client_other_params, time_info,
        local_train_tracker_info, local_test_tracker_info
    ):
        raise NotImplementedError



